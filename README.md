# Slidex

Slidex builds a slide deck live while you talk. Microphone audio streams to the
OpenAI Realtime API, which edits the deck by calling local tools (title slide,
new slide, bullet), and a browser page animates each change as it lands. Content
slides are illustrated automatically from Wikimedia Commons.

## Run

```bash
pipenv install
# add env vars to .env
pipenv run python slidex.py --transcribe --esp32 --web --push https://slidex-hackmit-2026-one.vercel.app/api/slides
```

Then open <http://127.0.0.1:8000>. Use `←`/`→`, `Space`, or the on-screen
buttons to move between slides; the view follows the slide being written and
otherwise stays where you left it.

Microphone capture needs the PortAudio runtime (`sudo apt-get install
libportaudio2` on Debian/Ubuntu).

## Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--microphone` | | Stream the default input device continuously. |
| `--web` | | Serve the deck UI on localhost. |
| `--web-port N` | `8000` | Port for `--web`. |
| `--push URL` | | Mirror the deck to a hosted display. Needs `SLIDEX_PUSH_TOKEN`. |
| `--transcribe` | | Transcribe continuously and revise slides on a timer (see below). |
| `--editor-seconds S` | `1.0` | With `--transcribe`, how often the current slide is revised. |
| `--editor-model M` | `gpt-4.1-mini` | With `--transcribe`, the model that writes slides. |
| `--vad` | | With `--transcribe`, let the server segment turns instead of committing on a timer. |
| `--chunk-seconds S` | `2.0` | Ceiling on audio per deck decision; a pause commits sooner. Set `0.5` for the old fixed-tick behaviour. |
| `PROMPT` | | Instead of `--microphone`, run a single text turn through the same tools. |

## Deploying the display to Vercel

The microphone (Bluetooth or otherwise) pairs to a host machine's audio stack,
and the Realtime session is a WebSocket held open for the length of a talk.
Neither can live in a request-scoped serverless function, so `slidex.py` keeps
running where the microphone is and Vercel hosts the display for everyone else.

```
 your machine                    Vercel                    viewers
 slidex.py --push  ──POST──▶  /api/slides  ◀──GET──  browser
 (mic + Realtime)                  │
                                   ▼
                                   KV
```

One function serves both directions: the capture machine POSTs the deck, the
audience GETs it.

The project deploys as-is with the Python preset: `api/*.py` become functions
and `public/` is served statically.

1. Provision a Redis-compatible KV store (Vercel KV, or Upstash from the
   marketplace). Functions share no memory, so the deck has to live outside
   them. The integration sets `KV_REST_API_URL` and `KV_REST_API_TOKEN`;
   `UPSTASH_REDIS_REST_URL` / `_TOKEN` are accepted too.
2. Set `SLIDEX_PUSH_TOKEN` to a secret of your choosing. `POST /api/slides` fails
   closed, so until this exists nothing can publish.
3. `vercel deploy`.
4. Run the capture side with the same secret:

```bash
export SLIDEX_PUSH_TOKEN=the-same-secret
pipenv run python slidex.py --microphone --web --push https://your-app.vercel.app/api/slides
```

`--web` stays useful as your own local monitor; `--push` mirrors the deck for
the audience. A publisher thread diffs the snapshot and uploads only on change,
off the Realtime event loop.

| Route | |
| --- | --- |
| `GET /` | the deck UI, from `public/index.html` |
| `GET /api/slides` | current deck; `503` if the store is unreachable |
| `POST /api/slides` | publish a deck; `Authorization: Bearer $SLIDEX_PUSH_TOKEN` |

Without a KV store the routes fall back to per-instance memory. That is fine
for `vercel dev` but will not hold a deck across invocations in production.

The capture program is called `slidex.py`, not `main.py`, on purpose: Vercel's
Python preset treats a root-level `main.py` as the app entrypoint and fails the
build because it exports no `app` or `handler`. `.vercelignore` keeps it and the
Pipfile out of the deployment anyway — the Pipfile pins `sounddevice`, which
needs the PortAudio system headers and cannot compile in the serverless image.
The `api/` routes are standard library only.

## Two ways of building the deck

The default asks the speech-to-speech model to edit the deck while it is still
listening. That is responsive, but a slow decision stops the ear, and a quiet
moment still demands an answer -- which is how invented bullets appear.

`--transcribe` splits the two. A transcription session only ever produces text;
a separate editor reads the transcript on its own clock and rewrites the current
slide. Nothing the editor does can starve the microphone, silence produces no
transcript and therefore no decision, and because each pass re-reads the
transcript the slide corrects itself instead of accumulating mistakes.

```bash
pipenv run python slidex.py --transcribe --web
```

Only the slide being written is mutable; finished slides are frozen, so the deck
does not churn behind the speaker. The title slide is the exception while the
talk is young: it names the subject of the whole talk, so it is left blank until
the speaker has said what that is, and may be retitled until three content
slides exist. A slide title is a complete phrase; when a slide has to open before
the speaker has finished naming its subject, the title ends with an ellipsis and
is completed on a later pass.

Editor calls overlap, so one slow request cannot stall the deck. Because answers
can then arrive out of order, each carries the revision it was written against:
one that a newer answer has already superseded is dropped, as is one written
against a slide that has since been closed. Measured round trip is about a
second (`gpt-4.1-mini` 1.02s, `gpt-4.1-nano` 0.78s, `gpt-4o-mini` 1.03s), so a
one second tick keeps roughly one edit in flight. Raise `--editor-seconds` to
cut cost; the editor already skips a tick when nothing new has been said. `tools/transcribe_probe.py` prints the raw
transcript stream if you want to check transcription quality on its own.

## How it fits together

- `slidex.py` — deck state and the tool handlers the model calls, a tiny HTTP
  server exposing `GET /api/slides`, the Realtime WebSocket client, and the
  optional publisher that mirrors the deck to a deployment.
- `public/index.html` — polls `/api/slides` twice a second and diffs the deck into
  the DOM, keyed by stable slide ids so inserts (the title slide goes at the
  front) don't rewrite the wrong card. The slide frame is a CSS container, so
  type and spacing scale with the slide rather than the window, and a measured
  auto-fit pass shrinks any text block that still would not fit.

Images are fetched on background threads and credited on the slide. Each search
returns ten candidates and slide N takes the Nth, skipping pictures already on
the deck, so similar slides are not all illustrated with the same top hit; when
there is no Nth result the pick is random from the ten. Nothing
that touches the network runs on the Realtime event loop: a slow lookup there
would stall every response and back the microphone up behind it.
