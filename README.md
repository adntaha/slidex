# Slidex

Slidex builds a slide deck live while you talk. Microphone audio streams to the
OpenAI Realtime API, which edits the deck by calling local tools (title slide,
new slide, bullet), and a browser page animates each change as it lands. Content
slides are illustrated automatically from Wikimedia Commons.

## Run

```bash
pipenv install
export OPENAI_API_KEY=sk-...
pipenv run python slidex.py --microphone --web
```

Then open <http://127.0.0.1:8000>. Use `←`/`→`, `Space`, or the on-screen
buttons to move between slides; the view jumps to a slide when it is created
and otherwise stays where you left it.

Microphone capture needs the PortAudio runtime (`sudo apt-get install
libportaudio2` on Debian/Ubuntu).

## Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--microphone` | | Stream the default input device continuously. |
| `--web` | | Serve the deck UI on localhost. |
| `--web-port N` | `8000` | Port for `--web`. |
| `--push URL` | | Mirror the deck to a hosted display. Needs `SLIDEX_PUSH_TOKEN`. |
| `--chunk-seconds S` | `0.5` | Seconds of audio committed before each deck decision. |
| `PROMPT` | | Instead of `--microphone`, run a single text turn through the same tools. |

## Deploying the display to Vercel

The microphone (Bluetooth or otherwise) pairs to a host machine's audio stack,
and the Realtime session is a WebSocket held open for the length of a talk.
Neither can live in a request-scoped serverless function, so `slidex.py` keeps
running where the microphone is and Vercel hosts the display for everyone else.

```
your machine                          Vercel                   viewers
slidex.py --microphone --push  ──POST──▶ /api/push ──▶ KV ──▶ /api/slides ──▶ browser
```

The project deploys as-is with the Python preset: `api/*.py` become functions
and `public/` is served statically.

1. Provision a Redis-compatible KV store (Vercel KV, or Upstash from the
   marketplace). Functions share no memory, so the deck has to live outside
   them. The integration sets `KV_REST_API_URL` and `KV_REST_API_TOKEN`;
   `UPSTASH_REDIS_REST_URL` / `_TOKEN` are accepted too.
2. Set `SLIDEX_PUSH_TOKEN` to a secret of your choosing. `/api/push` fails
   closed, so until this exists nothing can publish.
3. `vercel deploy`.
4. Run the capture side with the same secret:

```bash
export SLIDEX_PUSH_TOKEN=the-same-secret
pipenv run python slidex.py --microphone --web --push https://your-app.vercel.app/api/push
```

`--web` stays useful as your own local monitor; `--push` mirrors the deck for
the audience. A publisher thread diffs the snapshot and uploads only on change,
off the Realtime event loop.

| Route | |
| --- | --- |
| `GET /` | the deck UI, from `public/index.html` |
| `GET /api/slides` | current deck; `503` if the store is unreachable |
| `POST /api/push` | publish a deck; `Authorization: Bearer $SLIDEX_PUSH_TOKEN` |

Without a KV store the routes fall back to per-instance memory. That is fine
for `vercel dev` but will not hold a deck across invocations in production.

The capture program is called `slidex.py`, not `main.py`, on purpose: Vercel's
Python preset treats a root-level `main.py` as the app entrypoint and fails the
build because it exports no `app` or `handler`. `.vercelignore` keeps it and the
Pipfile out of the deployment anyway — the Pipfile pins `sounddevice`, which
needs the PortAudio system headers and cannot compile in the serverless image.
The `api/` routes are standard library only.

## How it fits together

- `slidex.py` — deck state and the tool handlers the model calls, a tiny HTTP
  server exposing `GET /api/slides`, the Realtime WebSocket client, and the
  optional publisher that mirrors the deck to a deployment.
- `public/index.html` — polls `/api/slides` twice a second and diffs the deck into
  the DOM, keyed by stable slide ids so inserts (the title slide goes at the
  front) don't rewrite the wrong card. The slide frame is a CSS container, so
  type and spacing scale with the slide rather than the window, and a measured
  auto-fit pass shrinks any text block that still would not fit.

Images are fetched on background threads and credited on the slide. Nothing
that touches the network runs on the Realtime event loop: a slow lookup there
would stall every response and back the microphone up behind it.
