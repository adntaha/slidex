# Slidex

Slidex builds a slide deck live while you talk. Microphone audio streams to the
OpenAI Realtime API, which edits the deck by calling local tools (title slide,
new slide, bullet), and a browser page animates each change as it lands. Content
slides are illustrated automatically from Wikimedia Commons.

## Run

```bash
pipenv install
export OPENAI_API_KEY=sk-...
pipenv run python main.py --microphone --web
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
| `--chunk-seconds S` | `0.5` | Seconds of audio committed before each deck decision. |
| `PROMPT` | | Instead of `--microphone`, run a single text turn through the same tools. |

## How it fits together

- `main.py` — deck state and the tool handlers the model calls, a tiny HTTP
  server exposing `GET /api/slides`, and the Realtime WebSocket client.
- `index.html` — polls `/api/slides` twice a second and diffs the deck into
  the DOM, keyed by stable slide ids so inserts (the title slide goes at the
  front) don't rewrite the wrong card. The slide frame is a CSS container, so
  type and spacing scale with the slide rather than the window, and a measured
  auto-fit pass shrinks any text block that still would not fit.

Images are fetched on background threads and credited on the slide. Nothing
that touches the network runs on the Realtime event loop: a slow lookup there
would stall every response and back the microphone up behind it.
