"""Slidex: build a slide deck live from a speaker's microphone.

Audio streams to the OpenAI Realtime API, which edits the in-memory deck by
calling the local tools in ``TOOL_HANDLERS``. A tiny HTTP server publishes the
deck to ``index.html``, which polls it and animates the changes.

Set OPENAI_API_KEY, then run:
    pipenv run python slidex.py --microphone --web
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import websocket


MODEL = "gpt-realtime-2.1"
REALTIME_URL = "wss://api.openai.com/v1/realtime"


def trace(message: str) -> None:
    """Emit progress immediately; useful when running a silent-text session."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


MAX_BULLETS_PER_SLIDE = 5
AUDIO_RATE = 24000  # Hz, 16-bit mono PCM
# A decision normally takes a second or two. These bound what happens when one
# does not: audio stops piling up, and a wedged turn eventually releases.
MAX_BUFFERED_AUDIO_SECONDS = 8.0
MAX_TURN_SECONDS = 20.0
MAX_TOOL_ROUNDS = 4
# Slides alternate sides so a run of illustrated slides has some visual rhythm.
IMAGE_PLACEMENTS = ("right", "left")
IMAGE_SEARCH_TIMEOUT = 12

SLIDES: list[dict[str, Any]] = []
DECK_LOCK = threading.RLock()
# Image lookups are HTTP calls. They must never run on the Realtime event loop:
# a slow one stalls every response, and the microphone backs up behind it.
IMAGE_WORKERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="image-search")
_LAST_SLIDE_ID = 0


def new_slide(kind: str, title: str) -> dict[str, Any]:
    """Build a slide with an id that survives the renumbering an insert causes."""
    global _LAST_SLIDE_ID
    with DECK_LOCK:
        _LAST_SLIDE_ID += 1
        return {"id": _LAST_SLIDE_ID, "kind": kind, "title": title, "bullets": []}


def current_content_slide() -> dict[str, Any] | None:
    """Return the most recent slide that can receive speaker notes."""
    for slide in reversed(SLIDES):
        if slide.get("kind") != "title":
            return slide
    return None


def slide_state() -> dict[str, Any]:
    """Return enough state for the model to choose its next deck-editing action."""
    with DECK_LOCK:
        slide = current_content_slide()
        has_title_slide = any(item.get("kind") == "title" for item in SLIDES)
        if slide is None:
            return {
                "slide_count": len(SLIDES),
                "has_title_slide": has_title_slide,
                "current_slide": None,
            }
        return {
            "slide_count": len(SLIDES),
            "has_title_slide": has_title_slide,
            "current_slide": {
                "number": SLIDES.index(slide) + 1,
                "title": slide["title"],
                "bullets": list(slide["bullets"]),
            },
        }


def deck_snapshot() -> dict[str, Any]:
    """Create a stable copy for the local browser without exposing tool internals."""
    with DECK_LOCK:
        return {
            "slides": [
                {
                    "id": slide["id"],
                    "number": index,
                    "kind": slide["kind"],
                    "title": slide["title"],
                    "bullets": list(slide["bullets"]),
                    "image": slide.get("image"),
                }
                for index, slide in enumerate(SLIDES, start=1)
            ]
        }


def create_title_slide(title: str) -> dict[str, Any]:
    """Create the deck's opening title slide once a clear topic is known."""
    with DECK_LOCK:
        cleaned = title.strip()
        if not cleaned:
            return {"status": "ignored", "reason": "A title is required.", **slide_state()}
        if any(slide.get("kind") == "title" for slide in SLIDES):
            return {
                "status": "ignored",
                "reason": "The opening title slide already exists.",
                **slide_state(),
            }
        # Recover gracefully if content arrived before the model could name the topic.
        SLIDES.insert(0, new_slide("title", cleaned))
        trace(f"TITLE SLIDE: {cleaned}")
        return {"status": "created", **slide_state()}


def append_content_slide(title: str) -> dict[str, Any]:
    with DECK_LOCK:
        slide = new_slide("content", title)
        SLIDES.append(slide)
        trace(f"NEW SLIDE #{len(SLIDES)}: {title}")
        return slide


def create_new_slide(title: str) -> dict[str, Any]:
    """Start a new topic in the live deck."""
    with DECK_LOCK:
        cleaned = title.strip()
        if not cleaned:
            return {"status": "ignored", "reason": "A slide title is required.", **slide_state()}
        append_content_slide(cleaned)
        return {"status": "created", **slide_state()}


def new_bullet_point(bullet_point: str, slide_title: str | None = None) -> dict[str, Any]:
    """Keep one model-selected, standalone note from the live audio stream."""
    with DECK_LOCK:
        cleaned = bullet_point.strip().lstrip("-• \t")
        if not cleaned:
            return {"status": "ignored", "reason": "The bullet point was empty.", **slide_state()}

        if current_content_slide() is None:
            if not slide_title:
                return {
                    "status": "ignored",
                    "reason": "Create a content slide before adding its first bullet.",
                    **slide_state(),
                }
            create_new_slide(slide_title)

        slide = current_content_slide()
        assert slide is not None
        continued = len(slide["bullets"]) >= MAX_BULLETS_PER_SLIDE
        if continued:
            title = slide["title"]
            if not title.lower().endswith("(continued)"):
                title = f"{title} (continued)"
            slide = append_content_slide(title)
        slide["bullets"].append(cleaned)
        trace(f"BULLET #{len(slide['bullets'])}: • {cleaned}")
        # Illustrate each content slide once, in the background. The tool result
        # does not wait for it; the browser poll picks the image up when it lands.
        if not slide.get("image_requested"):
            slide["image_requested"] = True
            IMAGE_WORKERS.submit(load_slide_image, slide, f"{slide['title']} {cleaned}")
        return {
            "status": "saved",
            "bullet_number": len(slide["bullets"]),
            "continued": continued,
            **slide_state(),
        }


def update_bullet_point(bullet_number: int, bullet_point: str) -> dict[str, Any]:
    """Replace a current-slide bullet when later speech refines or corrects it."""
    with DECK_LOCK:
        slide = current_content_slide()
        cleaned = bullet_point.strip().lstrip("-• \t")
        if slide is None:
            return {"status": "ignored", "reason": "There is no active slide.", **slide_state()}
        if not 1 <= bullet_number <= len(slide["bullets"]):
            return {
                "status": "ignored",
                "reason": f"Bullet {bullet_number} does not exist on the current slide.",
                **slide_state(),
            }
        if not cleaned:
            return {"status": "ignored", "reason": "The replacement bullet was empty.", **slide_state()}

        old_bullet = slide["bullets"][bullet_number - 1]
        slide["bullets"][bullet_number - 1] = cleaned
        trace(f"UPDATED BULLET #{bullet_number}: • {old_bullet} → {cleaned}")
        return {"status": "updated", "bullet_number": bullet_number, **slide_state()}


def find_image(query: str) -> dict[str, str] | None:
    """Return one reusable Commons image. Blocking: never call on the event loop."""
    cleaned = query.strip()
    if not cleaned:
        return None
    params = urlencode(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": cleaned,
            "gsrnamespace": "6",
            "gsrlimit": "8",
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": "1600",
        }
    )
    request = Request(
        f"https://commons.wikimedia.org/w/api.php?{params}",
        headers={"User-Agent": "Slidex/1.0 (local presentation tool)"},
    )
    try:
        with urlopen(request, timeout=IMAGE_SEARCH_TIMEOUT) as response:  # noqa: S310 - fixed Wikimedia endpoint
            pages = json.load(response).get("query", {}).get("pages", {}).values()
    except (OSError, ValueError) as exc:
        trace(f"IMAGE SEARCH FAILED: {exc}")
        return None

    for page in pages:
        info = next(iter(page.get("imageinfo", [])), {})
        image_url = info.get("thumburl") or info.get("url")
        # File search also returns audio, video, and PDFs; those have no usable thumbnail.
        if not image_url or not info.get("mime", "").startswith("image/"):
            continue
        title = page.get("title", "")
        trace(f"IMAGE FOUND: {cleaned}")
        return {
            "url": image_url,
            "alt": title.removeprefix("File:"),
            "source": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
        }
    trace(f"IMAGE NOT FOUND: {cleaned}")
    return None


def load_slide_image(slide: dict[str, Any], query: str) -> None:
    """Background worker: illustrate one slide, or quietly leave it text-only."""
    image = find_image(query)
    if image is None:
        return
    with DECK_LOCK:
        if slide not in SLIDES or "image" in slide:
            return
        rank = [item for item in SLIDES if item["kind"] == "content"].index(slide)
        placement = IMAGE_PLACEMENTS[rank % len(IMAGE_PLACEMENTS)]
        slide["image"] = {**image, "placement": placement}
        trace(f"IMAGE PLACED on slide {slide['id']} / {placement}")


TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "create_title_slide": create_title_slide,
    "create_new_slide": create_new_slide,
    "new_bullet_point": new_bullet_point,
    "update_bullet_point": update_bullet_point,
}

TOOLS = [
    {
        "type": "function",
        "name": "create_title_slide",
        "description": (
            "Create the opening title slide once the speaker's initial topic is clear. "
            "Call this only when has_title_slide is false; do not call it again until the deck resets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A concise presentation title that captures the speaker's topic.",
                }
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_new_slide",
        "description": (
            "Start a new titled slide when the speaker begins a new topic or moves on to a new subtopic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A short, specific title for the new slide.",
                }
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "new_bullet_point",
        "description": (
            "Create one concise, standalone bullet only from a clearly heard, explicit "
            "statement by the main speaker. Do not infer missing facts or themes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bullet_point": {
                    "type": "string",
                    "description": (
                        "A concise factual note in the user's intended meaning. "
                        "Do not include a bullet marker."
                    ),
                },
                "slide_title": {
                    "type": ["string", "null"],
                    "description": (
                        "Set only when this is the first bullet and no slide exists; "
                        "it becomes the first slide's title."
                    ),
                },
            },
            "required": ["bullet_point"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_bullet_point",
        "description": (
            "Replace an existing bullet on the current slide when later speech "
            "corrects, refines, or substantially improves that same point."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bullet_number": {
                    "type": "integer",
                    "description": "The one-based number of the current-slide bullet to replace.",
                },
                "bullet_point": {
                    "type": "string",
                    "description": "The complete replacement wording without a bullet marker.",
                },
            },
            "required": ["bullet_number", "bullet_point"],
            "additionalProperties": False,
        },
    },
]


class DeckRequestHandler(BaseHTTPRequestHandler):
    """Serve the local deck UI and its intentionally tiny JSON API."""

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/":
            body = (Path(__file__).parent / "public" / "index.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
        elif self.path == "/api/slides":
            body = json.dumps(deck_snapshot()).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found\n")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Browser polling should not swamp the useful microphone/tool trace output.
        return


def start_web_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), DeckRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="deck-web-server").start()
    trace(f"Deck display ready at http://127.0.0.1:{port}")
    return server


def start_deck_publisher(endpoint: str, token: str, interval: float = 0.4) -> threading.Event:
    """Mirror the deck to a hosted display for viewers who are not on this machine.

    Polls the local snapshot and POSTs only when it changes. This runs on its own
    thread on purpose: the upload must never sit in front of the Realtime event
    loop, and coalescing here keeps a burst of edits to a single request.
    """
    stop = threading.Event()

    def publish() -> None:
        published: bytes | None = None
        while not stop.wait(interval):
            body = json.dumps(deck_snapshot()).encode()
            if body == published:
                continue
            request = Request(
                endpoint,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            )
            try:
                with urlopen(request, timeout=10) as response:  # noqa: S310 - operator-supplied endpoint
                    response.read()
            except OSError as exc:
                trace(f"PUSH FAILED: {exc}")
                continue
            published = body

    threading.Thread(target=publish, daemon=True, name="deck-publisher").start()
    trace(f"Publishing the deck to {endpoint}")
    return stop


class RealtimeToolClient:
    """Holds one Realtime session and services its local function calls."""

    def __init__(self, api_key: str, model: str = MODEL) -> None:
        self.model = model
        trace(f"Connecting to OpenAI Realtime ({model})...")
        self.socket = websocket.create_connection(
            f"{REALTIME_URL}?model={model}",
            header=[f"Authorization: Bearer {api_key}"],
            timeout=30,
        )
        # The handshake should be quick, but a live session can legitimately go
        # quiet for longer than any fixed read timeout.
        self.socket.settimeout(None)
        self._send_lock = threading.Lock()
        self.response_idle = threading.Event()
        self.response_idle.set()
        self.turn_started_at = 0.0
        self.assistant_text = ""
        self.tool_calls: list[dict[str, Any]] = []
        trace("WebSocket connected. Waiting for session events.")

    def send(self, event: dict[str, Any]) -> None:
        # Audio capture and response handling run on different threads.
        with self._send_lock:
            self.socket.send(json.dumps(event))

    def request_response(self, tool_choice: str = "auto") -> None:
        self.send({"type": "response.create", "response": {"tool_choice": tool_choice}})

    def configure(self, manual_audio_turns: bool = False) -> None:
        turn_detection: dict[str, Any] | None
        if manual_audio_turns:
            turn_detection = None
            trace("Configuring text-only output, local tools, and manual audio turns...")
        else:
            turn_detection = {
                "type": "semantic_vad",
                "eagerness": "high",
                "create_response": True,
                "interrupt_response": True,
            }
            trace("Configuring text-only output, local tools, and eager semantic VAD...")
        self.send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "output_modalities": ["text"],
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": turn_detection,
                        }
                    },
                    "instructions": (
                        "You are a conservative, evidence-only real-time slide-deck editor. "
                        "Build a coherent deck only from the main speaker's clearly heard, "
                        "explicit ideas using tools, not visible text. Never invent, assume, "
                        "or add a theme, fact, entity, relationship, or motivation the speaker "
                        "did not state. Background conversations, television, ambient speech, "
                        "noise, partial phrases, and uncertain audio are not deck content. If "
                        "you are not highly confident the statement is from the main speaker and "
                        "can be recorded without adding meaning, make no tool call. Do not "
                        "reinterpret a statement through the lens of the current slide's topic. "
                        "As soon as the opening topic is clear, first call create_title_slide with "
                        "a concise presentation title. Then, when there is a substantive "
                        "point, create a separate content slide with create_new_slide "
                        "before adding bullets. Do not make the title slide from a greeting, "
                        "filler, or an unclear fragment. Call create_title_slide exactly "
                        "once per deck: after its tool result reports has_title_slide true, "
                        "never call it again unless the deck has been reset. Add a concise, "
                        "standalone bullet only for substantive, sufficiently complete "
                        "ideas. Do not add bullets for filler, false starts, or "
                        "repetition. When later speech corrects or meaningfully refines "
                        "a current-slide bullet, use update_bullet_point instead of "
                        "adding a duplicate. When one coherent topic is finished and "
                        "the speaker begins a genuinely new topic, call "
                        "create_new_slide with a short title before adding that topic's "
                        "bullets. Treat a clear, explicitly stated change of subject, entity, timeframe, or "
                        "question as a new topic; do not force it into the current slide or "
                        "rewrite a current bullet merely because it was the latest one. "
                        "Keep content slides to five bullets maximum. If the "
                        "same topic needs another point after five bullets, start a new "
                        "slide titled with the previous title plus ' (continued)'. Do not "
                        "make a new slide merely because of a pause. "
                        "Slides are illustrated automatically; there is no image tool to call. "
                        "Prefer tool calls over text responses."
                    ),
                    "tools": TOOLS,
                    "tool_choice": "auto",
                },
            }
        )

    def ask(self, prompt: str) -> str:
        # trace(f"Submitting text prompt ({len(prompt)} characters).")
        self.configure()
        self.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            }
        )
        # Require a tool on the first pass. After a tool result, ``auto`` lets
        # the model either call another tool or produce its hidden text answer.
        self.request_response("required")
        while self._finish_response():
            self.request_response()
        return self.assistant_text

    def begin_manual_audio_turn(self) -> bool:
        """Commit the buffered PCM and ask the model to evaluate that time slice."""
        if not self.response_idle.is_set():
            return False
        self.response_idle.clear()
        self.turn_started_at = time.monotonic()
        self.send({"type": "input_audio_buffer.commit"})
        self.request_response()
        return True

    def release_stalled_turn(self) -> bool:
        """Let audio flow again if a turn never reported back."""
        if self.response_idle.is_set() or time.monotonic() - self.turn_started_at < MAX_TURN_SECONDS:
            return False
        trace(f"No response after {MAX_TURN_SECONDS:g}s; abandoning the turn and resuming audio.")
        self.response_idle.set()
        return True

    def listen_forever(self, manual_audio_turns: bool = False) -> None:
        """Process automatic VAD turns or app-scheduled microphone turns."""
        rounds = 0
        while True:
            if self._finish_response() and rounds < MAX_TOOL_ROUNDS:
                # VAD (or the audio committer) starts the initial response; the
                # follow-up must wait until the local tool outputs are present.
                rounds += 1
                self.request_response()
                continue
            if rounds >= MAX_TOOL_ROUNDS:
                trace(f"Stopping after {rounds} chained tool rounds; waiting for new audio.")
            rounds = 0
            if manual_audio_turns:
                self.response_idle.set()
                trace("Audio decision complete; waiting for the next chunk.")

    def _finish_response(self) -> bool:
        """Block until a response completes; return whether it ran any local tools."""
        while True:
            event = json.loads(self.socket.recv())
            event_type = event.get("type")

            if event_type == "response.output_text.delta":
                # Deliberately retain, rather than display, assistant text.
                self.assistant_text += event.get("delta", "")
            elif event_type == "input_audio_buffer.speech_started":
                trace("VAD: speech detected; continuing to stream microphone audio.")
            elif event_type == "input_audio_buffer.speech_stopped":
                trace("VAD: speech boundary detected; model may now start a tool-using response.")
            elif event_type == "response.done":
                calls = [
                    item
                    for item in event["response"].get("output", [])
                    if item.get("type") == "function_call"
                ]
                trace(f"Response complete with {len(calls)} local tool call(s).")
                for call in calls:
                    self._run_tool(call)
                return bool(calls)
            elif event_type == "error":
                # A failed request must not end the talk, and must not leave the
                # audio committer waiting on a response.done that never comes.
                trace(f"Realtime API error: {event.get('error', event)}")
                return False

    def _run_tool(self, call: dict[str, Any]) -> None:
        name = call.get("name", "")
        arguments: dict[str, Any] = {}
        handler = TOOL_HANDLERS.get(name)
        try:
            arguments = json.loads(call.get("arguments") or "{}")
            trace(f"Running local tool {name!r} with arguments: {arguments}")
            if handler is None:
                raise LookupError("no such tool")
            result: dict[str, Any] = handler(**arguments)
        except Exception as exc:  # noqa: BLE001 - report the failure to the model, keep the session alive
            result = {"error": f"{name or 'unknown tool'} failed: {exc}"}
        trace(f"Tool {name!r} finished with result: {result}")

        self.tool_calls.append({"name": name, "arguments": arguments, "result": result})
        self.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(result),
                },
            }
        )

    def close(self) -> None:
        trace("Closing Realtime WebSocket.")
        self.socket.close()


def stream_microphone(client: RealtimeToolClient, chunk_seconds: float) -> None:
    """Send PCM continuously and manually ask for a deck decision every few seconds."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Install Python dependencies first: pipenv install") from exc
    except OSError as exc:
        raise SystemExit(
            "Microphone streaming requires the system PortAudio runtime. "
            "On Ubuntu/Debian run: sudo apt-get install libportaudio2"
        ) from exc

    stop_commits = threading.Event()
    buffered_bytes = 0
    dropping = False
    buffer_lock = threading.Lock()
    max_buffered_bytes = int(MAX_BUFFERED_AUDIO_SECONDS * AUDIO_RATE * 2)

    def on_audio(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        nonlocal buffered_bytes, dropping
        if status:
            trace(f"Microphone status: {status}")
        raw_audio = bytes(indata)
        with buffer_lock:
            # Once a decision is slow, buffering everything said meanwhile only
            # produces one huge catch-up turn. Cap it and drop the excess.
            if buffered_bytes + len(raw_audio) > max_buffered_bytes:
                if not dropping:
                    dropping = True
                    trace(f"Buffer full at {MAX_BUFFERED_AUDIO_SECONDS:g}s; dropping audio until the deck catches up.")
                return
            buffered_bytes += len(raw_audio)
        client.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(raw_audio).decode("ascii"),
            }
        )

    def commit_audio_chunks() -> None:
        nonlocal buffered_bytes, dropping
        while not stop_commits.wait(chunk_seconds):
            client.release_stalled_turn()
            with buffer_lock:
                pending = buffered_bytes
            if not pending:
                continue
            if not client.response_idle.is_set():
                continue
            with buffer_lock:
                buffered_bytes = 0
                dropping = False
            trace(f"Committing {pending / (AUDIO_RATE * 2):.1f}s of audio for a deck decision.")
            client.begin_manual_audio_turn()

    client.configure(manual_audio_turns=True)
    trace("Opening the default 24 kHz mono PCM microphone device...")
    with sd.RawInputStream(
        samplerate=24000,
        blocksize=2400,  # 100 ms chunks
        channels=1,
        dtype="int16",
        callback=on_audio,
    ):
        trace(
            f"Microphone is live. A deck decision runs every {chunk_seconds:g} seconds; "
            "assistant text remains hidden. Press Ctrl-C to stop."
        )
        threading.Thread(target=commit_audio_chunks, daemon=True, name="audio-commit-loop").start()
        try:
            client.listen_forever(manual_audio_turns=True)
        finally:
            stop_commits.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a slide deck live from speech.")
    parser.add_argument("prompt", nargs="?", help="Optional text supplied to the Realtime session.")
    parser.add_argument(
        "--microphone",
        action="store_true",
        help="Continuously stream the default microphone; text output remains hidden.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve the live deck at http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8000,
        help="Localhost port for --web (default: 8000).",
    )
    parser.add_argument(
        "--push",
        metavar="URL",
        help=(
            "Also publish the deck to a hosted display, e.g. "
            "https://your-app.vercel.app/api/push. Requires SLIDEX_PUSH_TOKEN."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.5,
        help="Seconds of continuous microphone audio before each deck decision (default: 0.5).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this program.")

    if not args.microphone and not args.prompt:
        parser.error("provide a prompt or use --microphone")
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be greater than zero")

    push_token = os.environ.get("SLIDEX_PUSH_TOKEN", "")
    if args.push and not push_token:
        parser.error("--push needs SLIDEX_PUSH_TOKEN set to the same secret as the deployment")

    web_server = start_web_server(args.web_port) if args.web else None
    stop_publisher = start_deck_publisher(args.push, push_token) if args.push else None
    client = RealtimeToolClient(api_key)
    try:
        if args.microphone:
            # trace("Starting continuous microphone mode.")
            stream_microphone(client, args.chunk_seconds)
        else:
            client.ask(args.prompt)
        trace(f"Completed {len(client.tool_calls)} tool call(s).")
    except KeyboardInterrupt:
        trace(f"Stopped after {len(client.tool_calls)} tool call(s).")
    finally:
        client.close()
        IMAGE_WORKERS.shutdown(wait=False, cancel_futures=True)
        if stop_publisher is not None:
            stop_publisher.set()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()


if __name__ == "__main__":
    main()
