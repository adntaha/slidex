"""A minimal, tool-first OpenAI Realtime text client.

Set OPENAI_API_KEY, then run:
    pipenv run python main.py "What time is it in UTC?"

The assistant's text is captured in ``RealtimeToolClient.assistant_text`` but is
not printed. Replace the example tools in ``TOOL_HANDLERS`` with application
actions as the app grows.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import websocket


MODEL = "gpt-realtime-2.1"
REALTIME_URL = "wss://api.openai.com/v1/realtime"
LOGGER = logging.getLogger(__name__)


def trace(message: str) -> None:
    """Emit progress immediately; useful when running a silent-text session."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def get_current_time(timezone: str = "UTC") -> dict[str, str]:
    """Example local tool. Add real application tools beside this function."""
    if timezone.upper() != "UTC":
        return {"error": "This example only supports UTC."}
    return {"timezone": "UTC", "time": datetime.now(UTC).isoformat()}


SLIDES: list[dict[str, Any]] = []
DECK_LOCK = threading.RLock()


def current_slide() -> dict[str, Any] | None:
    return SLIDES[-1] if SLIDES else None


def slide_state() -> dict[str, Any]:
    """Return enough state for the model to choose its next deck-editing action."""
    with DECK_LOCK:
        slide = current_slide()
        if slide is None:
            return {"slide_count": 0, "current_slide": None}
        return {
            "slide_count": len(SLIDES),
            "current_slide": {
                "number": len(SLIDES),
                "title": slide["title"],
                "bullets": list(slide["bullets"]),
            },
        }


def deck_snapshot() -> dict[str, Any]:
    """Create a stable copy for the local browser without exposing tool internals."""
    with DECK_LOCK:
        return {
            "slides": [
                {"number": index, "title": slide["title"], "bullets": list(slide["bullets"])}
                for index, slide in enumerate(SLIDES, start=1)
            ]
        }


def create_new_slide(title: str) -> dict[str, Any]:
    """Start a new topic in the live deck."""
    with DECK_LOCK:
        cleaned = title.strip()
        if not cleaned:
            return {"status": "ignored", "reason": "A slide title is required.", **slide_state()}

        SLIDES.append({"title": cleaned, "bullets": []})
        trace(f"NEW SLIDE #{len(SLIDES)}: {cleaned}")
        return {"status": "created", **slide_state()}


def new_bullet_point(bullet_point: str, slide_title: str | None = None) -> dict[str, Any]:
    """Keep one model-selected, standalone note from the live audio stream."""
    with DECK_LOCK:
        cleaned = bullet_point.strip().lstrip("-• \t")
        if not cleaned:
            return {"status": "ignored", "reason": "The bullet point was empty.", **slide_state()}

        if current_slide() is None:
            if not slide_title:
                return {
                    "status": "ignored",
                    "reason": "Create a titled slide before adding its first bullet.",
                    **slide_state(),
                }
            create_new_slide(slide_title)

        slide = current_slide()
        assert slide is not None
        slide["bullets"].append(cleaned)
        trace(f"BULLET #{len(slide['bullets'])}: • {cleaned}")
        return {"status": "saved", "bullet_number": len(slide["bullets"]), **slide_state()}


def update_bullet_point(bullet_number: int, bullet_point: str) -> dict[str, Any]:
    """Replace a current-slide bullet when later speech refines or corrects it."""
    with DECK_LOCK:
        slide = current_slide()
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


TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_current_time": get_current_time,
    "create_new_slide": create_new_slide,
    "new_bullet_point": new_bullet_point,
    "update_bullet_point": update_bullet_point,
}

TOOLS = [
    {
        "type": "function",
        "name": "get_current_time",
        "description": "Get the current UTC time.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Use UTC."}
            },
            "required": [],
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
            "Create one concise, standalone bullet point from a substantive "
            "idea the user just expressed in a speech segment."
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
            body = Path(__file__).with_name("index.html").read_bytes()
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


class RealtimeToolClient:
    """Runs one text turn and services local function calls synchronously."""

    def __init__(self, api_key: str, model: str = MODEL) -> None:
        self.model = model
        trace(f"Connecting to OpenAI Realtime ({model})...")
        self.socket = websocket.create_connection(
            f"{REALTIME_URL}?model={model}",
            header=[f"Authorization: Bearer {api_key}"],
            timeout=30,
        )
        self._send_lock = threading.Lock()
        self.response_idle = threading.Event()
        self.response_idle.set()
        self.assistant_text = ""
        self.tool_calls: list[dict[str, Any]] = []
        trace("WebSocket connected. Waiting for session events.")

    def send(self, event: dict[str, Any]) -> None:
        # Audio capture and response handling run on different threads.
        with self._send_lock:
            self.socket.send(json.dumps(event))
        # event_type = event["type"]
        # if event_type != "input_audio_buffer.append":
        #     trace(f"Sent client event: {event_type}")

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
                        "You are a real-time slide-deck editor. Build a coherent deck "
                        "from the speaker's ideas using tools, not visible text. Start "
                        "the first clear topic with create_new_slide, or provide "
                        "slide_title on its first new_bullet_point. Add a concise, "
                        "standalone bullet only for substantive, sufficiently complete "
                        "ideas. Do not add bullets for filler, false starts, or "
                        "repetition. When later speech corrects or meaningfully refines "
                        "a current-slide bullet, use update_bullet_point instead of "
                        "adding a duplicate. When one coherent topic is finished and "
                        "the speaker begins a genuinely new topic, call "
                        "create_new_slide with a short title before adding that topic's "
                        "bullets. Do not make a new slide merely because of a pause. "
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
        self.send({"type": "response.create", "response": {"tool_choice": "required"}})
        # trace("Requested a response; the first pass must choose a tool.")
        return self._receive_until_complete()

    def _receive_until_complete(self) -> str:
        while True:
            event = json.loads(self.socket.recv())
            event_type = event.get("type")
            # trace(f"Received server event: {event_type}")

            if event_type == "response.output_text.delta":
                # Deliberately retain, rather than display, assistant text.
                self.assistant_text += event.get("delta", "")
                # trace(f"Received hidden text delta ({len(event.get('delta', ''))} characters).")
            elif event_type == "response.done":
                calls = [
                    item
                    for item in event["response"].get("output", [])
                    if item.get("type") == "function_call"
                ]
                if not calls:
                    # trace("Response complete; no tool call requested.")
                    return self.assistant_text
                trace(f"Response complete with {len(calls)} local tool call(s).")
                for call in calls:
                    self._run_tool(call)
                self.send({"type": "response.create", "response": {"tool_choice": "auto"}})
            elif event_type == "error":
                raise RuntimeError(f"Realtime API error: {event.get('error', event)}")

    def begin_manual_audio_turn(self) -> bool:
        """Commit the buffered PCM and ask the model to evaluate that time slice."""
        if not self.response_idle.is_set():
            return False
        self.response_idle.clear()
        self.send({"type": "input_audio_buffer.commit"})
        self.send({"type": "response.create", "response": {"tool_choice": "auto"}})
        return True

    def listen_forever(self, manual_audio_turns: bool = False) -> None:
        """Process automatic VAD turns or app-scheduled microphone turns."""
        while True:
            event = json.loads(self.socket.recv())
            event_type = event.get("type")
            # trace(f"Received server event: {event_type}")

            if event_type == "response.output_text.delta":
                # Text is intentionally held for the application, not displayed.
                self.assistant_text += event.get("delta", "")
                # trace(f"Received hidden text delta ({len(event.get('delta', ''))} characters).")
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
                if calls:
                    # VAD starts the initial response. We explicitly start the
                    # follow-up only after the local tool outputs are present.
                    self.send(
                        {"type": "response.create", "response": {"tool_choice": "auto"}}
                    )
                elif manual_audio_turns:
                    self.response_idle.set()
                    trace("Audio decision complete; waiting for the next chunk.")
            elif event_type == "error":
                raise RuntimeError(f"Realtime API error: {event.get('error', event)}")

    def _run_tool(self, call: dict[str, Any]) -> None:
        name = call.get("name", "")
        arguments: dict[str, Any] = {}
        try:
            arguments = json.loads(call.get("arguments", "{}"))
            trace(f"Running local tool {name!r} with arguments: {arguments}")
            result: dict[str, Any] = TOOL_HANDLERS[name](**arguments)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
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

    audio_chunks = 0
    audio_bytes = 0
    last_reported_at = time.monotonic()
    audio_pending = threading.Event()
    stop_commits = threading.Event()

    def on_audio(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        nonlocal audio_chunks, audio_bytes, last_reported_at
        if status:
            LOGGER.warning("Microphone status: %s", status)
            trace(f"Microphone status: {status}")
        raw_audio = bytes(indata)
        client.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(raw_audio).decode("ascii"),
            }
        )
        audio_pending.set()
        audio_chunks += 1
        audio_bytes += len(raw_audio)
        now = time.monotonic()
        # if now - last_reported_at >= 1:
        #     trace(
        #         "Microphone streaming: "
        #         f"{audio_chunks} chunks, {audio_bytes} PCM bytes sent in the last second."
        #     )
        #     audio_chunks = 0
        #     audio_bytes = 0
        #     last_reported_at = now

    def commit_audio_chunks() -> None:
        while not stop_commits.wait(chunk_seconds):
            if not audio_pending.is_set():
                continue
            if not client.response_idle.is_set():
                trace("Keeping audio buffered while the previous tool decision finishes.")
                continue
            audio_pending.clear()
            trace(f"Committing up to {chunk_seconds:g} seconds of audio for a deck decision.")
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
    parser = argparse.ArgumentParser(description="Run a tool-first Realtime text turn.")
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
        "--chunk-seconds",
        type=float,
        default=0.25,
        help="Seconds of continuous microphone audio before each deck decision (default: 0.25).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this program.")

    if not args.microphone and not args.prompt:
        parser.error("provide a prompt or use --microphone")
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be greater than zero")

    web_server = start_web_server(args.web_port) if args.web else None
    client = RealtimeToolClient(api_key)
    try:
        if args.microphone:
            # trace("Starting continuous microphone mode.")
            stream_microphone(client, args.chunk_seconds)
        else:
            client.ask(args.prompt)
        LOGGER.info("Completed %d tool call(s).", len(client.tool_calls))
        trace(f"Completed {len(client.tool_calls)} tool call(s).")
    except KeyboardInterrupt:
        LOGGER.info("Stopped after %d tool call(s).", len(client.tool_calls))
        trace(f"Stopped after {len(client.tool_calls)} tool call(s).")
    finally:
        client.close()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
