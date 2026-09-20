"""Probe the Realtime transcription session before committing to a rewrite.

This answers one question the docs leave ambiguous: do transcript deltas arrive
*while* you are speaking, or only when a turn ends?

It matters because the proposed architecture runs the deck editor on its own
timer against an accumulating transcript. If deltas stream continuously, turn
detection stops being load-bearing and VAD can be wrong without the deck
suffering. If deltas only flush at turn boundaries, turn detection is back on
the critical path and the whole design needs rethinking.

Run it twice:

    pipenv run python tools/transcribe_probe.py                 # manual commits, no VAD
    pipenv run python tools/transcribe_probe.py --vad           # server turn detection

Talk normally for the first, then run it again and talk without pausing. Nothing
here touches slidex.py.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import threading
import time
from typing import Any

import websocket

REALTIME_URL = "wss://api.openai.com/v1/realtime"
# Documented transcription models, most recent first. Availability varies by key,
# which is the other thing this probe is here to find out.
CANDIDATE_MODELS = [
    "gpt-live-transcribe",
    "gpt-transcribe",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-realtime-whisper",
    "whisper-1",
]
AUDIO_RATE = 24000
BLOCK_FRAMES = 2400  # 100 ms
# How soon after a delta its turn must close for that delta to count as having
# streamed ahead of the commit, rather than been dumped after one.
AHEAD_WINDOW = 4.0


def stamp(started: float) -> str:
    return f"[{time.monotonic() - started:6.2f}s]"


def session_payload(model: str, use_vad: bool) -> dict[str, Any]:
    turn_detection = {"type": "server_vad"} if use_vad else None
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": AUDIO_RATE},
                    "transcription": {"model": model},
                    "turn_detection": turn_detection,
                }
            },
        },
    }


def open_session(api_key: str, model: str, use_vad: bool, verbose: bool = True) -> tuple[websocket.WebSocket, str]:
    """Open a transcription session, trying each documented URL form.

    A socket that opens is not a session that works: the server accepts the
    connection and only then rejects the config, so negotiation has to be part
    of the attempt rather than a later step.
    """
    attempts = [
        f"{REALTIME_URL}?intent=transcription",
        f"{REALTIME_URL}?model={model}",
        REALTIME_URL,
    ]
    problems = []
    for url in attempts:
        try:
            socket = websocket.create_connection(
                url, header=[f"Authorization: Bearer {api_key}"], timeout=20
            )
        except Exception as exc:  # noqa: BLE001 - report every failure mode to the operator
            problems.append(f"{url} -> {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  {problems[-1]}")
            continue
        socket.settimeout(None)
        problem = negotiate(socket, model, use_vad)
        if problem is None:
            if verbose:
                print(f"  {url} -> session accepted")
            return socket, url
        socket.close()
        problems.append(f"{url} -> {problem}")
        if verbose:
            print(f"  {url} -> {problem}")
    raise ConnectionError("; ".join(problems))


def negotiate(socket: websocket.WebSocket, model: str, use_vad: bool, timeout: float = 10.0) -> str | None:
    """Send the session config; return an error string, or None when accepted."""
    socket.send(json.dumps(session_payload(model, use_vad)))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        socket.settimeout(max(0.1, deadline - time.monotonic()))
        try:
            event = json.loads(socket.recv())
        except Exception:  # noqa: BLE001 - a timeout here just means no verdict yet
            break
        if event.get("type") == "session.updated":
            socket.settimeout(None)
            return None
        if event.get("type") == "error":
            socket.settimeout(None)
            error = event.get("error", {})
            return error.get("message") or json.dumps(error)
    socket.settimeout(None)
    return "no session.updated within the timeout"


def probe_models(api_key: str, use_vad: bool) -> None:
    print("Checking which transcription models this key accepts:\n")
    for model in CANDIDATE_MODELS:
        print(f"  {model}")
        try:
            socket, url = open_session(api_key, model, use_vad)
        except ConnectionError as exc:
            print(f"    -> unavailable\n")
            continue
        socket.close()
        print(f"    -> OK via {url}\n")
    print()


def run(api_key: str, model: str, use_vad: bool, seconds: float, commit_seconds: float) -> None:
    try:
        import sounddevice as sd
    except OSError as exc:
        raise SystemExit(
            "Microphone capture needs the PortAudio runtime "
            "(sudo apt-get install libportaudio2)."
        ) from exc

    print(f"Opening a transcription session ({model}, "
          f"{'server VAD' if use_vad else f'manual commits every {commit_seconds:g}s'})")
    try:
        socket, url = open_session(api_key, model, use_vad)
    except ConnectionError as exc:
        raise SystemExit(
            f"Could not open a transcription session:\n  {exc}\n"
            f"Run with --probe-models to see what this key accepts."
        ) from exc
    print(f"  using {url}\n")

    started = time.monotonic()
    counts: collections.Counter[str] = collections.Counter()
    deltas: list[tuple[float, str]] = []      # (elapsed, text)
    completed: list[tuple[float, str]] = []
    commits: list[float] = []                 # when each turn was closed
    transcript: list[str] = []
    send_lock = threading.Lock()
    stop = threading.Event()

    def send(event: dict[str, Any]) -> None:
        with send_lock:
            socket.send(json.dumps(event))

    def on_audio(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"{stamp(started)} mic status: {status}")
        send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(bytes(indata)).decode("ascii"),
        })

    def commit_loop() -> None:
        while not stop.wait(commit_seconds):
            send({"type": "input_audio_buffer.commit"})

    def read_events() -> None:
        while not stop.is_set():
            try:
                event = json.loads(socket.recv())
            except Exception:  # noqa: BLE001 - the socket closing ends the probe
                return
            kind = event.get("type", "?")
            counts[kind] += 1
            now = time.monotonic() - started
            if kind.endswith("input_audio_transcription.delta"):
                text = event.get("delta", "")
                deltas.append((now, text))
                transcript.append(text)
                print(f"{stamp(started)} delta      {text!r}")
            elif kind.endswith("input_audio_transcription.completed"):
                text = event.get("transcript", "")
                completed.append((now, text))
                print(f"{stamp(started)} COMPLETED  {text!r}")
            elif kind.endswith("input_audio_transcription.failed"):
                print(f"{stamp(started)} FAILED     {json.dumps(event)[:200]}")
            elif kind == "error":
                print(f"{stamp(started)} ERROR      {json.dumps(event.get('error', event))[:200]}")
            elif kind == "input_audio_buffer.committed":
                commits.append(now)
                print(f"{stamp(started)} · {kind}")
            elif kind not in {"input_audio_buffer.append"}:
                print(f"{stamp(started)} · {kind}")

    reader = threading.Thread(target=read_events, daemon=True, name="events")
    reader.start()

    print(f"Talk for {seconds:g}s. Ctrl-C to stop early.\n")
    with sd.RawInputStream(samplerate=AUDIO_RATE, blocksize=BLOCK_FRAMES,
                           channels=1, dtype="int16", callback=on_audio):
        committer = None
        if not use_vad:
            committer = threading.Thread(target=commit_loop, daemon=True, name="commits")
            committer.start()
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
    time.sleep(1.0)  # let any trailing events land
    socket.close()

    report(counts, deltas, completed, commits, transcript, use_vad)


def report(counts, deltas, completed, commits, transcript, use_vad) -> None:
    print("\n" + "=" * 68)
    print("EVENT TYPES SEEN")
    for kind, n in counts.most_common():
        print(f"  {n:>4}  {kind}")

    print(f"\nTRANSCRIPT ({sum(len(t.split()) for t in [''.join(transcript)])} words)")
    print(" ", "".join(transcript).strip() or "(nothing)")

    print("\nDELTA CADENCE")
    if not deltas:
        print("  No deltas at all. Text only arrives on .completed, so the editor")
        print("  would be entirely dependent on turn detection.")
    else:
        times = [t for t, _ in deltas]
        gaps = [round(b - a, 2) for a, b in zip(times, times[1:])]
        print(f"  {len(deltas)} deltas, first at {times[0]:.2f}s, last at {times[-1]:.2f}s")
        if gaps:
            print(f"  gap between deltas: median {sorted(gaps)[len(gaps)//2]:.2f}s, max {max(gaps):.2f}s")
        print(f"  {len(completed)} completed event(s)"
              + (f", {len(deltas)/len(completed):.1f} deltas each" if completed else ""))

    # Counting deltas per turn cannot tell streaming apart from a burst: a model
    # that transcribes the whole turn after it closes still emits one delta per
    # token, just all at once. The question is whether a delta arrives *before*
    # the commit that ends its turn -- that is the only thing that means text
    # showed up while the speaker was still talking.
    ahead = []
    for when, _ in deltas:
        following = [c for c in commits if c >= when]
        if following and following[0] - when <= AHEAD_WINDOW:
            ahead.append(following[0] - when)
    streaming_share = len(ahead) / len(deltas) if deltas else 0.0

    print(f"  {len(ahead)} of {len(deltas)} deltas arrived before their turn closed"
          f" ({streaming_share:.0%})")
    if ahead:
        print(f"  median lead over the commit: {sorted(ahead)[len(ahead)//2]:.2f}s")

    print("\nVERDICT")
    if not deltas:
        verdict = ("TURN-BOUNDARY ONLY -- no deltas at all. Text arrives only on "
                   ".completed, so turn detection is squarely on the critical path.")
    elif streaming_share < 0.5:
        verdict = ("BURST AFTER THE TURN -- deltas land only once the turn closes, so "
                   "a long unbroken sentence produces no text until the speaker stops. "
                   "Turn length is the deck's latency.")
    else:
        verdict = ("CONTINUOUS -- text arrives while speaking, so the deck editor can "
                   "tick on its own timer and turn detection is not load-bearing.")
    print(" ", verdict)
    print(f"  (mode: {'server VAD' if use_vad else 'manual commits, no VAD'})")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=CANDIDATE_MODELS[0], help=f"default: {CANDIDATE_MODELS[0]}")
    parser.add_argument("--vad", action="store_true", help="use server turn detection instead of manual commits")
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to record (default: 30)")
    parser.add_argument("--commit-seconds", type=float, default=2.0, help="manual commit cadence (default: 2.0)")
    parser.add_argument("--probe-models", action="store_true", help="report which models this key accepts, then exit")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this program.")

    if args.probe_models:
        probe_models(api_key, args.vad)
        return
    run(api_key, args.model, args.vad, args.seconds, args.commit_seconds)


if __name__ == "__main__":
    main()
