"""Measure the ESP32 audio stream before wiring it into Slidex.

The framing in ``parse_audio.py`` says how many samples arrive per frame but not
how fast they were captured, and the Realtime API has to be told the right rate
-- feed it 16 kHz audio while claiming 24 kHz and every word comes out stretched.

So measure it: count frames over a fixed window, multiply by the samples per
frame, and the arrival rate *is* the capture rate for a device streaming in real
time. Also reports signal level, so a silent or clipping microphone is obvious
before it becomes a transcription mystery.

    pipenv run python tools/esp32_probe.py --seconds 10
"""

from __future__ import annotations

import argparse
import array
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serial  # noqa: E402

import parse_audio  # noqa: E402  - reuse the framing rather than restate it

# Rates the Realtime API is documented to accept, for snapping the measurement.
PLAUSIBLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="serial port (default: auto-detect the ESP32)")
    parser.add_argument("--baud", type=int, default=parse_audio.BAUD_RATE)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    try:
        port = args.port or parse_audio.find_port()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Opening {port} at {args.baud} baud for {args.seconds:g}s...")
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Could not open {port}: {exc}", file=sys.stderr)
        return 1

    frames = 0
    truncated = 0
    peaks: list[int] = []
    buttons: list[tuple[float, str]] = []
    previous = 0
    silent_frames = 0

    # The first frame can be a long time coming if the link is idle, so the
    # clock starts when audio actually does.
    parse_audio.wait_for_header(ser)
    started = time.monotonic()
    deadline = started + args.seconds
    try:
        while time.monotonic() < deadline:
            frame = parse_audio.read_exact(ser, 1 + parse_audio.AUDIO_PAYLOAD_SIZE)
            if frame is None:
                truncated += 1
                parse_audio.wait_for_header(ser)
                continue

            button_byte = frame[0]
            samples = array.array("h", frame[1:])
            if sys.byteorder == "big":
                samples.byteswap()
            peak = max(abs(min(samples)), abs(max(samples))) if samples else 0
            peaks.append(peak)
            if peak < 200:
                silent_frames += 1
            frames += 1

            for bit, command in parse_audio.BUTTONS:
                if button_byte & ~previous & bit:
                    buttons.append((time.monotonic() - started, command.decode()))
                    print(f"  [{time.monotonic() - started:5.2f}s] button {command.decode()}")
            previous = button_byte

            parse_audio.wait_for_header(ser)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    elapsed = time.monotonic() - started
    if not frames:
        print("No frames arrived. Is the ESP32 powered on and streaming?")
        return 1

    samples_per_frame = parse_audio.AUDIO_PAYLOAD_SIZE // 2
    measured = frames * samples_per_frame / elapsed
    closest = min(PLAUSIBLE_RATES, key=lambda r: abs(r - measured))
    drift = abs(closest - measured) / closest

    print(f"\n{'=' * 62}")
    print(f"  frames            {frames} in {elapsed:.2f}s  ({frames / elapsed:.1f}/s)")
    print(f"  truncated         {truncated}")
    print(f"  samples/frame     {samples_per_frame}")
    print(f"  MEASURED RATE     {measured:.0f} Hz")
    print(f"  closest standard  {closest} Hz  ({drift:.1%} away)")
    print(f"  bytes/s on wire   {frames * (4 + 1 + parse_audio.AUDIO_PAYLOAD_SIZE) / elapsed:.0f}")
    print(f"  peak amplitude    median {statistics.median(peaks):.0f}, max {max(peaks)} (of 32767)")
    print(f"  silent frames     {silent_frames}/{frames} ({silent_frames / frames:.0%})")
    print(f"  buttons           {[b for _, b in buttons] or 'none pressed'}")
    print(f"{'=' * 62}")
    if drift > 0.05:
        print("\n  The rate does not land near a standard one. Either the link is not")
        print("  keeping up with the capture rate, or frames are being dropped --")
        print("  check `truncated` above before trusting this number.")
    elif max(peaks) < 500:
        print("\n  Frames are arriving but carry almost no signal. The microphone may")
        print("  be muted or unwired; transcription would return nothing.")
    else:
        print(f"\n  Looks like {closest} Hz, 16-bit mono. Pass that to Slidex as --esp32-rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
