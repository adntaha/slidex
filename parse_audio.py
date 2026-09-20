import argparse
import os
import socket
import sys

import serial
import serial.tools.list_ports

# Config
BAUD_RATE = 115200
IPC_HOST = "127.0.0.1"
IPC_PORT = 5005

HEADER_MAGIC = b'\xAA\xBB\xCC\xDD'
AUDIO_PAYLOAD_SIZE = 256 * 2  # 256 samples * 2 bytes per 16-bit sample

ESP32_BT_NAME = 'ESP32_Speech_Link'
ESP32_BT_ADDRESS = '409151FD5516'

BUTTONS = (
    (0x01, b"CMD_WIPE"),
    (0x02, b"CMD_PAUSE"),
    (0x04, b"CMD_NEXT"),
)


def running_under_wsl():
    try:
        with open("/proc/version") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def describe_ports(ports):
    if not ports:
        return "  (none)"
    return "\n".join(f"  {p.device:<12} {p.description}  [{p.hwid}]" for p in ports)


def find_port():
    """Locate the serial port the ESP32 is paired on.

    Preference order: explicit override, the port bound to the ESP32's
    Bluetooth address or name, then any outgoing Bluetooth SPP port.
    """
    override = os.environ.get("SLIDEX_SERIAL_PORT")
    if override:
        return override

    ports = list(serial.tools.list_ports.comports())

    for port in ports:
        haystack = " ".join(filter(None, (port.device, port.description, port.hwid)))
        haystack = haystack.upper().replace(":", "").replace("-", "")
        if ESP32_BT_ADDRESS in haystack or ESP32_BT_NAME.upper() in haystack:
            return port.device

    # No address match -- fall back to an outgoing Bluetooth SPP port. The
    # LOCALMFG&0000 entry is the incoming (listening) side with an all-zero
    # remote address, so it is never the ESP32.
    for port in ports:
        hwid = (port.hwid or "").upper()
        if "BTHENUM" in hwid and "LOCALMFG&0000" not in hwid:
            return port.device

    hint = ""
    if running_under_wsl():
        hint = (
            "\n\nYou are running under WSL2, which does not map Windows COM ports "
            "to /dev/ttyS*. Bluetooth SPP cannot be forwarded in (usbipd only "
            "handles USB), so run this script with Windows Python, or set "
            "SLIDEX_SERIAL_PORT to a port reachable from here."
        )
    raise RuntimeError(
        "Could not find the ESP32 serial port. Ports visible to this process:\n"
        + describe_ports(ports)
        + "\n\nPass --port explicitly or set SLIDEX_SERIAL_PORT to override." + hint
    )


def read_exact(ser, size):
    """Read exactly `size` bytes, or None if the stream stalls mid-frame.

    pyserial's read() returns a short chunk when the timeout expires, so a
    single read() is not enough to guarantee a whole frame.
    """
    buf = bytearray()
    while len(buf) < size:
        chunk = ser.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def wait_for_header(ser):
    """Slide a window over the stream until it matches HEADER_MAGIC."""
    window = bytearray()
    while True:
        byte = ser.read(1)
        if not byte:
            continue
        window += byte
        if len(window) > len(HEADER_MAGIC):
            del window[0]
        if window == HEADER_MAGIC:
            return


def main():
    parser = argparse.ArgumentParser(description="Stream ESP32 audio + button frames.")
    parser.add_argument("--port", help="Serial port to open (default: auto-detect).")
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--list", action="store_true", help="List serial ports and exit.")
    args = parser.parse_args()

    if args.list:
        print(describe_ports(list(serial.tools.list_ports.comports())))
        return 0

    try:
        port = args.port or find_port()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    ipc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Connecting to ESP32 stream on {port}...")
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Could not open {port}: {exc}", file=sys.stderr)
        ipc_socket.close()
        return 1
    print("Connected! Processing audio stream frames and asynchronous buttons...")

    prev_buttons = 0
    try:
        while True:
            # Step 1: Search for the start marker
            wait_for_header(ser)

            # Step 2: Read the 1-byte button snapshot plus the audio payload
            frame = read_exact(ser, 1 + AUDIO_PAYLOAD_SIZE)
            if frame is None:
                print("\n[warn] Truncated frame, resyncing...", file=sys.stderr)
                continue

            button_byte = frame[0]
            raw_audio_chunk = frame[1:]

            # Step 3: Trigger on rising edges only, so a held button fires once
            pressed = button_byte & ~prev_buttons
            for bit, command in BUTTONS:
                if pressed & bit:
                    print(f"\n[Event] {command.decode()} Triggered")
                    ipc_socket.sendto(command, (IPC_HOST, IPC_PORT))
            prev_buttons = button_byte
            print(frame)

            # Step 4: The clean raw binary audio array chunk
            # --- VOSK / WHISPER INTEGRATION GOES HERE ---
            # Pass `raw_audio_chunk` straight into your speech engine buffer.
            # It is perfectly safe and contains 0% text noise or line interference.
            del raw_audio_chunk

    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        ser.close()
        ipc_socket.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
