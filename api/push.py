"""POST /api/push - the machine holding the microphone publishes the deck here.

Authenticated with a shared secret so the public URL cannot be written to. The
route fails closed: with no ``SLIDEX_PUSH_TOKEN`` set, nothing can publish.
"""

from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler

from _store import StoreUnavailable, write_deck

MAX_BODY_BYTES = 1_000_000


class handler(BaseHTTPRequestHandler):  # noqa: N801 - name required by the Vercel Python runtime
    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        # Read the body before anything else. Answering a request without
        # draining it resets the connection, so rejections must consume it too.
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            self._send(400, {"error": "bad Content-Length"})
            return
        raw = self.rfile.read(min(max(length, 0), MAX_BODY_BYTES + 1))

        secret = os.environ.get("SLIDEX_PUSH_TOKEN", "")
        offered = self.headers.get("Authorization", "").removeprefix("Bearer ")
        if not secret or not hmac.compare_digest(secret, offered):
            self._send(401, {"error": "unauthorized"})
            return
        if not raw:
            self._send(400, {"error": "empty body"})
            return
        if len(raw) > MAX_BODY_BYTES:
            self._send(413, {"error": "body too large"})
            return

        try:
            deck = json.loads(raw)
        except ValueError as exc:
            self._send(400, {"error": f"invalid JSON: {exc}"})
            return
        if not isinstance(deck, dict) or not isinstance(deck.get("slides"), list):
            self._send(400, {"error": "expected an object with a 'slides' list"})
            return

        try:
            write_deck(deck)
        except StoreUnavailable as exc:
            self._send(503, {"error": str(exc)})
            return
        self._send(200, {"status": "published", "slides": len(deck["slides"])})

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return
