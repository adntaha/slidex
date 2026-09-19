"""GET /api/slides - the deck the browser polls. Public; viewers only read."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

from _store import StoreUnavailable, read_deck


class handler(BaseHTTPRequestHandler):  # noqa: N801 - name required by the Vercel Python runtime
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        try:
            body = json.dumps(read_deck()).encode()
        except StoreUnavailable as exc:
            # A 503 leaves the page showing RECONNECTING rather than an empty deck.
            body = json.dumps({"error": str(exc)}).encode()
            self._send(503, body)
            return
        self._send(200, body)

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return
