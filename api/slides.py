"""The deck API, as a single Vercel function.

``GET``  returns the published deck for the browser to render.
``POST`` publishes a deck; the machine holding the microphone calls this, and it
needs ``Authorization: Bearer $SLIDEX_PUSH_TOKEN``.

Both live in one file on purpose. Vercel bundles each function separately, and a
sibling ``from _store import ...`` depends on how the builder resolves paths
inside ``api/`` -- which fails the whole invocation when it guesses wrong. Only
the standard library is imported here, so there is nothing left to resolve.

Vercel functions share no memory, so the deck lives in a Redis-compatible KV
store reached over HTTPS. Set ``KV_REST_API_URL`` and ``KV_REST_API_TOKEN``
(Vercel KV and the Upstash integration both provide them; the
``UPSTASH_REDIS_REST_*`` names work too). Without them this falls back to a
per-instance dict: fine for ``vercel dev``, not durable in production.
"""

from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.request import Request, urlopen

DECK_KEY = "slidex:deck"
EMPTY_DECK: dict[str, Any] = {"slides": []}
KV_TIMEOUT = 5
MAX_BODY_BYTES = 1_000_000

_fallback: dict[str, str] = {}


class StoreUnavailable(RuntimeError):
    """The configured KV store could not be reached."""


def credentials() -> tuple[str, str] | None:
    url = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    return (url.rstrip("/"), token) if url and token else None


def kv_command(*args: str) -> Any:
    """Run one Redis command through the Upstash REST endpoint."""
    resolved = credentials()
    if resolved is None:
        raise StoreUnavailable("no KV store configured")
    url, token = resolved
    request = Request(
        url,
        data=json.dumps(list(args)).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=KV_TIMEOUT) as response:  # noqa: S310 - configured KV endpoint
            payload = json.load(response)
    except (OSError, ValueError) as exc:
        raise StoreUnavailable(f"{type(exc).__name__}: {exc}") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise StoreUnavailable(str(payload["error"]))
    return payload.get("result") if isinstance(payload, dict) else None


def read_deck() -> dict[str, Any]:
    raw = _fallback.get(DECK_KEY) if credentials() is None else kv_command("GET", DECK_KEY)
    if not raw:
        return EMPTY_DECK
    try:
        deck = json.loads(raw)
    except ValueError as exc:
        raise StoreUnavailable(f"stored deck is not valid JSON: {exc}") from exc
    return deck if isinstance(deck, dict) and "slides" in deck else EMPTY_DECK


def deck_version(deck: dict[str, Any]) -> int:
    version = deck.get("version", 0)
    return version if isinstance(version, int) and not isinstance(version, bool) else 0


def write_deck(deck: dict[str, Any]) -> bool:
    """Store a snapshot unless a newer publisher has already won."""
    if deck_version(deck) < deck_version(read_deck()):
        return False
    raw = json.dumps(deck)
    if credentials() is None:
        _fallback[DECK_KEY] = raw
        return True
    kv_command("SET", DECK_KEY, raw)
    return True


class handler(BaseHTTPRequestHandler):  # noqa: N801 - name required by the Vercel Python runtime
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        try:
            deck = read_deck()
        except StoreUnavailable as exc:
            # A 503 leaves the page showing RECONNECTING rather than an empty deck.
            self._send(503, {"error": str(exc)})
            return
        self._send(200, deck)

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
            accepted = write_deck(deck)
        except StoreUnavailable as exc:
            self._send(503, {"error": str(exc)})
            return
        self._send(200, {"status": "published" if accepted else "stale", "slides": len(deck["slides"])})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return
