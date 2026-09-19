"""Deck storage shared by the hosted API routes.

Vercel functions are request-scoped and do not share memory, so the deck cannot
live in a module global the way it does in ``slidex.py``. It lives in a
Redis-compatible KV store reached over plain HTTPS instead, which keeps these
routes dependency-free.

Set ``KV_REST_API_URL`` and ``KV_REST_API_TOKEN`` (Vercel KV and the Upstash
integration both provide them; the ``UPSTASH_REDIS_REST_*`` names work too).
Without them the module falls back to a per-instance dict, which is enough for
``vercel dev`` but will not survive across invocations in production.

Files in ``api/`` whose names start with an underscore are importable helpers
rather than routes.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen

DECK_KEY = "slidex:deck"
EMPTY_DECK: dict[str, Any] = {"slides": []}
TIMEOUT = 5

_fallback: dict[str, str] = {}


class StoreUnavailable(RuntimeError):
    """The configured KV store could not be reached."""


def credentials() -> tuple[str, str] | None:
    url = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    return (url.rstrip("/"), token) if url and token else None


def _command(*args: str) -> Any:
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
        with urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - configured KV endpoint
            payload = json.load(response)
    except (OSError, ValueError) as exc:
        raise StoreUnavailable(str(exc)) from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise StoreUnavailable(str(payload["error"]))
    return payload.get("result") if isinstance(payload, dict) else None


def read_deck() -> dict[str, Any]:
    """Return the published deck, or an empty one if nothing has been pushed."""
    raw = _fallback.get(DECK_KEY) if credentials() is None else _command("GET", DECK_KEY)
    if not raw:
        return EMPTY_DECK
    try:
        deck = json.loads(raw)
    except ValueError as exc:
        raise StoreUnavailable(f"stored deck is not valid JSON: {exc}") from exc
    return deck if isinstance(deck, dict) and "slides" in deck else EMPTY_DECK


def write_deck(deck: dict[str, Any]) -> None:
    raw = json.dumps(deck)
    if credentials() is None:
        _fallback[DECK_KEY] = raw
        return
    _command("SET", DECK_KEY, raw)
