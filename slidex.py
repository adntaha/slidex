"""Slidex: build a slide deck live from a speaker's microphone.

Audio streams to the OpenAI Realtime API, which edits the in-memory deck by
calling the local tools in ``TOOL_HANDLERS``. A tiny HTTP server publishes the
deck to ``deck.html``, which polls it and animates the changes.

Set OPENAI_API_KEY, then run:
    pipenv run python slidex.py --microphone --web
"""

from __future__ import annotations

import argparse
import array
import base64
import collections
import contextlib
import json
import os
import platform
import random
import socket
import sys
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


# Trace lines carry bullets and arrows. A legacy Windows console in cp1252 or
# cp437 raises UnicodeEncodeError on those rather than printing them, which would
# take the program down over a log line.
try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, OSError):  # pragma: no cover - not every stream supports it
    pass


def trace(message: str) -> None:
    """Emit progress immediately; useful when running a silent-text session."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


MAX_BULLETS_PER_SLIDE = 5
AUDIO_RATE = 24000  # Hz, 16-bit mono PCM
# A deck decision made mid-sentence is a decision made on half a thought, so a
# pause commits early. Semantic boundaries would be better than peak amplitude,
# but a speaker who never pauses must not be left with one slide at the end --
# hence the pause is only ever an *earlier* trigger than --chunk-seconds.
SILENCE_PEAK = 900  # int16 amplitude below which a block counts as silence
SILENCE_HOLD_SECONDS = 0.35  # a pause this long ends an utterance
MIN_UTTERANCE_SECONDS = 1.0  # never commit a sliver just because it opened quietly
MIN_WORDS_TO_EDIT = 4  # below this the transcript is not worth a deck decision
# Stating a subject takes a sentence. The first tick fires at four words, and
# four words into a talk is a greeting or a run-up, never the subject -- yet
# that was exactly what became the deck title.
MIN_WORDS_FOR_TITLE = 12
# The title may be revised while the subject is still emerging, and is frozen
# once this many content slides exist: by then the subject is settled, and a
# title that keeps changing reads as indecision to the audience.
TITLE_SETTLES_AFTER_SLIDES = 3
MIN_SPEECH_SECONDS = 0.3  # a window holding less speech than this is not a decision
PRE_ROLL_BLOCKS = 3  # ~300 ms of lead-in kept back so a word onset is not clipped

# The sounddevice wheels bundle PortAudio on Windows and macOS; only Linux needs
# the system package.
PORTAUDIO_HINT = (
    "Microphone streaming needs the PortAudio runtime. On Ubuntu/Debian run: "
    "sudo apt-get install libportaudio2"
    if platform.system() == "Linux"
    else "Microphone streaming could not load PortAudio. Reinstall dependencies: pipenv install"
)
# A decision normally takes a second or two. These bound what happens when one
# does not: audio stops piling up, and a wedged turn eventually releases.
MAX_BUFFERED_AUDIO_SECONDS = 8.0
MAX_TURN_SECONDS = 20.0
MAX_TOOL_ROUNDS = 4
# Slides alternate sides so a run of illustrated slides has some visual rhythm.
IMAGE_PLACEMENTS = ("right", "left")
IMAGE_SEARCH_TIMEOUT = 12
# How many search results a slide may choose from. Slide N takes result N so a
# run of similar queries (continuation slides, a title slide next to its first
# content slide) is illustrated with different pictures rather than the same top
# hit repeated; a search with fewer results than that falls back to a random
# pick from this many.
IMAGE_CANDIDATES = 10
IMAGE_QUERY_MODEL = "gpt-4.1-mini"  # a naming task, and it runs off the critical path
IMAGE_QUERY_TIMEOUT = 12

IMAGE_QUERY_SYSTEM = (
    "You turn one presentation slide into a search query for Wikimedia Commons, a "
    "library of photographs, maps and diagrams. "
    "Name the most concrete photographable thing the slide is about, in two to five "
    "words: an object, place, animal, person, machine or event. Search in English "
    "whatever language the slide is written in, because that is the language Commons "
    "is catalogued in. "
    # A slide's own words are the talk's vocabulary, not a picture's. Searching them
    # verbatim is what returned a wedding ceremony for a slide titled "Introduction".
    "Ignore the words that describe a slide's role in a talk rather than its subject "
    "-- overview, introduction, summary, agenda, next steps, challenges, approach, "
    "conclusion, results -- and look at what the bullets are actually about. Prefer a "
    "physical thing over an abstraction: for a slide about streaming audio from a "
    "microcontroller, 'ESP32 microcontroller board' will find a picture and "
    "'audio streaming architecture' will not. "
    "Plenty of slides have no subject worth a picture: an agenda, a greeting, a list "
    "of abstract goals. Set 'depictable' to false for those and leave the query empty. "
    "A confidently wrong photograph on a presentation screen is worse than no "
    "photograph at all."
)

IMAGE_QUERY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_query",
        "strict": True,
        "schema": {
            "type": "object",
            # 'subject' first: naming what the slide is about before judging whether
            # it can be pictured stops an abstract title being waved through.
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "What the slide is about, ignoring presentation scaffolding words.",
                },
                "depictable": {
                    "type": "boolean",
                    "description": "Whether a photograph, map or diagram of that subject would mean anything.",
                },
                "query": {
                    "type": "string",
                    "description": "Two to five English words to search Commons for. Empty if not depictable.",
                },
            },
            "required": ["subject", "depictable", "query"],
            "additionalProperties": False,
        },
    },
}

SLIDES: list[dict[str, Any]] = []
DECK_LOCK = threading.RLock()
# The display receives this with every snapshot so its status matches the audio gate.
DECK_PAUSED = threading.Event()
DECK_VERSION = 0
# Image lookups are HTTP calls. They must never run on the Realtime event loop:
# a slow one stalls every response, and the microphone backs up behind it.
IMAGE_WORKERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="image-search")
_LAST_SLIDE_ID = 0


def mark_deck_changed() -> None:
    """Advance a version that lets viewers discard an older published deck."""
    global DECK_VERSION
    DECK_VERSION = max(DECK_VERSION + 1, time.time_ns())


def new_slide(kind: str, title: str) -> dict[str, Any]:
    """Build a slide with an id that survives the renumbering an insert causes."""
    global _LAST_SLIDE_ID
    with DECK_LOCK:
        _LAST_SLIDE_ID += 1
        mark_deck_changed()
        return {"id": _LAST_SLIDE_ID, "kind": kind, "title": title, "bullets": []}


def current_content_slide() -> dict[str, Any] | None:
    """Return the most recent slide that can receive speaker notes."""
    for slide in reversed(SLIDES):
        if slide.get("kind") != "title":
            return slide
    return None


def active_slide_id() -> int | None:
    """The slide the recorder is writing to, which is the one the display follows.

    Derived rather than tracked. Every edit lands on current_content_slide(), so
    that is the live slide whenever one exists; only an otherwise empty deck is
    showing its title slide. A tracked pointer fell out of step here: the title
    slide is created after the first content slide often enough, and the pointer
    followed it while the bullets kept going to the content slide -- so the
    viewer was parked on the title, and stepping forward to the slide actually
    being written read as browsing away and paused the recorder.
    """
    with DECK_LOCK:
        slide = current_content_slide()
        if slide is None and SLIDES:
            slide = SLIDES[0]
        return None if slide is None else slide["id"]


def slide_state() -> dict[str, Any]:
    """Return enough state for the model to choose its next deck-editing action.

    ``deck`` lists every slide so the model can tell that the speaker has come
    back to an earlier subject instead of forcing the point into whatever slide
    happens to be last. Only titles and counts appear there: every tool result
    carries this payload and the Realtime conversation keeps all of them, so
    repeating every bullet of every slide would grow the context quadratically.
    Full bullets accompany the one slide that can still be edited.
    """
    with DECK_LOCK:
        slide = current_content_slide()
        return {
            "slide_count": len(SLIDES),
            "has_title_slide": any(item["kind"] == "title" for item in SLIDES),
            "deck": [
                {
                    "number": index,
                    "kind": item["kind"],
                    "title": item["title"],
                    "bullet_count": len(item["bullets"]),
                }
                for index, item in enumerate(SLIDES, start=1)
            ],
            "current_slide": None
            if slide is None
            else {
                "number": SLIDES.index(slide) + 1,
                "title": slide["title"],
                "bullets": list(slide["bullets"]),
            },
        }


def deck_snapshot() -> dict[str, Any]:
    """Create a stable copy for the local browser without exposing tool internals."""
    with DECK_LOCK:
        return {
            "version": DECK_VERSION,
            "paused": DECK_PAUSED.is_set(),
            "active_slide_id": active_slide_id(),
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


ELLIPSIS = "\u2026"


def tidy_title(text: str) -> str:
    """Normalise a title from the model: one ellipsis character marks a provisional one.

    The model is told to end a title it had to cut short with an ellipsis. It
    writes that as either the character or three periods; the deck keeps one
    form so a provisional title can be recognised, and so the slide never shows
    'Tax farming...' next to 'Tax farming\u2026'.
    """
    cleaned = text.strip()
    stripped = cleaned.rstrip(". " + ELLIPSIS)
    provisional = len(cleaned) - len(stripped) >= 2 or cleaned.endswith(ELLIPSIS)
    return f"{stripped}{ELLIPSIS}" if provisional and stripped else stripped or cleaned.strip(". ")


def is_provisional(title: str) -> bool:
    return title.endswith(ELLIPSIS)


def title_slide() -> dict[str, Any] | None:
    return next((slide for slide in SLIDES if slide.get("kind") == "title"), None)


def create_title_slide(title: str) -> dict[str, Any]:
    """Create the deck's title slide, or revise it while the talk is still young.

    The opening words of a talk are almost never its subject -- a greeting, a
    joke, the run-up -- yet that is when a title is first asked for. So the title
    is a standing judgement rather than a one-shot: the first plausible one goes
    up, and a better one replaces it until enough content slides exist that the
    subject is settled. A one-shot title froze whatever the first fragment was.
    """
    with DECK_LOCK:
        cleaned = tidy_title(title)
        if not cleaned:
            return {"status": "ignored", "reason": "A title is required.", **slide_state()}
        existing = title_slide()
        if existing is None:
            slide = new_slide("title", cleaned)
            SLIDES.insert(0, slide)
            request_slide_image(slide)
            trace(f"TITLE SLIDE: {cleaned}")
            return {"status": "created", **slide_state()}
        if existing["title"] == cleaned:
            return {"status": "unchanged", **slide_state()}
        if sum(1 for slide in SLIDES if slide["kind"] != "title") >= TITLE_SETTLES_AFTER_SLIDES:
            return {
                "status": "ignored",
                "reason": f"The title is final once the deck has {TITLE_SETTLES_AFTER_SLIDES} content slides.",
                **slide_state(),
            }
        trace(f"TITLE SLIDE RETITLED: {existing['title']!r} -> {cleaned!r}")
        existing["title"] = cleaned
        # The illustration was chosen for the old title; choose it again.
        existing.pop("image", None)
        existing.pop("image_requested", None)
        request_slide_image(existing)
        return {"status": "updated", **slide_state()}


def append_content_slide(title: str) -> dict[str, Any]:
    with DECK_LOCK:
        slide = new_slide("content", title)  # the last content slide is the one on display
        SLIDES.append(slide)
        trace(f"NEW SLIDE #{len(SLIDES)}: {title}")
        return slide


def create_new_slide(title: str) -> dict[str, Any]:
    """Start a new topic in the live deck."""
    with DECK_LOCK:
        cleaned = tidy_title(title)
        if not cleaned:
            return {"status": "ignored", "reason": "A slide title is required.", **slide_state()}
        append_content_slide(cleaned)
        return {"status": "created", **slide_state()}


def update_slide_title(title: str) -> dict[str, Any]:
    """Complete or correct the current slide's title.

    A slide sometimes has to open before the speaker has finished naming its
    subject; that title ends with an ellipsis. Without this tool the Realtime
    session had no way to finish it, so the ellipsis stayed for the whole talk.
    """
    with DECK_LOCK:
        slide = current_content_slide()
        cleaned = tidy_title(title)
        if slide is None:
            return {"status": "ignored", "reason": "There is no current slide.", **slide_state()}
        if not cleaned:
            return {"status": "ignored", "reason": "A slide title is required.", **slide_state()}
        if slide["title"] == cleaned:
            return {"status": "unchanged", **slide_state()}
        trace(f"SLIDE #{SLIDES.index(slide) + 1} RETITLED: {slide['title']!r} -> {cleaned!r}")
        if is_provisional(slide["title"]):
            # The picture was chosen for a fragment; choose it again for the subject.
            slide.pop("image", None)
            slide.pop("image_requested", None)
        slide["title"] = cleaned
        request_slide_image(slide)
        return {"status": "updated", **slide_state()}


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
        # A full slide must not swallow what the speaker just said. Refusing the
        # bullet loses the point outright, so the overflow opens the next slide.
        rolled = len(slide["bullets"]) >= MAX_BULLETS_PER_SLIDE
        if rolled:
            slide = append_content_slide(continued_title(slide["title"]))
        slide["bullets"].append(cleaned)
        if rolled:
            trace(f"SLIDE #{len(SLIDES)} '{slide['title']}': rolled over, the slide before was full")
        trace(f"BULLET #{len(slide['bullets'])}: • {cleaned}")
        # Image lookup happens in the background and retries when earlier search
        # terms found nothing, so every completed slide gets an illustration.
        request_slide_image(slide)
        return {
            "status": "saved",
            "bullet_number": len(slide["bullets"]),
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


def pick_image(candidates: list[dict[str, str]], rank: int, used: set[str]) -> dict[str, str] | None:
    """Choose a slide's picture from its search results.

    Result ``rank`` (the slide's position in the deck) is the first choice, so
    the first slide gets the first relevant picture, the second slide the
    second, and so on. When the search did not return that many, or that result
    is already on another slide, the pick is random among what is left -- never
    the top hit by default, which is what put the same picture on every slide of
    one subject.
    """
    fresh = [item for item in candidates if item["url"] not in used] or candidates
    if not fresh:
        return None
    if rank < len(candidates) and candidates[rank]["url"] not in used:
        return candidates[rank]
    return random.choice(fresh[:IMAGE_CANDIDATES])


def find_image(query: str, rank: int = 0, used: set[str] | None = None) -> dict[str, str] | None:
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
            "gsrlimit": str(IMAGE_CANDIDATES),
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": "1600",
        }
    )
    request = Request(
        f"https://commons.wikimedia.org/w/api.php?{params}",
        # Wikimedia's User-Agent policy wants a contact it can reach. A generic
        # agent gets throttled to 429s, which read here exactly like "no such
        # picture" and quietly leave a whole talk unillustrated.
        headers={"User-Agent": "Slidex/1.0 (https://github.com/adntaha/slidex)"},
    )
    try:
        with urlopen(request, timeout=IMAGE_SEARCH_TIMEOUT) as response:  # noqa: S310 - fixed Wikimedia endpoint
            pages = json.load(response).get("query", {}).get("pages", {}).values()
    except (OSError, ValueError) as exc:
        trace(f"IMAGE SEARCH FAILED: {exc}")
        return None

    candidates = []
    # Search results carry an ``index`` giving their relevance order; the pages
    # object itself is keyed by page id and arrives in no useful order.
    for page in sorted(pages, key=lambda item: item.get("index", 0)):
        info = next(iter(page.get("imageinfo", [])), {})
        image_url = info.get("thumburl") or info.get("url")
        # File search also returns audio, video, and PDFs; those have no usable thumbnail.
        if not image_url or not info.get("mime", "").startswith("image/"):
            continue
        title = page.get("title", "")
        candidates.append({
            "url": image_url,
            "alt": title.removeprefix("File:"),
            "source": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
        })
    chosen = pick_image(candidates, rank, used or set())
    if chosen is None:
        trace(f"IMAGE NOT FOUND: {cleaned}")
        return None
    position = candidates.index(chosen) + 1
    how = f"result {position} for slide {rank + 1}" if position == rank + 1 else f"result {position}, random"
    trace(f"IMAGE FOUND: {cleaned} ({how} of {len(candidates)})")
    return chosen


def image_query(title: str, bullets: list[str]) -> str | None:
    """Turn a slide into words Wikimedia Commons can actually match.

    Searching the slide's own text searches the talk's vocabulary rather than for
    a picture, which is how a slide titled "Introduction" ended up illustrated
    with a wedding ceremony. Returns None when the slide is not worth a picture,
    and falls back to the slide's own words if the call fails -- a slow network
    should cost a better query, not the image.
    """
    fallback = " ".join(part for part in (title, *bullets[:2]) if part.strip())
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return fallback
    body = json.dumps(
        {
            "model": IMAGE_QUERY_MODEL,
            "messages": [
                {"role": "system", "content": IMAGE_QUERY_SYSTEM},
                {"role": "user", "content": json.dumps({"title": title, "bullets": bullets})},
            ],
            "response_format": IMAGE_QUERY_SCHEMA,
            "max_completion_tokens": 200,
        }
    ).encode()
    request = Request(
        EDITOR_URL,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=IMAGE_QUERY_TIMEOUT) as response:  # noqa: S310 - fixed OpenAI endpoint
            answer = json.loads(json.load(response)["choices"][0]["message"]["content"])
    except (OSError, ValueError, KeyError, IndexError) as exc:
        trace(f"IMAGE QUERY FAILED: {exc}; searching the slide's own words instead.")
        return fallback
    if not answer.get("depictable"):
        trace(f"IMAGE SKIPPED for '{title}': {answer.get('subject', '')!r} is not worth a picture.")
        return None
    query = (answer.get("query") or "").strip()
    trace(f"IMAGE QUERY for '{title}': {query!r}")
    return query or fallback


def load_slide_image(slide: dict[str, Any], title: str, bullets: list[str]) -> None:
    """Background worker: illustrate one slide and leave failures retryable."""
    query = image_query(title, bullets)
    if query is None:
        # Deliberately unillustrated. Leave image_requested set so the slide is
        # not asked about again every time a bullet lands on it.
        return
    with DECK_LOCK:
        if slide not in SLIDES:
            return
        # Read under the lock, used outside it: the search is a network call.
        rank = SLIDES.index(slide)
        used = {item["image"]["url"] for item in SLIDES if item.get("image")}
    image = find_image(query, rank, used)
    if image is None:
        with DECK_LOCK:
            if slide in SLIDES:
                slide.pop("image_requested", None)
                trace(f"IMAGE DEFERRED for slide {slide['id']}; will retry with more context.")
        return
    with DECK_LOCK:
        if slide not in SLIDES or "image" in slide:
            return
        if slide["kind"] == "title":
            placement = "background"
        else:
            rank = [item for item in SLIDES if item["kind"] == "content"].index(slide)
            placement = IMAGE_PLACEMENTS[rank % len(IMAGE_PLACEMENTS)]
        slide["image"] = {**image, "placement": placement}
        trace(f"IMAGE PLACED on slide {slide['id']} / {placement}")


TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "create_title_slide": create_title_slide,
    "create_new_slide": create_new_slide,
    "new_bullet_point": new_bullet_point,
    "update_bullet_point": update_bullet_point,
    "update_slide_title": update_slide_title,
}

# The local display can send the same command as a physical controller. This is
# intentionally process-local: a hosted viewer must not control the recorder.
ACTIVE_DECK_COMMAND: Callable[[str], None] | None = None

TOOLS = [
    {
        "type": "function",
        "name": "create_title_slide",
        "description": (
            "Set the deck's opening title slide: the subject of the whole talk in a few words, "
            "as a poster would carry it. Never the speaker's opening words, a greeting, or the "
            "run-up to the subject; wait until the speaker has said what the talk is about. Call "
            "it again with a better title if the talk turns out to be about something broader or "
            f"different. The title is final once the deck has {TITLE_SETTLES_AFTER_SLIDES} content slides."
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
                    "description": (
                        "A short, specific title for the new slide: a complete phrase naming its subject. "
                        "Only when the slide must open before the speaker has finished naming the subject, "
                        "end it with an ellipsis (\u2026) and finish it with update_slide_title."
                    ),
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
    {
        "type": "function",
        "name": "update_slide_title",
        "description": (
            "Replace the current slide's title: to complete one that ends with an ellipsis (\u2026) "
            "once the speaker has finished naming the subject, or when they name it more precisely."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The complete replacement title, without an ellipsis.",
                }
            },
            "required": ["title"],
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
        elif self.path in ("/deck", "/deck.html"):
            body = (Path(__file__).parent / "public" / "deck.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
        elif self.path == "/api/slides":
            body = json.dumps(deck_snapshot()).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found\n")

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/api/command":
            self._send(404, "text/plain; charset=utf-8", b"Not found\n")
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            self._send(400, "text/plain; charset=utf-8", b"Bad command length\n")
            return
        command = self.rfile.read(min(max(length, 0), 64)).decode("ascii", "replace").strip()
        if command not in DECK_COMMANDS:
            self._send(400, "text/plain; charset=utf-8", b"Unknown command\n")
            return
        if ACTIVE_DECK_COMMAND is None:
            self._send(409, "text/plain; charset=utf-8", b"Recorder is not running\n")
            return
        ACTIVE_DECK_COMMAND(command)
        self._send(200, "application/json; charset=utf-8", json.dumps(deck_snapshot()).encode())

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


EXPORTS_DIR = Path(__file__).parent / "exports"


def wipe_deck() -> None:
    """Clear the deck, first writing what was on it to a PDF in exports/.

    The reset is the end of a talk, and a talk cannot be given again -- so the
    deck is saved before it goes. The export downloads every slide picture and
    so runs on its own thread; the deck itself is cleared at once. The thread is
    not a daemon: a Ctrl-C right after the reset must not lose the file.
    """
    with DECK_LOCK:
        snapshot = deck_snapshot()
        SLIDES.clear()
        mark_deck_changed()
    if not snapshot["slides"]:
        return

    def export() -> None:
        try:
            from deck_pdf import export_deck  # reportlab is only needed here
        except ImportError:
            trace("EXPORT SKIPPED: run `pipenv install` to get reportlab, which writes the PDF.")
            return
        try:
            path = export_deck(snapshot, EXPORTS_DIR)
        except Exception as exc:  # noqa: BLE001 - a failed export must never take the recorder down
            trace(f"EXPORT FAILED: {exc}")
            return
        trace(f"EXPORTED {len(snapshot['slides'])} slide(s) to {path}")

    threading.Thread(target=export, name="deck-export").start()


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
                        "You are a conservative, evidence-only real-time slide-deck editor. Build a coherent deck"
                        " only from the main speaker's clearly heard, explicit ideas using tools, not visible "
                        "text. Never invent, assume, or add a theme, fact, entity, relationship, or motivation "
                        "the speaker did not state. Background conversations, television, ambient speech, noise, "
                        "partial phrases, and uncertain audio are not deck content. If you are not highly "
                        "confident the statement is from the main speaker and can be recorded without adding "
                        "meaning, make no tool call. Do not reinterpret a statement through the lens of the "
                        "current slide's topic. Preserve the speaker's language in every title and bullet: never "
                        "translate it into English or any other language. Once the speaker has actually said what "
                        "the talk is about, call create_title_slide with the subject of the talk in a few words, "
                        "as a poster would carry it -- never their opening words, a greeting, or the run-up such "
                        "as 'today I want to talk about'. A fragment is not a subject: wait. If the talk later "
                        "proves to be about something broader or different, call create_title_slide again with "
                        "the better title. When there is a substantive point, create a separate content slide "
                        "with create_new_slide before adding bullets. Add a concise,"
                        " standalone bullet only for substantive, sufficiently complete ideas. Do not add bullets"
                        " for filler, false starts, or repetition. When later speech corrects or meaningfully "
                        "refines a current-slide bullet, use update_bullet_point instead of adding a duplicate. "
                        "Preserve established, detailed bullets as written; do not expand, condense, or rephrase "
                        "them unless the speaker explicitly corrects or substantially refines that exact point. "
                        "When one coherent topic is finished and the speaker begins a genuinely new topic, call "
                        "create_new_slide with a short title before adding that topic's bullets. Earlier slides "
                        "are immutable: never retitle, edit, or add bullets to a slide once a newer content slide "
                        "exists. Never stop recording a point because the slide looks full: keep calling "
                        "new_bullet_point and the deck rolls the overflow onto a continuation slide by itself. "
                        "Never call create_new_slide just to make room, though -- a new slide is for a new "
                        "subject, and its title comes from what the speaker starts saying after the transition. "
                        "A slide title is a complete phrase that names the subject. Prefer to wait for the "
                        "speaker to finish naming a new subject before opening its slide; only when the "
                        "subject has plainly changed but its name is still cut off, open the slide with the "
                        "words so far ending in an ellipsis (\u2026), then call update_slide_title with the "
                        "complete title as soon as the speaker finishes it. A complete title never carries "
                        "an ellipsis. "
                        "Treat a clear, "
                        "explicitly stated change of subject, entity, timeframe, or question as a new topic; do "
                        "not force it into the current slide or rewrite a current bullet merely because it was "
                        "the latest one. Ignore isolated offhand remarks that neither support the current topic "
                        "nor establish a new one. If those remarks become a sustained, coherent discussion, then "
                        "create a new slide with a title for that new topic. "
                        "Every tool result lists the whole deck: check it before assuming the "
                        "speaker is still on the current slide's subject. Do not make a new slide merely because of a pause. Use tools rather "
                        "than prose, but making no call at all is the correct and expected answer whenever "
                        "nothing new was clearly said."
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


def stream_microphone(client: RealtimeToolClient, args: argparse.Namespace) -> None:
    """Send PCM continuously and manually ask for a deck decision every few seconds."""
    chunk_seconds = args.chunk_seconds
    stop_commits = threading.Event()
    boundary = threading.Event()
    buffered_bytes = 0
    speech_bytes = 0
    silent_seconds = 0.0
    dropping = False
    buffer_lock = threading.Lock()
    pre_roll: collections.deque[bytes] = collections.deque(maxlen=PRE_ROLL_BLOCKS)
    max_buffered_bytes = int(MAX_BUFFERED_AUDIO_SECONDS * AUDIO_RATE * 2)
    min_utterance_bytes = int(MIN_UTTERANCE_SECONDS * AUDIO_RATE * 2)

    def on_audio(raw_audio: bytes) -> None:
        """Buffer speech only.

        Silence must never reach the model. Committing a window of it asks what
        should change about the deck when nothing was said, and the model
        answers by inventing a plausible next bullet.
        """
        nonlocal buffered_bytes, speech_bytes, dropping, silent_seconds
        samples = array.array("h", raw_audio)
        speaking = bool(samples) and max(abs(min(samples)), abs(max(samples))) >= SILENCE_PEAK

        with buffer_lock:
            silent_seconds = 0.0 if speaking else silent_seconds + len(samples) / AUDIO_RATE
            if not speaking and speech_bytes >= min_utterance_bytes and silent_seconds >= SILENCE_HOLD_SECONDS:
                boundary.set()

            if speaking:
                # Once a decision is slow, buffering everything said meanwhile
                # only produces one huge catch-up turn. Cap it, drop the excess.
                if buffered_bytes + len(raw_audio) > max_buffered_bytes:
                    if not dropping:
                        dropping = True
                        trace(f"Buffer full at {MAX_BUFFERED_AUDIO_SECONDS:g}s; dropping audio until the deck catches up.")
                    return
                blocks = [*pre_roll, raw_audio] if speech_bytes == 0 else [raw_audio]
                pre_roll.clear()
                speech_bytes += len(raw_audio)
            elif speech_bytes and silent_seconds <= SILENCE_HOLD_SECONDS:
                blocks = [raw_audio]  # a breath inside a phrase is part of the phrase
            else:
                pre_roll.append(raw_audio)  # keep a little lead-in, send nothing
                return
            buffered_bytes += sum(len(block) for block in blocks)

        for block in blocks:
            client.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(block).decode("ascii"),
                }
            )

    def commit_audio_chunks() -> None:
        nonlocal buffered_bytes, speech_bytes, dropping, silent_seconds
        while not stop_commits.is_set():
            # Whichever comes first: the speaker drew breath, or --chunk-seconds.
            paused = boundary.wait(chunk_seconds)
            boundary.clear()
            if stop_commits.is_set():
                break
            client.release_stalled_turn()
            with buffer_lock:
                pending, speech = buffered_bytes, speech_bytes
            # No speech means no decision to make. Asking anyway is exactly what
            # makes the model invent bullets while the speaker is simply quiet.
            if speech < MIN_SPEECH_SECONDS * AUDIO_RATE * 2 or not client.response_idle.is_set():
                continue
            with buffer_lock:
                buffered_bytes = speech_bytes = 0
                silent_seconds = 0.0
                dropping = False
            trace(
                f"Committing {pending / (AUDIO_RATE * 2):.1f}s of audio "
                f"({'pause' if paused else 'max wait'})."
            )
            client.begin_manual_audio_turn()

    client.configure(manual_audio_turns=True)
    # This mode has no editor state for a button to act on, so the ESP32's
    # buttons only gate the audio here; the deck commands live in --transcribe.
    DECK_PAUSED.clear()

    def on_button(command: str) -> None:
        if command == "CMD_PAUSE":
            DECK_PAUSED.clear() if DECK_PAUSED.is_set() else DECK_PAUSED.set()
            trace(f"COMMAND CMD_PAUSE: {'paused' if DECK_PAUSED.is_set() else 'resumed'}.")
        else:
            trace(f"COMMAND {command} is only wired up in --transcribe mode.")

    global ACTIVE_DECK_COMMAND
    ACTIVE_DECK_COMMAND = on_button
    with audio_source(args, on_audio, on_button, DECK_PAUSED):
        trace(
            f"Audio is live. A deck decision runs at each pause, and at least every "
            f"{chunk_seconds:g} seconds; "
            "assistant text remains hidden. Press Ctrl-C to stop."
        )
        threading.Thread(target=commit_audio_chunks, daemon=True, name="audio-commit-loop").start()
        # The event loop runs off the main thread so Ctrl-C lands during a sleep
        # rather than inside a blocking socket read, which Windows will not
        # interrupt. Failures are carried back rather than lost with the thread.
        failure: list[BaseException] = []

        def listen() -> None:
            try:
                client.listen_forever(manual_audio_turns=True)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                failure.append(exc)

        listener = threading.Thread(target=listen, daemon=True, name="realtime-events")
        listener.start()
        try:
            while listener.is_alive():
                time.sleep(0.25)
        finally:
            stop_commits.set()
        if failure:
            raise failure[0]


# --------------------------------------------------------------------------
# Transcribe-then-derive mode
#
# The speech-to-speech model has to decide what to do with the deck while it is
# still listening, so a slow decision stops the ear and a quiet moment still
# demands an answer. Splitting the two removes both: a transcription session
# only ever produces text, and a separate editor reads that text on its own
# clock. Nothing the editor does can starve the microphone.
# --------------------------------------------------------------------------

TRANSCRIBE_URL = f"{REALTIME_URL}?intent=transcription"
TRANSCRIBE_MODEL = "gpt-live-transcribe"
# Measured against the API: the newest transcription models stream text as you
# speak but reject turn_detection outright, while the ones that accept it only
# emit text once a turn closes. You get live text or clean sentence boundaries,
# not both.
VAD_CAPABLE_MODELS = ("gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1")
TRANSCRIBE_COMMIT_SECONDS = 8.0
# Measured on the transition cases: gpt-4.1-mini folded a new subject into the
# open slide on 2 of 6, while gpt-4.1 took all 6 -- and did it in less time
# (median 0.80s against 0.96s), so there is nothing to trade away here.
EDITOR_MODEL = "gpt-4.1"
EDITOR_URL = "https://api.openai.com/v1/chat/completions"
EDITOR_TIMEOUT = 25
EDITOR_WORKERS = 3  # editor calls overlap; see the staleness guards in run_transcription_mode
# A structured response containing non-Latin text can be longer than its visual
# slide content suggests. Leave enough room that JSON is not cut inside a
# Unicode escape sequence.
EDITOR_MAX_TOKENS = 800
DETAILED_SLIDES_IN_PAYLOAD = 2  # older slides contribute titles only, so the prompt stays flat

# The judgement this prompt exists to get right is that 'so', 'okay', 'now' and
# 'anyway' open a new topic about as often as they open nothing at all, so the
# decision is keyed on whether a different subject follows the phrase rather than
# on the phrase itself. Rules are stated once: a restated rule reads to the model
# as a second, subtly different rule.
EDITOR_SYSTEM = (
    "You maintain one slide of a live deck being built from a talk as it happens. "
    "You are given the titles of earlier slides, the last few finished slides in full, "
    "the slide currently being written, and the transcript of everything said since "
    "that slide started. "
    # Without this the model narrates the talk instead of captioning it: real runs
    # produced "Speaker intends to avoid a jokey tone" from "we're gonna be fun".
    "Write slides, not minutes. A bullet is a short phrase an audience takes in at a "
    "glance, not a sentence copied out of the transcript and not a report of what the "
    "speaker did. Compress each point to its content: keep names, numbers and "
    "technical terms exactly as spoken, and drop the scaffolding around them -- "
    "'I think', 'we should', 'what I want to say is'. Never describe the speaker in "
    "the third person; state the point itself. "
    "Ground every word in the transcript: never add a fact, entity or theme that is "
    "not there. "
    # A speaker does not change language mid-talk, so a span that looks like one is
    # almost always the transcriber slipping. Anchoring on the deck rather than on
    # the span stops one bad span from turning the deck bilingual.
    "The deck is written in one language throughout: the language of the talk, which "
    "the existing slides already establish. Never translate the deck into another "
    "language, and never mix two. A speaker does not switch language mid-talk, so "
    "when a span looks like a different language it is a mis-transcription or a "
    "quoted foreign term -- keep writing in the language the deck is already in. "
    "Proper nouns and technical terms stay exactly as spoken. "
    "Speech carries a great deal that means nothing on a slide: hesitations, restarts, "
    "self-repetition, hedges, and asides to the room. Strip it. When a span is only "
    "that, answer 'none' and leave the slide untouched. "
    "Keep wording stable between calls. You are shown your own previous output; leave "
    "an existing bullet exactly as it stands unless the speaker corrected it or "
    "genuinely advanced that point, because rewording for style alone makes the deck "
    "flicker in front of the audience. "
    # Measured: with the rollover advertised and nothing said about its limits, a
    # full slide plus a transition came back as 'update' with six bullets, and the
    # next topic landed on a slide titled "(continued)" after the previous one.
    f"A slide shows {MAX_BULLETS_PER_SLIDE} bullets. When one subject genuinely runs "
    "longer than that, list it all anyway -- the deck rolls the remainder onto a "
    "continuation slide, so never drop a point to stay under the limit. That overflow "
    "is for one subject that will not fit, never for a second subject: if the speaker "
    "has moved on, answer 'new' instead of adding their new subject to a full slide. "
    "A full slide is not a reason to choose 'update', and a continuation slide must "
    "never be where a new topic starts. "
    "Reply with one JSON object choosing an action: "
    "'none' when the span adds nothing worth showing; "
    "'update' to rewrite the current slide; "
    "'new' when the subject has changed, which freezes the finished slide and opens "
    "the next one from your title and bullets. "
    # Measured: given one span four parts old subject to one part new, the model
    # folded the new subject in as a sixth bullet every time, however the rule was
    # worded. Restating the current slide is simply the cheapest thing it can emit.
    # So the boundary is settled in its own fields, which the schema puts before
    # the content: having named the subject, it cannot quietly file it under the
    # old title.
    "You get two views of the speech. 'recent_speech' is what has been said since your "
    "last decision -- that is where the speaker is now. "
    "'transcript_since_slide_started' is everything since the current slide began; use "
    "it only to keep the bullets you carry forward accurate. "
    "Answer the fields in order. In 'recent_subject', name what 'recent_speech' is "
    "about in a few words, leaving it empty when that is only filler. Then set "
    "'continues_current_slide': true only when 'recent_subject' is the same subject "
    "the current slide already covers. When it is false and 'recent_subject' is not "
    "empty, 'action' must be 'new' -- a slide holds one subject and one only, so "
    "never append a second subject to it as a further bullet, however much of the "
    "longer transcript came before it. "
    "Decide this on the subject, never on the phrasing. A phrase like 'so', 'okay', "
    "'now', 'right' or 'anyway' is a hand-off only when a subject the current slide "
    "does not cover follows it -- 'moving on to X', 'next', \"let's talk about X\", "
    "'that brings me to X', 'switching gears'. Then answer 'new' even if the current "
    "slide is nearly empty. When more of the same subject follows instead, the phrase "
    "is filler: ignore it and keep updating. On 'new', title and fill the slide from "
    "what was said after the hand-off only, and leave bullets empty if the speaker has "
    "so far only named the topic; never carry the old subject's points across. "
    "A title is a complete phrase that names the slide's subject. Prefer a complete "
    "title: when the speaker has only begun to name the new subject and the rest is "
    "surely one pass away, that fragment is not yet a subject, so wait. Only when the "
    "subject has plainly changed but its name is still cut off does the slide open with "
    "the words so far ending in an ellipsis (\u2026) to mark the title provisional; complete "
    "it, dropping the ellipsis, on the next pass. A complete title never carries one. "
    "A single offhand remark belonging to neither slide is not a transition -- start a "
    "new slide for it only once it grows into a discussion of its own. "
    # Measured: asked for the title as a one-shot action, the model named the deck
    # from the first four words heard -- a greeting or a run-up, frozen for the
    # whole talk. The title is answered on every call instead, last, with the
    # content already written, so it is judged against everything said so far.
    "Finally, 'deck_title' is the deck's opening slide: the subject of the whole talk in "
    "a few words, as a poster would carry it. It is never the speaker's opening words -- "
    "a greeting, a joke, or the run-up such as 'today I want to talk about' -- and a "
    "fragment is not a subject. Leave it empty until the speaker has actually said what "
    "the talk is about -- it can wait, so it is always complete and never ends in an "
    "ellipsis. You are shown the current one as 'deck_title'; repeat it unchanged "
    "unless the talk has plainly turned out to be about something broader or different, "
    "in which case give the better title."
)

EDITOR_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "slide_edit",
        "strict": True,
        "schema": {
            "type": "object",
            # Order matters: these are generated in sequence, so naming the subject
            # and judging the boundary happen before any slide content exists to
            # be anchored on. Reversing them puts the decision after the answer.
            "properties": {
                "recent_subject": {
                    "type": "string",
                    "description": "What recent_speech is about, in a few words. Empty if it is only filler.",
                },
                "continues_current_slide": {
                    "type": "boolean",
                    "description": "True only if recent_subject is the subject the current slide already covers.",
                },
                "action": {"type": "string", "enum": ["none", "update", "new"]},
                "title": {
                    "type": "string",
                    "description": "The slide's title: a complete phrase, ending with \u2026 only while provisional.",
                },
                "bullets": {"type": "array", "items": {"type": "string"}},
                # Last on purpose: the subject of the whole talk is judged after
                # the content, with everything said so far in view.
                "deck_title": {
                    "type": "string",
                    "description": (
                        "The subject of the whole talk in a few words. "
                        "Empty until the speaker has said what it is about."
                    ),
                },
            },
            "required": ["recent_subject", "continues_current_slide", "action", "title", "bullets", "deck_title"],
            "additionalProperties": False,
        },
    },
}


class TranscriptStream:
    """A live transcript from a Realtime transcription session.

    Deltas and completions are both handled because the cadence is not
    guaranteed: the model may stream text as it arrives, or only emit a final
    transcript when a turn closes. Deltas accumulate per item and the matching
    completion replaces them, so neither is double counted and the transcript is
    correct whichever the server actually sends. Manual commits give the editor a
    floor on freshness even if no delta ever arrives.
    """

    def __init__(self, api_key: str, model: str = TRANSCRIBE_MODEL, use_vad: bool = False) -> None:
        trace(f"Connecting to Realtime transcription ({model})...")
        self.socket = websocket.create_connection(
            TRANSCRIBE_URL, header=[f"Authorization: Bearer {api_key}"], timeout=20
        )
        self.socket.settimeout(None)
        self.use_vad = use_vad
        self._send_lock = threading.Lock()
        self._lock = threading.RLock()
        self.settled: list[str] = []
        self.pending: dict[str, str] = {}
        self.stop = threading.Event()
        self.send(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": AUDIO_RATE},
                            "transcription": {"model": model},
                            "turn_detection": {"type": "server_vad"} if use_vad else None,
                        }
                    },
                },
            }
        )
        trace("Transcription session open.")

    def send(self, event: dict[str, Any]) -> None:
        with self._send_lock:
            self.socket.send(json.dumps(event))

    def mark(self) -> int:
        """A position in the transcript to measure a slide's span from."""
        with self._lock:
            return len(self.settled)

    def text_since(self, mark: int) -> str:
        with self._lock:
            parts = [*self.settled[mark:], *self.pending.values()]
        return " ".join(part.strip() for part in parts if part.strip())

    def discard_pending(self) -> None:
        """Forget partial audio text when the presenter resets the deck."""
        with self._lock:
            self.pending.clear()

    def read_events(self) -> None:
        while not self.stop.is_set():
            try:
                event = json.loads(self.socket.recv())
            except Exception:  # noqa: BLE001 - a closed socket simply ends the stream
                return
            kind = event.get("type", "")
            if kind.endswith("input_audio_transcription.delta"):
                with self._lock:
                    item = event.get("item_id", "")
                    self.pending[item] = self.pending.get(item, "") + event.get("delta", "")
            elif kind.endswith("input_audio_transcription.completed"):
                text = (event.get("transcript") or "").strip()
                with self._lock:
                    self.pending.pop(event.get("item_id", ""), None)
                    if text:
                        self.settled.append(text)
                        trace(f"HEARD: {text}")
            elif kind.endswith("input_audio_transcription.failed"):
                trace(f"Transcription failed: {json.dumps(event.get('error', event))[:160]}")
            elif kind == "error":
                trace(f"Transcription error: {json.dumps(event.get('error', event))[:160]}")

    def close(self) -> None:
        self.stop.set()
        self.socket.close()


def edit_deck(api_key: str, model: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Ask the editor what the current slide should now say. Blocking; own thread."""
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": EDITOR_SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            "response_format": EDITOR_SCHEMA,
            "max_completion_tokens": EDITOR_MAX_TOKENS,
        }
    ).encode()
    request = Request(
        EDITOR_URL,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=EDITOR_TIMEOUT) as response:  # noqa: S310 - fixed OpenAI endpoint
            reply = json.load(response)
        return json.loads(reply["choices"][0]["message"]["content"])
    except (OSError, ValueError, KeyError, IndexError) as exc:
        trace(f"EDITOR FAILED: {exc}")
        return None


def request_slide_image(slide: dict[str, Any]) -> None:
    """Illustrate every titled or populated slide once, in the background."""
    if (slide["title"].strip() or slide["bullets"]) and not slide.get("image_requested"):
        slide["image_requested"] = True
        # Hand over a snapshot, not the slide: naming the query is a network call
        # and this runs with DECK_LOCK held, so none of it can happen here.
        IMAGE_WORKERS.submit(load_slide_image, slide, slide["title"], list(slide["bullets"]))


def continued_title(title: str) -> str:
    base = title.removesuffix(ELLIPSIS).rstrip()
    return title if title.lower().endswith("(continued)") else f"{base} (continued)"


def fill_slide(slide: dict[str, Any], title: str, bullets: list[str]) -> bool:
    """Write a slide, rolling anything past the cap onto continuation slides.

    A full slide must not silently swallow what the speaker just said. Dropping
    the sixth bullet loses the point entirely, so the overflow starts the next
    slide instead -- which is what the speaker would have wanted anyway.

    Returns whether the deck grew, because that means the slide being written is
    no longer the one the caller started with.
    """
    with DECK_LOCK:
        head, overflow = bullets[:MAX_BULLETS_PER_SLIDE], bullets[MAX_BULLETS_PER_SLIDE:]
        if (slide["title"], slide["bullets"]) != (title, head):
            old_title, old_bullets = slide["title"], list(slide["bullets"])
            if old_title != title and is_provisional(old_title):
                # The picture was chosen for a fragment; choose it again for the subject.
                slide.pop("image", None)
                slide.pop("image_requested", None)
            slide["title"] = title
            slide["bullets"] = head
            request_slide_image(slide)
            trace(f"SLIDE #{SLIDES.index(slide) + 1} '{title}': {len(head)} bullet(s)")
            changed = [str(index + 1) for index, (old, new) in enumerate(zip(old_bullets, head)) if old != new]
            detail = (
                f"title={'changed' if old_title != title else 'unchanged'}, "
                f"bullets {len(old_bullets)}→{len(head)}, "
                f"revised={','.join(changed) or 'none'}, "
                f"added={max(0, len(head) - len(old_bullets))}, "
                f"removed={max(0, len(old_bullets) - len(head))}"
            )
            trace(f"SLIDE CHANGE #{SLIDES.index(slide) + 1}: {detail}")

        spilled = False
        rolling = title
        while overflow:
            spilled = True
            rolling = continued_title(rolling)
            slide = append_content_slide(rolling)
            slide["bullets"] = overflow[:MAX_BULLETS_PER_SLIDE]
            overflow = overflow[MAX_BULLETS_PER_SLIDE:]
            request_slide_image(slide)
            trace(f"SLIDE #{len(SLIDES)} '{rolling}': {len(slide['bullets'])} bullet(s) rolled over")
        return spilled


def apply_edit(edit: dict[str, Any], stream: TranscriptStream, mark: int) -> int:
    """Fold one editor decision into the deck; return the new transcript mark."""
    action = edit.get("action", "none")
    title = tidy_title(edit.get("title") or "")
    bullets = [b.strip() for b in edit.get("bullets", []) if b and b.strip()]
    deck_title = tidy_title(edit.get("deck_title") or "")

    with DECK_LOCK:
        # The title stands apart from the action, so naming the talk never costs
        # a tick of content. On an empty deck ``mark`` is where the talk began,
        # so the floor measures the whole of it; once anything is on the deck,
        # the subject has been spoken about and the floor no longer applies.
        if deck_title and (SLIDES or len(stream.text_since(mark).split()) >= MIN_WORDS_FOR_TITLE):
            create_title_slide(deck_title)
    if action == "none" or not title:
        return mark
    with DECK_LOCK:
        if action == "new":
            # CMD_NEXT creates an empty visible destination immediately. The
            # next editor result belongs there; appending again would leave a
            # blank slide behind and make the deck jump two slides.
            current = current_content_slide()
            target = (
                current
                if current is not None and not current["title"].strip() and not current["bullets"]
                else append_content_slide(title)
            )
            fill_slide(target, title, bullets)
            # A new slide means the previous one is finished: measure the next
            # slide's span from here rather than replaying the whole talk.
            return stream.mark()

        slide = current_content_slide()
        if slide is None:
            slide = append_content_slide(title)
        if fill_slide(slide, title, bullets):
            # The overflow opened a slide of its own, so the span that produced
            # it belongs to the slide before. Start the next one from here, or
            # the editor would regenerate the same bullets and split again.
            return stream.mark()
    return mark


# --------------------------------------------------------------------------
# Audio input
#
# Two sources, one shape: both hand raw 24 kHz mono PCM to a deliver() callback.
# The ESP32 arrives over a Bluetooth serial link in the frame format defined by
# parse_audio.py, which is imported rather than restated so the protocol has
# exactly one definition.
# --------------------------------------------------------------------------

ESP32_RATE = 16000  # measured off the device with tools/esp32_probe.py, not assumed
ESP32_BAUD = 115200
# The session only accepts 24 kHz -- 16, 8 and 48 kHz are all rejected -- so
# ESP32 frames have to be stretched 3:2 on the way through.
DELIVERY_BYTES = AUDIO_RATE // 10 * 2  # hand over ~100 ms at a time, like a sound card


class Resampler:
    """Linear resampler that keeps its phase between calls.

    Each ESP32 frame is only 16 ms long, so resampling them independently would
    restart the interpolation 62 times a second and put a click at every frame
    boundary. Carrying the sub-sample position and the previous frame's last
    sample makes the blocks join into one continuous signal.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.step = source_rate / target_rate
        self.position = 0.0
        self.carry = 0

    def __call__(self, samples: array.array) -> bytes:
        extended = array.array("h", [self.carry])
        extended.extend(samples)
        out = array.array("h")
        position = self.position
        limit = len(extended) - 1
        while position < limit:
            index = int(position)
            first = extended[index]
            out.append(int(first + (extended[index + 1] - first) * (position - index)))
            position += self.step
        self.carry = extended[-1]
        self.position = position - limit
        return out.tobytes()


@contextlib.contextmanager
def esp32_audio(args: argparse.Namespace, deliver: Callable[[bytes], None],
                on_button: Callable[[str], None], paused: threading.Event):
    try:
        import serial

        import parse_audio
    except ImportError as exc:
        raise SystemExit("ESP32 input needs pyserial: pipenv install") from exc

    try:
        port = args.serial_port or parse_audio.find_port()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    trace(f"Opening the ESP32 stream on {port} ({ESP32_RATE} Hz -> {AUDIO_RATE} Hz)...")
    try:
        link = serial.Serial(port, args.esp32_baud, timeout=1)
        # Whatever a previous reader left buffered is mid-frame, and consuming it
        # lands one byte into a header -- which then decodes as a button press.
        link.reset_input_buffer()
    except serial.SerialException as exc:
        raise SystemExit(f"Could not open {port}: {exc}") from exc

    stop = threading.Event()

    # Only these bits can legitimately appear in a button snapshot. Anything else
    # means we are not looking at one: a dropped byte slides the frame and the
    # header magic lands in this position, where 0xBB and 0xDD both decode as
    # WIPE -- clearing the deck mid-talk because the link hiccuped.
    button_mask = 0
    for bit, _ in parse_audio.BUTTONS:
        button_mask |= bit

    def pump() -> None:
        resample = Resampler(ESP32_RATE, AUDIO_RATE)
        pending = bytearray()
        # Seeded from the first frame, not from zero: whatever the buttons read
        # at startup is their resting state, and treating that as a press fires
        # a phantom command before the speaker has touched anything.
        previous: int | None = None
        desyncs = 0
        while not stop.is_set():
            try:
                parse_audio.wait_for_header(link)
                frame = parse_audio.read_exact(link, 1 + parse_audio.AUDIO_PAYLOAD_SIZE)
            except (OSError, serial.SerialException) as exc:
                if not stop.is_set():
                    trace(f"ESP32 link lost: {exc}")
                return
            if frame is None:
                continue  # truncated mid-frame; wait_for_header resynchronises

            if frame[0] & ~button_mask:
                # Impossible as a button snapshot, so the frame is misaligned.
                # Drop it, and re-seed the edge detector so the first good frame
                # after the gap is treated as a resting state, not a press.
                desyncs += 1
                if desyncs in (1, 10, 100):
                    trace(f"ESP32 frame out of step ({desyncs}); resynchronising.")
                previous = None
                continue

            # Rising edges only, so a held button fires once.
            if previous is not None:
                for bit, command in parse_audio.BUTTONS:
                    if frame[0] & ~previous & bit:
                        on_button(command.decode())
            previous = frame[0]

            if paused.is_set():
                continue
            samples = array.array("h", frame[1:])
            if sys.byteorder == "big":
                samples.byteswap()
            pending += resample(samples)
            if len(pending) >= DELIVERY_BYTES:
                deliver(bytes(pending))
                pending.clear()

    threading.Thread(target=pump, daemon=True, name="esp32-reader").start()
    try:
        yield
    finally:
        stop.set()
        link.close()


# CMD_PAUSE toggles, which is what a single hardware button needs. The display
# must not use it: a viewer browsing back has no idea which way a toggle will
# land, and guessing wrong mutes the talk for good. It says what it wants instead.
DECK_COMMANDS = ("CMD_WIPE", "CMD_PAUSE", "CMD_PAUSE_ON", "CMD_PAUSE_OFF", "CMD_NEXT")
COMMAND_PORT = 5005  # the port parse_audio.py already publishes button presses on


def deck_command_handler(stream: TranscriptStream, state: dict[str, Any],
                         state_lock: threading.Lock, paused: threading.Event
                         ) -> Callable[[str], None]:
    """Build the callback that ESP32 buttons and UDP commands both drive.

    Kept out of run_transcription_mode so the three deck actions can be
    exercised without a microphone, a serial link, or an API key.
    """

    def close_current_slide() -> None:
        # Moving the mark and bumping the generation together means the next
        # edit starts from a blank slide, and any edit still in flight against
        # the old one is discarded rather than landing on the new one.
        with state_lock:
            state["mark"] = state["seen"] = stream.mark()
            state["generation"] += 1
            state["last_text"] = ""

    def handle(command: str) -> None:
        if command == "CMD_WIPE":
            wipe_deck()
            stream.discard_pending()
            close_current_slide()
            trace("COMMAND CMD_WIPE: deck cleared.")
        elif command == "CMD_NEXT":
            # Create the visible destination at once. It remains intentionally
            # blank until subsequent speech gives the editor enough context to
            # title and fill it, while the old slide is already frozen.
            with DECK_LOCK:
                append_content_slide("")
            close_current_slide()
            trace("COMMAND CMD_NEXT: blank slide created and selected.")
        elif command in ("CMD_PAUSE", "CMD_PAUSE_ON", "CMD_PAUSE_OFF"):
            wanted = not paused.is_set() if command == "CMD_PAUSE" else command == "CMD_PAUSE_ON"
            if wanted == paused.is_set():
                return  # already there; say nothing rather than repeat the trace
            paused.set() if wanted else paused.clear()
            trace(f"COMMAND {command}: "
                  + ("paused, audio is being dropped." if wanted else "resumed, audio is flowing again."))
        else:
            trace(f"Ignoring unknown deck command {command!r}")

    return handle


def listen_for_commands(on_command: Callable[[str], None], port: int,
                        stop: threading.Event) -> None:
    """Accept deck commands over UDP.

    parse_audio.py already sends button presses here and nothing was listening.
    Anything that speaks the same datagrams can drive the deck, which is also
    how these actions get tested when the hardware is not reporting presses.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        trace(f"Deck commands unavailable on port {port}: {exc}")
        sock.close()
        return
    sock.settimeout(0.5)
    trace(f"Deck commands accepted on udp://127.0.0.1:{port} ({', '.join(DECK_COMMANDS)})")
    try:
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(64)
            except TimeoutError:
                continue
            except OSError:
                return
            on_command(data.decode("ascii", "replace").strip())
    finally:
        sock.close()


@contextlib.contextmanager
def audio_source(args: argparse.Namespace, deliver: Callable[[bytes], None],
                 on_button: Callable[[str], None], paused: threading.Event):
    """Feed 24 kHz mono PCM to `deliver` from the ESP32 or the local microphone."""
    if getattr(args, "esp32", False):
        with esp32_audio(args, deliver, on_button, paused):
            yield
        return

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Install Python dependencies first: pipenv install") from exc
    except OSError as exc:
        raise SystemExit(PORTAUDIO_HINT) from exc

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            trace(f"Microphone status: {status}")
        if not paused.is_set():
            deliver(bytes(indata))

    device = getattr(args, "device", None)
    trace(f"Opening the {AUDIO_RATE // 1000} kHz mono PCM input"
          + (f" '{device}'..." if device is not None else " (system default)..."))
    with sd.RawInputStream(samplerate=AUDIO_RATE, blocksize=2400, channels=1,
                           dtype="int16", device=device, callback=callback):
        yield


def run_transcription_mode(api_key: str, args: argparse.Namespace) -> None:
    """Stream audio to a transcription session; edit the deck on a timer."""
    stream = TranscriptStream(api_key, args.transcribe_model, args.vad)
    pool = ThreadPoolExecutor(max_workers=EDITOR_WORKERS, thread_name_prefix="deck-editor")
    # Edits overlap so one slow call cannot stall the deck, which means answers
    # can come back out of order. ``sequence`` drops an answer a newer one has
    # already superseded; ``generation`` drops one written against a slide that
    # has since been closed, whose transcript span now belongs to the slide before.
    state = {
        "mark": 0, "generation": 0, "applied": 0, "issued": 0, "in_flight": 0,
        "last_text": "", "seen": 0,
    }
    state_lock = threading.Lock()

    DECK_PAUSED.clear()
    on_button = deck_command_handler(stream, state, state_lock, DECK_PAUSED)
    global ACTIVE_DECK_COMMAND
    ACTIVE_DECK_COMMAND = on_button
    appended = threading.Event()

    def deliver(raw_audio: bytes) -> None:
        appended.set()
        stream.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(raw_audio).decode("ascii"),
            }
        )

    def commit_loop() -> None:
        # Measured: gpt-live-transcribe streams deltas roughly every 0.2s
        # regardless of when turns close, so committing is not what keeps the
        # editor fed. It only decides where sentences are cut -- and cutting
        # every couple of seconds costs accuracy, because the transcriber loses
        # the surrounding words it would otherwise use to decide what it heard.
        # So commit rarely: often enough to bound a turn, seldom enough to let
        # whole sentences through.
        while not stream.stop.wait(args.commit_seconds):
            # Committing a buffer nothing was appended to is an API error, which
            # happens whenever the link is quiet or the speaker has paused.
            if not appended.is_set():
                continue
            appended.clear()
            stream.send({"type": "input_audio_buffer.commit"})

    def build_payload(spoken: str, recent: str) -> dict[str, Any]:
        with DECK_LOCK:
            current = current_content_slide()
            finished = [item for item in SLIDES if item is not current]
            detailed = finished[-DETAILED_SLIDES_IN_PAYLOAD:] if DETAILED_SLIDES_IN_PAYLOAD else []
            earlier = finished[: len(finished) - len(detailed)]
            return {
                "earlier_slide_titles": [item["title"] for item in earlier],
                "recent_slides": [
                    {"title": item["title"], "bullets": list(item["bullets"])} for item in detailed
                ],
                "deck_title": (title_slide() or {}).get("title", ""),
                "current_slide": None
                if current is None
                else {"title": current["title"], "bullets": list(current["bullets"])},
                # Where the speaker is now, kept apart from the whole span. The
                # span is mostly the current subject by construction, and the
                # editor judged the boundary by bulk when shown only that.
                "recent_speech": recent,
                "transcript_since_slide_started": spoken,
            }

    def run_edit(sequence: int, generation: int, payload: dict[str, Any]) -> None:
        # A reset or Next may happen while this job waits in the executor. Do
        # not spend an editor request on a deck generation that no longer exists.
        with state_lock:
            if generation != state["generation"]:
                state["in_flight"] -= 1
                trace(f"Discarding queued edit #{sequence}: deck was reset or advanced.")
                return
        try:
            edit = edit_deck(api_key, args.editor_model, payload)
        finally:
            with state_lock:
                state["in_flight"] -= 1
        if edit is None:
            # Let the same transcript be retried. Without this, a malformed or
            # length-truncated JSON response leaves last_text set forever and
            # the slide never receives the speech that caused the failure.
            with state_lock:
                if generation == state["generation"]:
                    state["last_text"] = ""
            return
        with state_lock:
            if sequence <= state["applied"] or generation != state["generation"]:
                trace(f"Discarding edit #{sequence}: a newer one already landed.")
                return
            trace(
                f"EDITOR #{sequence} (generation {generation}): heard={edit.get('recent_subject', '')!r}, "
                f"same slide={edit.get('continues_current_slide')}, action={edit.get('action')!r}, "
                f"title={edit.get('title', '')!r}, bullets={len(edit.get('bullets', []))}, "
                f"deck_title={edit.get('deck_title', '')!r}."
            )
            state["applied"] = sequence
            moved = apply_edit(edit, stream, state["mark"])
            if moved != state["mark"]:
                state["mark"] = state["seen"] = moved
                state["generation"] += 1
                state["last_text"] = ""

    def edit_loop() -> None:
        while not stream.stop.wait(args.editor_seconds):
            with state_lock:
                mark, generation = state["mark"], state["generation"]
                if state["in_flight"] >= EDITOR_WORKERS:
                    continue
            spoken = stream.text_since(mark)
            if len(spoken.split()) < MIN_WORDS_TO_EDIT:
                continue
            # Everything heard since the last decision. Only this loop touches
            # ``seen``, and only this loop runs here, so no lock is needed.
            recent = stream.text_since(state["seen"]) or spoken
            state["seen"] = stream.mark()
            with state_lock:
                # Re-asking the same question invites a different answer, and a
                # different answer with no new speech behind it is an invention.
                if spoken == state["last_text"] or generation != state["generation"]:
                    continue
                state["last_text"] = spoken
                state["issued"] += 1
                state["in_flight"] += 1
                sequence = state["issued"]
                trace(
                    f"QUEUE EDITOR #{sequence} (generation {generation}): "
                    f"{len(spoken.split())} words since the current slide began."
                )
            pool.submit(run_edit, sequence, generation, build_payload(spoken, recent))

    threading.Thread(target=stream.read_events, daemon=True, name="transcript").start()
    if args.commands_port:
        threading.Thread(target=listen_for_commands, daemon=True, name="deck-commands",
                         args=(on_button, args.commands_port, stream.stop)).start()
    with audio_source(args, deliver, on_button, DECK_PAUSED):
        trace(
            f"Audio is live. Transcribing continuously; the deck is revised every "
            f"{args.editor_seconds:g}s by {args.editor_model}. Press Ctrl-C to stop."
        )
        if not args.vad:
            threading.Thread(target=commit_loop, daemon=True, name="audio-commit-loop").start()
        threading.Thread(target=edit_loop, daemon=True, name="deck-editor").start()
        try:
            # time.sleep, not Event.wait: a blocking wait is not reliably
            # interrupted by Ctrl-C on Windows.
            while not stream.stop.is_set():
                time.sleep(0.25)
        finally:
            stream.close()
            pool.shutdown(wait=False, cancel_futures=True)


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
        "--transcribe",
        action="store_true",
        help=(
            "Transcribe continuously and revise the deck on a timer, instead of "
            "asking the speech-to-speech model to edit while it listens."
        ),
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help=(
            "With --transcribe, let the server segment turns instead of committing on a "
            "timer. Sentences come out cleaner, but no text arrives until you stop "
            f"speaking. Needs one of: {', '.join(VAD_CAPABLE_MODELS)}."
        ),
    )
    parser.add_argument(
        "--transcribe-model",
        default=TRANSCRIBE_MODEL,
        help=f"Transcription model for --transcribe (default: {TRANSCRIBE_MODEL}).",
    )
    parser.add_argument(
        "--esp32",
        action="store_true",
        help=(
            "Take audio from the ESP32 over Bluetooth serial instead of a local "
            "microphone, using the frame format in parse_audio.py. Its buttons wipe "
            "the deck, start a new slide, and pause."
        ),
    )
    parser.add_argument(
        "--serial-port",
        help="Serial port for --esp32 (default: auto-detect, as parse_audio.py does).",
    )
    parser.add_argument(
        "--esp32-baud",
        type=int,
        default=ESP32_BAUD,
        help=f"Baud for --esp32 (default: {ESP32_BAUD}; Bluetooth SPP ignores it).",
    )
    parser.add_argument(
        "--device",
        help="Input device name or index for the local microphone (default: system default).",
    )
    parser.add_argument(
        "--commands-port",
        type=int,
        default=COMMAND_PORT,
        help=(
            f"UDP port to accept deck commands on (default: {COMMAND_PORT}, the port "
            "parse_audio.py already sends button presses to). 0 disables it."
        ),
    )
    parser.add_argument(
        "--commit-seconds",
        type=float,
        default=TRANSCRIBE_COMMIT_SECONDS,
        help=(
            "With --transcribe and no --vad, how often to close a transcription turn "
            f"(default: {TRANSCRIBE_COMMIT_SECONDS:g}). Text still streams between commits; "
            "committing more often only chops sentences and costs accuracy."
        ),
    )
    parser.add_argument(
        "--editor-model",
        default=EDITOR_MODEL,
        help=f"Model that rewrites slides for --transcribe (default: {EDITOR_MODEL}).",
    )
    parser.add_argument(
        "--editor-seconds",
        type=float,
        default=1.0,
        help="How often --transcribe revises the current slide (default: 1.0).",
    )
    parser.add_argument(
        "--push",
        metavar="URL",
        help=(
            "Also publish the deck to a hosted display, e.g. "
            "https://your-app.vercel.app/api/slides. Requires SLIDEX_PUSH_TOKEN."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=2.0,
        help=(
            "Longest run of audio before a forced deck decision; a pause in speech "
            "commits sooner (default: 2.0)."
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this program.")

    if not args.microphone and not args.prompt and not args.transcribe:
        parser.error("provide a prompt, or use --microphone or --transcribe")
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be greater than zero")
    if args.editor_seconds <= 0:
        parser.error("--editor-seconds must be greater than zero")
    if args.commit_seconds <= 0:
        parser.error("--commit-seconds must be greater than zero")
    # Catch this here rather than letting the session be rejected mid-connect,
    # where the message is buried under two unrelated fallback attempts.
    if args.vad and args.transcribe_model not in VAD_CAPABLE_MODELS:
        parser.error(
            f"{args.transcribe_model} does not support turn detection. Either drop --vad, "
            f"or pass --transcribe-model with one of: {', '.join(VAD_CAPABLE_MODELS)}"
        )

    push_token = os.environ.get("SLIDEX_PUSH_TOKEN", "")
    if args.push and not push_token:
        parser.error("--push needs SLIDEX_PUSH_TOKEN set to the same secret as the deployment")

    web_server = start_web_server(args.web_port) if args.web else None
    stop_publisher = start_deck_publisher(args.push, push_token) if args.push else None

    if args.transcribe:
        try:
            run_transcription_mode(api_key, args)
        except KeyboardInterrupt:
            trace("Stopped.")
        finally:
            IMAGE_WORKERS.shutdown(wait=False, cancel_futures=True)
            if stop_publisher is not None:
                stop_publisher.set()
            if web_server is not None:
                web_server.shutdown()
                web_server.server_close()
        return

    client = RealtimeToolClient(api_key)
    try:
        if args.microphone:
            # trace("Starting continuous microphone mode.")
            stream_microphone(client, args)
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
