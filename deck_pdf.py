"""Write a deck snapshot to a single PDF.

The reset button clears the deck, and a talk that has just been given is the
one thing the speaker cannot say again -- so the deck is written out first.
Pages are 16:9 and follow the display's layout: a title slide with a washed-out
background picture, content slides with the picture in a column beside the text.

Only ``export_deck`` is meant to be called from outside. It downloads the slide
pictures, so it is blocking and belongs on a background thread.
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = 960.0, 540.0  # points; the display's 16:9 frame
PADDING = 44.0  # the display's 4.6cqw
IMAGE_TIMEOUT = 12
USER_AGENT = "Slidex/1.0 (https://github.com/adntaha/slidex)"
ELLIPSIS = "\u2026"

INK = HexColor("#202124")
MUTED = HexColor("#5f6368")
CREDIT_INK = HexColor("#3c4043")
WHITE = HexColor("#ffffff")

# The built-in PDF fonts stop at Latin-1, and the deck is written in whatever
# language the talk is in. A system TrueType font is embedded when one can be
# found; the pairs are (regular, bold), tried in order.
FONT_CANDIDATES = (
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
)
_fonts: tuple[str, str] | None = None


def fonts() -> tuple[str, str]:
    """Return the (regular, bold) font names to draw with, registering them once."""
    global _fonts
    if _fonts is None:
        _fonts = ("Helvetica", "Helvetica-Bold")
        for regular, bold in FONT_CANDIDATES:
            if Path(regular).is_file() and Path(bold).is_file():
                try:
                    pdfmetrics.registerFont(TTFont("Deck", regular))
                    pdfmetrics.registerFont(TTFont("Deck-Bold", bold))
                except Exception:  # noqa: BLE001 - a broken font file just means the next candidate
                    continue
                _fonts = ("Deck", "Deck-Bold")
                break
    return _fonts


def fetch_image(url: str) -> ImageReader | None:
    """Download one slide picture; None leaves the slide unillustrated."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=IMAGE_TIMEOUT) as response:  # noqa: S310 - the deck's own image URLs
            data = response.read()
        reader = ImageReader(io.BytesIO(data))
        reader.getSize()  # decode now, so a corrupt file fails here and not mid-page
        return reader
    except Exception:  # noqa: BLE001 - any failure means the page is drawn without the picture
        return None


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    """Greedy word wrap; a single word wider than the line stands alone."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and pdfmetrics.stringWidth(candidate, font, size) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def fit(text: str, font: str, largest: float, smallest: float, width: float,
        max_lines: int) -> tuple[float, list[str]]:
    """Shrink the type until the text fits the box, as the display does."""
    size = largest
    while True:
        lines = wrap(text, font, size, width)
        if len(lines) <= max_lines or size <= smallest:
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] = lines[-1].rstrip(".,;: ") + ELLIPSIS
            return size, lines
        size -= 1


def draw_cover(c: canvas.Canvas, image: ImageReader, x: float, y: float,
               width: float, height: float, alpha: float = 1.0) -> None:
    """Draw an image filling the box, cropped rather than squashed (CSS object-fit: cover)."""
    iw, ih = image.getSize()
    scale = max(width / iw, height / ih)
    dw, dh = iw * scale, ih * scale
    c.saveState()
    path = c.beginPath()
    path.rect(x, y, width, height)
    c.clipPath(path, stroke=0, fill=0)
    if alpha < 1.0:
        c.setFillAlpha(alpha)
    c.drawImage(image, x + (width - dw) / 2, y + (height - dh) / 2, dw, dh, mask="auto")
    c.restoreState()


def draw_credit(c: canvas.Canvas, image: dict[str, Any]) -> None:
    regular, _ = fonts()
    text = f"Image: {image.get('alt', '')} · Wikimedia Commons"
    size = 7.0
    limit = PAGE_WIDTH * 0.52
    while pdfmetrics.stringWidth(text, regular, size) > limit and len(text) > 20:
        text = text[:-2].rstrip() + ELLIPSIS
    x, y = 26.0, 20.0
    width = pdfmetrics.stringWidth(text, regular, size)
    c.setFillColor(WHITE)
    c.rect(x - 5, y - 3.5, width + 10, size + 7, stroke=0, fill=1)
    c.setFillColor(CREDIT_INK)
    c.setFont(regular, size)
    c.drawString(x, y, text)
    if image.get("source"):
        c.linkURL(image["source"], (x - 5, y - 3.5, x + 5 + width, y + size + 3.5))


def draw_number(c: canvas.Canvas, number: int) -> None:
    regular, _ = fonts()
    c.setFillColor(MUTED)
    c.setFont(regular, 8)
    c.drawRightString(PAGE_WIDTH - PADDING, 20, f"SLIDE {number:02d}")


def draw_title_slide(c: canvas.Canvas, slide: dict[str, Any], image: ImageReader | None) -> None:
    _, bold = fonts()
    if image is not None:
        draw_cover(c, image, 0, 0, PAGE_WIDTH, PAGE_HEIGHT, alpha=0.24)
    size, lines = fit(slide["title"], bold, 58, 30, PAGE_WIDTH - 2 * PADDING, 3)
    leading = size * 1.02
    top = PAGE_HEIGHT / 2 + (len(lines) * leading) / 2 - size * 0.8
    c.setFillColor(INK)
    c.setFont(bold, size)
    for index, line in enumerate(lines):
        c.drawCentredString(PAGE_WIDTH / 2, top - index * leading, line)
    if image is not None and slide.get("image"):
        draw_credit(c, slide["image"])


def draw_content_slide(c: canvas.Canvas, slide: dict[str, Any], image: ImageReader | None) -> None:
    regular, bold = fonts()
    picture = slide.get("image") if image is not None else None
    placement = (picture or {}).get("placement")
    text_left, text_width = PADDING, PAGE_WIDTH - 2 * PADDING
    if placement in ("left", "right"):
        # The display gives the picture a column of the frame beside the text.
        column = (PAGE_WIDTH - 2 * PADDING) * 0.40
        gap = 30.0
        picture_height = PAGE_HEIGHT * 0.74
        picture_y = (PAGE_HEIGHT - picture_height) / 2
        if placement == "right":
            picture_x = PAGE_WIDTH - PADDING - column
        else:
            picture_x = PADDING
            text_left = PADDING + column + gap
        text_width = PAGE_WIDTH - 2 * PADDING - column - gap
        draw_cover(c, image, picture_x, picture_y, column, picture_height)
    elif placement == "background":
        draw_cover(c, image, 0, 0, PAGE_WIDTH, PAGE_HEIGHT, alpha=0.14)

    title_size, title_lines = fit(slide["title"] or " ", bold, 40, 22, text_width, 3)
    bullets = [b for b in slide.get("bullets", []) if b.strip()]
    bullet_size = 17.0
    while True:
        bullet_lines = [wrap(b, regular, bullet_size, text_width - 22) for b in bullets]
        block = (len(title_lines) * title_size * 0.98 + (bullet_size * 1.7 if bullets else 0)
                 + sum(len(lines) * bullet_size * 1.35 + bullet_size * 0.55 for lines in bullet_lines))
        if block <= PAGE_HEIGHT - 2 * PADDING or bullet_size <= 10:
            break
        bullet_size -= 1

    y = PAGE_HEIGHT / 2 + block / 2 - title_size * 0.85
    c.setFillColor(INK)
    c.setFont(bold, title_size)
    for line in title_lines:
        c.drawString(text_left, y, line)
        y -= title_size * 0.98
    if bullets:
        y -= bullet_size * 0.7
        c.setFont(regular, bullet_size)
        for lines in bullet_lines:
            c.setFillColor(MUTED)
            c.drawString(text_left, y, "•")
            c.setFillColor(INK)
            for line in lines:
                c.drawString(text_left + 22, y, line)
                y -= bullet_size * 1.35
            y -= bullet_size * 0.55

    if picture:
        draw_credit(c, picture)
    draw_number(c, slide.get("number", 0))


def slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower().replace(ELLIPSIS, "")).strip("-")
    return cleaned[:40].rstrip("-") or "deck"


def export_deck(deck: dict[str, Any], folder: Path,
                fetch: Callable[[str], ImageReader | None] = fetch_image) -> Path:
    """Write every slide of ``deck`` (a deck_snapshot) to one PDF in ``folder``.

    The file is named after the deck's title and the time, so consecutive talks
    never overwrite each other. Returns the path written.
    """
    slides = deck.get("slides", [])
    if not slides:
        raise ValueError("The deck is empty; nothing to export.")
    folder.mkdir(parents=True, exist_ok=True)
    title = next((s["title"] for s in slides if s.get("kind") == "title"), "") or slides[0]["title"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stamp}-{slug(title)}.pdf"

    c = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    c.setTitle(title.replace(ELLIPSIS, "").strip() or "SlideX deck")
    c.setAuthor("SlideX")
    for slide in slides:
        image = fetch(slide["image"]["url"]) if slide.get("image") else None
        c.setFillColor(WHITE)
        c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
        if slide.get("kind") == "title":
            draw_title_slide(c, slide, image)
        else:
            draw_content_slide(c, slide, image)
        c.showPage()
    c.save()
    return path
