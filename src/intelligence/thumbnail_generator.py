"""
YouTube thumbnail generator for OSIA intelligence products.

Renders 1280×720 PNG thumbnails using the same WeasyPrint + PyMuPDF
pipeline as the slide renderer.  Thumbnails are generated alongside:

  - Intro webcast videos   (via intro_video_generator)
  - INTSUM PDF reports     (via report_generator)
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.intelligence.aesthetic import (
    desk_accent_colour,
    load_desk_badge_b64,
    load_desk_bg_b64,
    load_logo_b64,
    load_portrait_b64,
)

logger = logging.getLogger("osia.thumbnail_generator")

_REPO_ROOT = Path(__file__).parent.parent.parent
_TEMPLATES_DIR = _REPO_ROOT / "templates" / "report"

# YouTube standard thumbnail resolution
_THUMB_W = 1280
_THUMB_H = 720

# Title font size thresholds — shorter titles get larger type
_FONT_SIZE_SHORT = 84   # ≤ 30 chars
_FONT_SIZE_MEDIUM = 72  # ≤ 55 chars
_FONT_SIZE_LONG = 54    # > 55 chars

_MAX_TITLE_CHARS = 80
_MAX_SUBTITLE_CHARS = 110


def _title_font_size(title: str) -> int:
    n = len(title)
    if n <= 30:
        return _FONT_SIZE_SHORT
    if n <= 55:
        return _FONT_SIZE_MEDIUM
    return _FONT_SIZE_LONG


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "\u2026"


def _desk_display_name(slug: str) -> str:
    return slug.replace("-", " ").title()


def extract_title_from_intsum(analysis: str) -> str:
    """Extract a short, readable title from an INTSUM markdown document.

    Tries, in order:
    1. First Markdown heading (``# ...`` / ``## ...``)
    2. First non-empty non-comment line
    3. Static fallback "Intelligence Summary"
    """
    for line in analysis.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return _truncate(title, _MAX_TITLE_CHARS)
    for line in analysis.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("<!--") and not stripped.startswith("---"):
            return _truncate(stripped, _MAX_TITLE_CHARS)
    return "Intelligence Summary"


def generate_thumbnail(
    title: str,
    desk_slug: str,
    output_path: Path,
    *,
    subtitle: str | None = None,
    category_label: str | None = None,
    bg_category: str | None = None,
    show_portrait: bool = True,
    show_badge: bool = True,
    date_label: str | None = None,
) -> Path:
    """Render a 1280×720 YouTube thumbnail to *output_path* (PNG).

    Args:
        title:          Main headline shown on the thumbnail.
        desk_slug:      Kebab-case desk identifier — drives accent colour,
                        badge, background, and portrait selection.
        output_path:    Destination path (must end in ``.png``).
        subtitle:       Optional second line of text.
        category_label: Small all-caps label above the title
                        (e.g. ``"Intelligence Report"``).
        bg_category:    Override for the background category
                        (hero / terrain / archive / data_overlay / ecological).
                        Defaults to the desk's canonical category.
        show_portrait:  Whether to render the desk head's portrait on the
                        right-hand side.  Silently ignored if no portrait
                        asset exists for the given desk.
        date_label:     Optional date string for the footer
                        (e.g. ``"2026-04-06"``).  Defaults to today.

    Returns:
        Path to the generated PNG file.
    """
    # Lazy imports — keep module startup fast
    import fitz  # noqa: PLC0415 — PyMuPDF
    from weasyprint import HTML  # noqa: PLC0415

    if os.getenv("OSIA_DEBUG_PDF", "false").lower() != "true":
        logging.getLogger("weasyprint").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("thumbnail.html.j2")

    portrait_uri = load_portrait_b64(desk_slug) if show_portrait else None
    badge_uri = load_desk_badge_b64(desk_slug) if show_badge else None
    accent = desk_accent_colour(desk_slug)
    bg_uri = load_desk_bg_b64(bg_category or desk_slug, orientation="landscape")
    truncated_title = _truncate(title, _MAX_TITLE_CHARS)

    html_content = template.render(
        title=truncated_title,
        subtitle=_truncate(subtitle, _MAX_SUBTITLE_CHARS) if subtitle else None,
        category_label=category_label,
        desk_name=_desk_display_name(desk_slug) if desk_slug else None,
        desk_accent=accent,
        logo_data_uri=load_logo_b64(),
        badge_data_uri=badge_uri,
        portrait_data_uri=portrait_uri,
        show_portrait=bool(portrait_uri),
        show_badge=show_badge and bool(badge_uri),
        bg_image_data_uri=bg_uri,
        title_font_size=_title_font_size(truncated_title),
        date_label=date_label or datetime.now(UTC).strftime("%Y-%m-%d"),
        width=_THUMB_W,
        height=_THUMB_H,
    )

    pdf_bytes = HTML(string=html_content, base_url=str(_TEMPLATES_DIR)).write_pdf()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    # Scale 1280×720 at 96 DPI (PDF points are 72dpi, so ratio ≈ 1.333)
    mat = fitz.Matrix(96 / 72, 96 / 72)
    pix = doc[0].get_pixmap(matrix=mat)
    pix.save(str(output_path))
    doc.close()

    logger.info("Thumbnail saved: %s", output_path)
    return output_path


def generate_intro_thumbnail(output_dir: Path) -> Path:
    """Generate the standard OSIA intro-video YouTube thumbnail.

    Uses the hero background, no desk portrait, and OSIA branding.

    Args:
        output_dir: Directory where the thumbnail will be written.

    Returns:
        Path to the generated PNG file.
    """
    output_path = output_dir / "OSIA_Intro_Webcast_thumbnail.png"
    return generate_thumbnail(
        title="Open Source Intelligence Agency",
        desk_slug="the-watch-floor",  # amber accent
        output_path=output_path,
        subtitle="Autonomous · Multi-Desk · Open Source · Event-Driven",
        category_label="Intro Webcast",
        bg_category="hero",
        show_portrait=False,
        show_badge=False,
        date_label=datetime.now(UTC).strftime("%Y-%m-%d"),
    )


def generate_report_thumbnail(
    analysis: str,
    desk_slug: str,
    report_path: Path,
) -> Path:
    """Generate a YouTube thumbnail alongside an INTSUM PDF report.

    The title is extracted from the first Markdown heading in *analysis*.
    The thumbnail is saved next to the PDF with a ``_thumbnail.png`` suffix.

    Args:
        analysis:    Raw markdown INTSUM text.
        desk_slug:   Kebab-case desk identifier.
        report_path: Path to the generated PDF (used to derive the output path).

    Returns:
        Path to the generated PNG thumbnail.
    """
    title = extract_title_from_intsum(analysis)
    subtitle = _desk_display_name(desk_slug)
    output_path = report_path.with_name(report_path.stem + "_thumbnail.png")

    return generate_thumbnail(
        title=title,
        desk_slug=desk_slug,
        output_path=output_path,
        subtitle=subtitle,
        category_label="Intelligence Summary",
        show_portrait=True,
        date_label=datetime.now(UTC).strftime("%Y-%m-%d"),
    )
