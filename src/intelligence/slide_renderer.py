"""
Slide deck renderer for OSIA weekly department briefings.

Generates HTML slides via Jinja2 and renders them to PNG images
using WeasyPrint. Supports both landscape (16:9) and portrait (9:16) formats.
"""

import base64
import logging
from datetime import UTC, datetime
from pathlib import Path

import markdown as md
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger("osia.slide_renderer")

# WeasyPrint's font subsetting via fontTools is extremely verbose at DEBUG/INFO.
# Suppress it to WARNING so it doesn't drown out application logs.
logging.getLogger("fontTools").setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).parent.parent.parent
_TEMPLATES_DIR = _REPO_ROOT / "templates" / "report"
_ASSETS_DIR = _REPO_ROOT / "assets"

# Slide dimensions in pixels (rendered at 2x for crisp output)
DIMENSIONS = {
    "landscape": (1920, 1080),
    "portrait": (1080, 1920),
}


def _load_logo_b64() -> str | None:
    """Return the OSIA logo as a base64 data URI."""
    logo_path = _ASSETS_DIR / "osia_logo_sm.png"
    if logo_path.exists():
        raw = logo_path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode()
    return None


def _load_portrait_b64(desk_slug: str) -> str | None:
    """Return the department head portrait as a base64 data URI, if it exists."""
    portrait_path = _ASSETS_DIR / "portraits" / f"{desk_slug}.png"
    if portrait_path.exists():
        raw = portrait_path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode()
    return None


def _md_to_html(text: str) -> str:
    """Convert markdown text to HTML, handling bullet points and emphasis."""
    return md.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )


def _desk_display_name(slug: str) -> str:
    return slug.replace("-", " ").title()


class SlideRenderer:
    """Renders briefing slides to PNG images using WeasyPrint."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=True,
        )
        self._logo_uri = _load_logo_b64()

    def render_deck(
        self,
        slides: list[dict],
        desk_slug: str,
        desk_name: str,
        persona_name: str,
        orientation: str,
        output_dir: Path,
        week_label: str,
        bg_images: list[Path | None] | None = None,
    ) -> list[Path]:
        """Render all slides to PNG images.

        Args:
            slides: List of slide dicts with keys: title, body, narration, slide_type
            desk_slug: Desk identifier.
            desk_name: Human-readable desk name.
            persona_name: Name of the department head character.
            orientation: 'landscape' or 'portrait'.
            output_dir: Directory to write PNG files.
            week_label: e.g. '2026-W14'.
            bg_images: Optional list of Paths to background images (one per slide, None entries OK).

        Returns:
            List of Paths to rendered PNG images.
        """
        from weasyprint import HTML

        # Suppress verbose WeasyPrint logging
        logging.getLogger("weasyprint").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)

        output_dir.mkdir(parents=True, exist_ok=True)
        width, height = DIMENSIONS[orientation]

        template = self._env.get_template("briefing_slide.html.j2")
        now = datetime.now(UTC)
        ref_prefix = desk_slug[:8].upper().replace("-", "")
        ref_number = f"OSIA-WB-{now.strftime('%Y%m%d')}-{ref_prefix}"
        portrait_uri = _load_portrait_b64(desk_slug)

        image_paths: list[Path] = []

        for i, slide in enumerate(slides):
            body_html = _md_to_html(slide.get("body", ""))

            # Encode background image as data URI if available
            bg_image_data_uri = None
            if bg_images and i < len(bg_images) and bg_images[i] and bg_images[i].exists():
                raw = bg_images[i].read_bytes()
                bg_image_data_uri = "data:image/png;base64," + base64.b64encode(raw).decode()

            html_content = template.render(
                slide_title=slide.get("title", ""),
                slide_body=body_html,
                slide_type=slide.get("slide_type", "content"),
                slide_number=i + 1,
                total_slides=len(slides),
                desk_name=desk_name,
                desk_slug=desk_slug,
                persona_name=persona_name,
                week_label=week_label,
                ref_number=ref_number,
                orientation=orientation,
                width=width,
                height=height,
                logo_data_uri=self._logo_uri,
                bg_image_data_uri=bg_image_data_uri,
                portrait_data_uri=portrait_uri if slide.get("slide_type") == "title" else None,
                timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
            )

            png_path = output_dir / f"slide_{i:02d}.png"

            # WeasyPrint removed write_png() in v53; render to PDF bytes then
            # convert the first page to PNG via PyMuPDF at 150 DPI.
            import fitz  # pymupdf

            pdf_bytes = HTML(string=html_content, base_url=str(_TEMPLATES_DIR)).write_pdf()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI (PDF default is 72)
            pix = doc[0].get_pixmap(matrix=mat)
            pix.save(str(png_path))
            doc.close()

            image_paths.append(png_path)
            logger.debug("Rendered slide %d/%d: %s", i + 1, len(slides), png_path.name)

        logger.info(
            "Rendered %d %s slides for %s → %s",
            len(slides),
            orientation,
            desk_slug,
            output_dir,
        )
        return image_paths

    def render_pdf_deck(
        self,
        slides: list[dict],
        desk_slug: str,
        desk_name: str,
        persona_name: str,
        orientation: str,
        output_path: Path,
        week_label: str,
        bg_images: list[Path | None] | None = None,
    ) -> Path:
        """Render all slides into a single multi-page PDF.

        Args:
            slides: List of slide dicts.
            desk_slug: Desk identifier.
            desk_name: Human-readable desk name.
            persona_name: Department head character name.
            orientation: 'landscape' or 'portrait'.
            output_path: Path for the output PDF.
            week_label: e.g. '2026-W14'.
            bg_images: Optional list of Paths to background images (one per slide, None entries OK).

        Returns:
            Path to the generated PDF.
        """
        from weasyprint import HTML

        logging.getLogger("weasyprint").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = DIMENSIONS[orientation]

        template = self._env.get_template("briefing_slide.html.j2")
        now = datetime.now(UTC)
        ref_prefix = desk_slug[:8].upper().replace("-", "")
        ref_number = f"OSIA-WB-{now.strftime('%Y%m%d')}-{ref_prefix}"
        portrait_uri = _load_portrait_b64(desk_slug)

        # Render each slide as a separate HTML page, then combine
        all_pages = []
        for i, slide in enumerate(slides):
            body_html = _md_to_html(slide.get("body", ""))

            # Encode background image as data URI if available
            bg_image_data_uri = None
            if bg_images and i < len(bg_images) and bg_images[i] and bg_images[i].exists():
                raw = bg_images[i].read_bytes()
                bg_image_data_uri = "data:image/png;base64," + base64.b64encode(raw).decode()

            html_content = template.render(
                slide_title=slide.get("title", ""),
                slide_body=body_html,
                slide_type=slide.get("slide_type", "content"),
                slide_number=i + 1,
                total_slides=len(slides),
                desk_name=desk_name,
                desk_slug=desk_slug,
                persona_name=persona_name,
                week_label=week_label,
                ref_number=ref_number,
                orientation=orientation,
                width=width,
                height=height,
                logo_data_uri=self._logo_uri,
                bg_image_data_uri=bg_image_data_uri,
                portrait_data_uri=portrait_uri if slide.get("slide_type") == "title" else None,
                timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
            )
            all_pages.append(html_content)

        # Combine pages with page breaks
        combined_html = "\n".join(
            f'<div style="page-break-before: always;">{page}</div>' if i > 0 else page
            for i, page in enumerate(all_pages)
        )

        HTML(string=combined_html, base_url=str(_TEMPLATES_DIR)).write_pdf(str(output_path))
        logger.info("PDF deck: %s (%d slides)", output_path, len(slides))
        return output_path
