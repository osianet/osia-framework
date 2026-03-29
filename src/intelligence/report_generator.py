"""PDF report generator for OSIA intelligence summaries.

Converts markdown INTSUM text into a styled PDF document using WeasyPrint,
with an intelligence agency + cyber aesthetic template.
"""

import base64
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import markdown as md
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger("osia.report_generator")

_REPO_ROOT = Path(__file__).parent.parent.parent
_TEMPLATES_DIR = _REPO_ROOT / "templates" / "report"
_ASSETS_DIR = _REPO_ROOT / "assets"
_REPORTS_DIR = _REPO_ROOT / "reports"


def _load_logo_b64() -> str | None:
    """Return the OSIA logo as a base64 data URI, or None if the file is absent."""
    logo_path = _ASSETS_DIR / "osia_logo_sm.png"
    if logo_path.exists():
        raw = logo_path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode()
    return None


def _desk_display_name(slug: str) -> str:
    return slug.replace("-", " ").title()


def _apply_confidence_classes(html: str) -> str:
    """Wrap Source Confidence lines with styled spans for HIGH/MODERATE/LOW."""
    html = re.sub(
        r"(Source Confidence[^:]*?:?\s*)(HIGH)",
        r'\1<span class="source-confidence-high">\2</span>',
        html,
        flags=re.IGNORECASE,
    )
    html = re.sub(
        r"(Source Confidence[^:]*?:?\s*)(MODERATE)",
        r'\1<span class="source-confidence-moderate">\2</span>',
        html,
        flags=re.IGNORECASE,
    )
    html = re.sub(
        r"(Source Confidence[^:]*?:?\s*)(LOW)",
        r'\1<span class="source-confidence-low">\2</span>',
        html,
        flags=re.IGNORECASE,
    )
    return html


def generate_intsum_pdf(
    analysis: str,
    desk_slug: str,
    source: str = "internal",
    *,
    reports_dir: Path | None = None,
) -> Path:
    """Convert a markdown INTSUM analysis to a styled PDF report.

    Args:
        analysis:    Full markdown text of the intelligence summary.
        desk_slug:   Kebab-case desk identifier (e.g. ``the-watch-floor``).
        source:      Origin tag for the metadata bar (e.g. ``signal:+44...``).
        reports_dir: Override output directory (defaults to ``<repo>/reports/``).

    Returns:
        Path to the generated PDF file.
    """
    # Lazy import — WeasyPrint is only needed at call time, keeping startup fast.
    from weasyprint import HTML  # noqa: PLC0415

    # Suppress verbose WeasyPrint and fontTools logging unless explicitly enabled
    if os.getenv("OSIA_DEBUG_PDF", "false").lower() != "true":
        logging.getLogger("weasyprint").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)

    output_dir = reports_dir or _REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    file_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
    desk_safe = re.sub(r"[^a-zA-Z0-9_-]", "-", desk_slug)
    filename = f"{file_ts}_{desk_safe}.pdf"
    output_path = output_dir / filename

    # Convert markdown → HTML
    html_body = md.markdown(
        analysis,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )

    # Post-process: colour Source Confidence indicators
    html_body = _apply_confidence_classes(html_body)

    # Render Jinja2 template
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("intsum_report.html.j2")

    ref_suffix = desk_slug[:8].upper().replace("-", "")
    ref_number = f"OSIA-{now.strftime('%Y%m%d')}-{ref_suffix}-{now.strftime('%H%M')}"

    html_content = template.render(
        report_body=html_body,
        timestamp=now.strftime("%Y-%m-%dT%H:%M:%S UTC"),
        desk_name=_desk_display_name(desk_slug),
        desk_slug=desk_slug,
        ref_number=ref_number,
        source=source,
        logo_data_uri=_load_logo_b64(),
    )

    HTML(string=html_content, base_url=str(_TEMPLATES_DIR)).write_pdf(str(output_path))
    logger.info("PDF archived: %s", output_path)
    return output_path
