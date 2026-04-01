"""Template-based infographic renderer for OSIA intelligence briefs.

Generates a 9:16 portrait infographic card from structured findings using
Jinja2 + WeasyPrint + PyMuPDF — the same pipeline as the weekly briefing
slide renderer. Text is real HTML, so no AI-generated spelling errors.

The LLM extracts structured key findings from the report; the template
handles all visual styling.
"""

import asyncio
import base64
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger("osia.infographic")

_REPO_ROOT = Path(__file__).parent.parent.parent
_TEMPLATES_DIR = _REPO_ROOT / "templates" / "report"
_ASSETS_DIR = _REPO_ROOT / "assets"


def _load_logo_b64() -> str | None:
    logo_path = _ASSETS_DIR / "osia_logo_sm.png"
    if logo_path.exists():
        raw = logo_path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode()
    return None


def _desk_display_name(slug: str) -> str:
    return slug.replace("-", " ").title()


_EXTRACT_PROMPT = """\
You are an intelligence analyst preparing a visual brief.
Read the following report and extract:
1. A short, punchy headline (max 10 words) summarising the core finding.
2. Between 4 and 6 key findings as concise bullet points (each max 25 words).
3. A short image prompt (max 30 words) describing a dark, moody, cinematic \
background scene that evokes the mood and theme of the report. No text, no \
words, no letters in the image. Abstract and atmospheric.

Return ONLY valid JSON in this exact format, no preamble or markdown fences:
{{"headline": "...", "findings": ["...", "...", "..."], "image_prompt": "..."}}

REPORT:
{report_text}"""


async def extract_findings(gemini_client, model_id: str, report_text: str) -> dict | None:
    """Use Gemini to extract structured headline + findings from a report.

    Returns a dict with 'headline' (str) and 'findings' (list[str]),
    or None if extraction fails.
    """
    prompt = _EXTRACT_PROMPT.format(report_text=report_text[:6000])
    try:
        res = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=model_id,
            contents=prompt,
        )
        raw = (res.text or "").strip()
        # Strip markdown code fences if the model wraps the JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        headline = data.get("headline", "").strip()
        findings = [f.strip() for f in data.get("findings", []) if f.strip()]
        if not headline or len(findings) < 2:
            logger.warning("Infographic extraction returned insufficient data: %s", raw[:200])
            return None
        return {"headline": headline, "findings": findings[:6], "image_prompt": data.get("image_prompt", "")}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error("Failed to parse infographic findings JSON: %s", e)
    except Exception as e:
        logger.error("Infographic findings extraction failed: %s", e)
    return None


def render_infographic_png(
    headline: str,
    findings: list[str],
    desk_slug: str,
    bg_image_bytes: bytes | None = None,
) -> bytes:
    """Render an infographic card to PNG bytes.

    Uses the same WeasyPrint → PDF → PyMuPDF pipeline as SlideRenderer.

    Args:
        headline:       Short headline for the card.
        findings:       List of key finding strings.
        desk_slug:      Desk identifier for metadata display.
        bg_image_bytes: Optional raw PNG bytes for the background image.

    Returns:
        Raw PNG image bytes.
    """
    import fitz  # pymupdf
    from weasyprint import HTML

    # Suppress verbose logging from rendering libraries
    logging.getLogger("weasyprint").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.ERROR)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("infographic_card.html.j2")

    now = datetime.now(UTC)
    ref_prefix = desk_slug[:8].upper().replace("-", "")
    ref_number = f"OSIA-IB-{now.strftime('%Y%m%d')}-{ref_prefix}"

    bg_image_data_uri = None
    if bg_image_bytes:
        bg_image_data_uri = "data:image/png;base64," + base64.b64encode(bg_image_bytes).decode()

    html_content = template.render(
        headline=headline,
        findings=findings,
        desk_name=_desk_display_name(desk_slug),
        desk_slug=desk_slug,
        ref_number=ref_number,
        timestamp=now.strftime("%Y-%m-%d %H:%M UTC"),
        logo_data_uri=_load_logo_b64(),
        bg_image_data_uri=bg_image_data_uri,
    )

    pdf_bytes = HTML(string=html_content, base_url=str(_TEMPLATES_DIR)).write_pdf()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
    pix = doc[0].get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()

    logger.info("Rendered infographic card (%d bytes, ref %s)", len(png_bytes), ref_number)
    return png_bytes


async def generate_infographic(
    gemini_client,
    model_id: str,
    report_text: str,
    desk_slug: str,
) -> str | None:
    """End-to-end: extract findings from a report and render an infographic card.

    Optionally generates a Venice AI background image derived from the report
    to give each card a unique mood. Falls back gracefully to a plain card
    if Venice is unavailable or VENICE_API_KEY is not set.

    Returns a base64-encoded PNG string suitable for Signal attachment,
    or None if extraction or rendering fails.
    """
    data = await extract_findings(gemini_client, model_id, report_text)
    if not data:
        return None

    # Generate a thematic background image via Venice (best-effort)
    bg_image_bytes: bytes | None = None
    image_prompt = data.get("image_prompt", "").strip()
    if image_prompt:
        try:
            from src.intelligence.venice_image_client import VeniceImageClient

            venice = VeniceImageClient()
            bg_image_bytes = await venice.generate(
                prompt=image_prompt,
                width=1080,
                height=1920,
            )
            logger.info("Venice background generated for infographic (%d bytes)", len(bg_image_bytes))
        except Exception as e:
            logger.warning("Venice background generation failed (non-fatal): %s", e)

    try:
        png_bytes = await asyncio.get_event_loop().run_in_executor(
            None,
            render_infographic_png,
            data["headline"],
            data["findings"],
            desk_slug,
            bg_image_bytes,
        )
        return base64.b64encode(png_bytes).decode()
    except Exception as e:
        logger.error("Infographic rendering failed: %s", e)
        return None
