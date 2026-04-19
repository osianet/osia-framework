"""Import existing PDF INTSUM archive into the Wiki.js intelligence wiki.

Reads PDFs from the reports/ directory, extracts text via PyMuPDF, and creates
wiki pages under each desk's /intsums/ section. Idempotent — skips pages that
already exist. Thumbnails and subdirectories are silently skipped.

Usage:
  uv run python scripts/wiki_import_pdfs.py [--dry-run] [--limit N] [--desk SLUG]

Environment variables:
  WIKIJS_URL      — default http://localhost:3000/graphql
  WIKIJS_API_KEY  — required for writes
"""

import argparse
import asyncio
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wiki_import_pdfs")

_REPO_ROOT = Path(__file__).parent.parent
_REPORTS_DIR = _REPO_ROOT / "reports"

sys.path.insert(0, str(_REPO_ROOT))

from src.intelligence.wiki_client import (  # noqa: E402
    WikiClient,
    build_intsum_page,
    desk_wiki_section,
)

# ---------------------------------------------------------------------------
# PDF filename parser
# ---------------------------------------------------------------------------

# Pattern: 2026-03-29_04-57-41_the-watch-floor.pdf
_FNAME_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<desk>[a-z0-9-]+)\.pdf$")


def _parse_filename(path: Path) -> tuple[str, str, str] | None:
    """Return (date, time_str, desk_slug) or None if not a valid INTSUM PDF."""
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    return m.group("date"), m.group("time"), m.group("desk")


def _extract_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception as e:
        logger.warning("PyMuPDF failed for %s: %s — trying pypdf", pdf_path.name, e)

    try:
        import pypdf

        reader = pypdf.PdfReader(str(pdf_path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e2:
        logger.error("Text extraction failed for %s: %s", pdf_path.name, e2)
        return ""


# ---------------------------------------------------------------------------
# Desk name lookup
# ---------------------------------------------------------------------------

_DESK_NAMES: dict[str, str] = {}


def _desk_display_name(slug: str) -> str:
    if slug in _DESK_NAMES:
        return _DESK_NAMES[slug]
    try:
        import yaml

        cfg_path = _REPO_ROOT / "config" / "desks" / f"{slug}.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            name = cfg.get("name", slug.replace("-", " ").title())
            _DESK_NAMES[slug] = name
            return name
    except Exception as exc:
        logger.debug("Could not load desk name for %s: %s", slug, exc)
    return slug.replace("-", " ").title()


# ---------------------------------------------------------------------------
# Main importer
# ---------------------------------------------------------------------------


async def import_pdfs(dry_run: bool, limit: int | None, desk_filter: str | None) -> None:
    import os

    if not os.getenv("WIKIJS_API_KEY") and not dry_run:
        logger.error("WIKIJS_API_KEY not set — cannot write to wiki (use --dry-run to preview)")
        return

    pdfs = sorted(p for p in _REPORTS_DIR.rglob("*.pdf") if p.is_file())
    logger.info("Found %d PDF files in %s", len(pdfs), _REPORTS_DIR)

    processed = 0
    skipped = 0
    errors = 0

    async with WikiClient() as wiki:
        for pdf_path in pdfs:
            parsed = _parse_filename(pdf_path)
            if not parsed:
                logger.debug("Skipping non-INTSUM file: %s", pdf_path.name)
                skipped += 1
                continue

            date_str, time_str, desk_slug = parsed

            if desk_filter and desk_slug != desk_filter:
                skipped += 1
                continue

            # Build wiki path from filename timestamp so it matches what live
            # orchestrator would have produced for that desk + date.
            dt = datetime.strptime(f"{date_str} {time_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            wiki_path = f"{desk_wiki_section(desk_slug)}/intsums/{date_str}-{time_str}-imported"

            # Skip if already exists
            if not dry_run:
                existing = await wiki.get_page(wiki_path)
                if existing:
                    logger.debug("Already exists, skipping: %s", wiki_path)
                    skipped += 1
                    continue

            text = _extract_text(pdf_path)
            if not text:
                logger.warning("Empty text from %s — skipping", pdf_path.name)
                errors += 1
                continue

            desk_name = _desk_display_name(desk_slug)
            desk_section = desk_wiki_section(desk_slug)
            ref_suffix = desk_slug[:8].upper().replace("-", "")
            ref_number = f"OSIA-{dt.strftime('%Y%m%d')}-{ref_suffix}-{dt.strftime('%H%M')}"
            title = f"{ref_number} — {desk_name} [PDF import]"

            content = build_intsum_page(
                analysis=f"*[Imported from PDF archive: `{pdf_path.name}`]*\n\n---\n\n{text}",
                desk_name=desk_name,
                desk_section=desk_section,
                ref_number=ref_number,
                timestamp=dt.strftime("%Y-%m-%dT%H:%M:%S UTC"),
                source="pdf-archive",
                entity_links=[],
            )

            if dry_run:
                logger.info("[DRY-RUN] Would create: %s — '%s'", wiki_path, title)
                processed += 1
            else:
                ok = await wiki.create_page(
                    wiki_path,
                    title,
                    content,
                    description=f"{desk_name} INTSUM — {date_str} [PDF archive import]",
                    tags=["intsum", "pdf-import", f"desk-{desk_slug[:20]}"],
                )
                if ok:
                    logger.info("Created: %s", wiki_path)
                    processed += 1
                else:
                    logger.warning("Failed to create: %s", wiki_path)
                    errors += 1

                await asyncio.sleep(0.5)

            if limit and processed >= limit:
                logger.info("Reached limit of %d pages — stopping", limit)
                break

    logger.info("Done. processed=%d skipped=%d errors=%d", processed, skipped, errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import OSIA PDF archive into Wiki.js")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Max pages to create")
    parser.add_argument("--desk", metavar="SLUG", default=None, help="Only import one desk's PDFs")
    args = parser.parse_args()
    asyncio.run(import_pdfs(args.dry_run, args.limit, args.desk))


if __name__ == "__main__":
    main()
