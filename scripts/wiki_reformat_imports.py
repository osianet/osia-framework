"""AI-based reformatter for PDF-imported wiki pages.

Iterates all pages tagged `pdf-import` (but not yet `wiki-reformatted`),
sends the raw extracted content to Claude Sonnet via OpenRouter, and patches
the `content` AUTO-fenced section back with a clean OSIA INTSUM structure.
Idempotent — pages already tagged `wiki-reformatted` are skipped.

Usage:
  uv run python scripts/wiki_reformat_imports.py [--dry-run] [--limit N]
                                                  [--desk SLUG] [--model MODEL]

Environment variables:
  WIKIJS_URL           — default http://localhost:3000/graphql
  WIKIJS_API_KEY       — required for writes
  OPENROUTER_API_KEY   — required for AI reformatting
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wiki_reformat_imports")

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.intelligence.wiki_client import WikiClient  # noqa: E402

_DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Wiki GraphQL — list all pages with tags
# ---------------------------------------------------------------------------

_LIST_TAGGED_GQL = """
query ListPages {
  pages {
    list(orderBy: PATH) {
      id
      path
      title
      description
      updatedAt
      tags
    }
  }
}
"""


async def _list_all_pages(wiki: WikiClient) -> list[dict]:
    """Return all pages. Each dict has id, path, title, description, tags (list[str])."""
    data = await wiki._gql(_LIST_TAGGED_GQL)
    pages = (data.get("data") or {}).get("pages", {}).get("list", [])
    result = []
    for p in pages:
        raw_tags = p.get("tags") or []
        # Wiki.js list returns tags as plain strings (not objects)
        if raw_tags and isinstance(raw_tags[0], dict):
            tags = [t["tag"] for t in raw_tags]
        else:
            tags = [str(t) for t in raw_tags]
        result.append({**p, "tags": tags})
    return result


# ---------------------------------------------------------------------------
# AI reformatter
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an OSIA intelligence analyst tasked with reformatting raw PDF-extracted text \
into clean, structured INTSUM documents. You have deep knowledge of intelligence \
reporting conventions and OSIA's house style.

Your output will replace the content section of a wiki page. It must be pure markdown — \
no preamble, no explanation, no wrapping code fences.

## OSIA INTSUM structure (follow exactly)

```
> **CLASSIFICATION: OSIA INTERNAL — [DESK CODE]**

# INTSUM — [Topic] — YYYY-MM-DD

## Executive Summary
One to three sentences. Core finding, key actors, significance.

## Assessment
Analytical paragraphs. What happened, what it means, confidence level.
Bold key entities on first mention: **Entity Name**.
Express uncertainty: "assessed with moderate confidence", "likely", "possibly".

## Key Entities
- **Entity Name** — role/description, relevance to this report

## Source Material
- Source type and brief description (e.g. *PDF archive report*, *SIGINT summary*)

## Implications & Tasking
- Strategic implication or recommended follow-up action
```

## Cleaning rules
1. Remove page headers/footers (repeated titles, page numbers like "Page 2 of 7", \
   "OSIA INTSUM", date lines appearing mid-text as artefacts).
2. Rejoin hyphenated line-breaks (e.g. "secu-\nrity" → "security").
3. Remove excessive blank lines (max two consecutive blank lines become one).
4. Preserve ALL intelligence content faithfully — do not omit facts, figures, names, \
   dates, locations, or assessments from the original text.
5. Preserve direct quotations verbatim within quote blocks.
6. If the original text lacks enough content to fill a section, write \
   "*Insufficient source material.*" — never fabricate intelligence.
7. Infer the DESK CODE from the desk name visible in the page metadata line \
   (e.g. "Geopolitical & Security" → GSD, "Cyber Intelligence & Warfare" → CWD, \
   "Watch Floor" → WF, "Human Intelligence & Profiling" → HUMINT, \
   "Cultural & Theological" → CTD, "Finance & Economics" → FED, \
   "Science, Technology & Commercial" → STCD, \
   "Information & Psychological Warfare" → IPWD, \
   "Environment & Ecological Intelligence" → EED).
8. Infer the date from the ref number or timestamp in the metadata table.
9. Infer the topic from the document content — use a concise 3–7 word phrase.
"""


async def _reformat_page(
    content_section: str,
    page_title: str,
    model: str,
    http,
) -> str | None:
    """Call OpenRouter and return reformatted markdown, or None on failure."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return None

    user_msg = (
        f"Page title: {page_title}\n\n"
        "Below is the raw content extracted from a PDF intelligence report. "
        "Reformat it into a clean OSIA INTSUM following the structure and rules "
        "in your instructions. Output only the reformatted markdown.\n\n"
        "---\n\n"
        f"{content_section.strip()}"
    )

    payload = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": user_msg}],
        "system": _SYSTEM_PROMPT,
    }

    try:
        resp = await http.post(
            f"{_OPENROUTER_BASE}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://osia.dev",
                "X-Title": "OSIA Wiki Reformatter",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("OpenRouter call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Content section extractor
# ---------------------------------------------------------------------------

_AUTO_RE = re.compile(
    r"<!-- OSIA:AUTO:content -->(.*?)<!-- /OSIA:AUTO:content -->",
    re.DOTALL,
)


def _extract_content_section(full_content: str) -> str | None:
    m = _AUTO_RE.search(full_content)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Tag updater — adds wiki-reformatted to tags list
# ---------------------------------------------------------------------------


async def _add_reformatted_tag(wiki: WikiClient, page: dict) -> bool:
    """Add `wiki-reformatted` tag to the page without touching content."""
    tags = list(page.get("tags") or [])
    if "wiki-reformatted" in tags:
        return True
    tags.append("wiki-reformatted")
    return await wiki.update_page(
        page["id"],
        page["content"],
        page["title"],
        page.get("description", ""),
        tags,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def reformat(
    dry_run: bool,
    limit: int | None,
    desk_filter: str | None,
    model: str,
) -> None:
    import httpx

    if not os.getenv("WIKIJS_API_KEY") and not dry_run:
        logger.error("WIKIJS_API_KEY not set — use --dry-run to preview")
        return

    if not os.getenv("OPENROUTER_API_KEY") and not dry_run:
        logger.error("OPENROUTER_API_KEY not set")
        return

    async with WikiClient() as wiki, httpx.AsyncClient(timeout=30.0) as http:
        # Pre-fetch format guide once
        logger.info("Fetching INTSUM format guide…")
        format_guide_page = await wiki.get_page("kb/methodology/intsum-format")
        if format_guide_page:
            logger.info("Format guide loaded (%d chars)", len(format_guide_page["content"]))
        else:
            logger.warning("Could not load kb/methodology/intsum-format — continuing without it")

        # List all pages and filter
        logger.info("Listing all wiki pages…")
        all_pages = await _list_all_pages(wiki)
        candidates = [p for p in all_pages if "pdf-import" in p["tags"] and "wiki-reformatted" not in p["tags"]]

        if desk_filter:
            desk_tag = f"desk-{desk_filter[:20]}"
            candidates = [p for p in candidates if desk_tag in p["tags"]]

        logger.info(
            "Found %d pdf-import pages (%d total, %d already reformatted)",
            len(candidates),
            len([p for p in all_pages if "pdf-import" in p["tags"]]),
            len([p for p in all_pages if "wiki-reformatted" in p["tags"]]),
        )

        if not candidates:
            logger.info("Nothing to do.")
            return

        processed = skipped = errors = 0

        for page_meta in candidates:
            if limit and processed >= limit:
                logger.info("Reached limit of %d — stopping", limit)
                break

            path = page_meta["path"]

            # Fetch full page content
            page = await wiki.get_page(path)
            if not page:
                logger.warning("Could not fetch page: %s", path)
                errors += 1
                continue

            content_section = _extract_content_section(page["content"])
            if not content_section:
                logger.warning("No content AUTO section in %s — skipping", path)
                skipped += 1
                continue

            # Skip pages that are just stub placeholders with no real text
            # (less than 200 chars of actual content after stripping import note)
            real_content = re.sub(r"\*\[Imported from PDF archive[^\]]*\]\*", "", content_section).strip()
            if len(real_content) < 200:
                logger.info("Skipping stub page (too little content): %s", path)
                skipped += 1
                continue

            if dry_run:
                logger.info(
                    "[DRY-RUN] Would reformat: %s ('%s', %d chars of content)",
                    path,
                    page["title"],
                    len(content_section),
                )
                processed += 1
                continue

            logger.info("Reformatting: %s", path)
            reformatted = await _reformat_page(content_section, page["title"], model, http)
            if not reformatted:
                logger.warning("AI reformatting failed for %s", path)
                errors += 1
                await asyncio.sleep(2)
                continue

            # Patch the content section
            ok = await wiki.patch_section(path, "content", reformatted)
            if not ok:
                logger.warning("patch_section failed for %s", path)
                errors += 1
                await asyncio.sleep(2)
                continue

            # Re-fetch to get updated content for tag update
            updated_page = await wiki.get_page(path)
            if updated_page:
                await _add_reformatted_tag(wiki, updated_page)

            logger.info("Done: %s", path)
            processed += 1
            await asyncio.sleep(1.0)  # gentle rate limiting

    logger.info("Finished. processed=%d skipped=%d errors=%d", processed, skipped, errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI reformatter for PDF-imported OSIA wiki pages")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Max pages to reformat")
    parser.add_argument("--desk", metavar="SLUG", default=None, help="Only reformat pages from this desk")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {_DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    asyncio.run(reformat(args.dry_run, args.limit, args.desk, args.model))


if __name__ == "__main__":
    main()
