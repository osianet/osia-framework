#!/usr/bin/env python3
"""
Backfill Instagram source accounts from historical Qdrant intel.

Scans all desk collections for points triggered by Instagram reels and
extracts account handles from the report text using @-mention and inline
instagram.com/profile URL patterns. No Instagram requests are made.

Usage:
    uv run python scripts/ig_backfill_sources.py [--dry-run] [--limit N]

Environment variables:
    REDIS_URL, QDRANT_URL, QDRANT_API_KEY, WIKIJS_API_KEY
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ig_backfill_sources")

import redis.asyncio as redis_async  # noqa: E402

from src.intelligence.qdrant_store import QdrantStore  # noqa: E402
from src.intelligence.wiki_client import (  # noqa: E402
    WikiClient,
    build_social_account_page,
    social_account_wiki_path,
)

_INTEL_SOURCES_KEY = "osia:ig:intel_sources"

_COLLECTIONS = [
    "geopolitical-and-security-desk",
    "the-watch-floor",
    "information-warfare-desk",
    "human-intelligence-and-profiling-desk",
    "cultural-and-theological-intelligence-desk",
    "science-technology-and-commercial-desk",
    "finance-and-economics-directorate",
    "cyber-intelligence-and-warfare-desk",
    "environment-and-ecology-desk",
    "collection-directorate",
]

# @handle — explicit mention in report text
_AT_RE = re.compile(r"@([A-Za-z0-9._]{4,30})")
# instagram.com/<username> — profile URL mentioned inline (not a reel/post path)
_IG_PROFILE_RE = re.compile(r"instagram\.com/(?!reel/|p/|tv/|stories/|explore/)([A-Za-z0-9._]{4,30})(?:[/?'\"\s)$]|$)")

# Noise to exclude: common English words captured by the patterns, email domains,
# and single-word fragments that are clearly not handles.
_EXCLUDE = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "the",
    "and",
    "this",
    "that",
    "with",
    "from",
    "have",
    "been",
    "they",
    "their",
    "about",
    "which",
    "also",
    "into",
    "more",
    "some",
    "when",
    "what",
    "will",
    "were",
    "your",
    "each",
    "over",
    "such",
    "only",
    "than",
    "then",
    "those",
    "these",
    "through",
    "while",
    "after",
    "there",
    "other",
    "both",
    "well",
    "even",
    "most",
    "said",
    "here",
    "just",
}


def _extract_handles(text: str) -> set[str]:
    handles: set[str] = set()
    for m in _AT_RE.finditer(text):
        h = m.group(1).lower().strip("._")
        if h and h not in _EXCLUDE and not h.endswith((".com", ".org", ".net", ".edu")):
            handles.add(h)
    for m in _IG_PROFILE_RE.finditer(text):
        h = m.group(1).lower().strip("._")
        if h and h not in _EXCLUDE and not h.endswith((".com", ".org", ".net", ".edu")):
            handles.add(h)
    return handles


async def backfill(dry_run: bool, limit: int | None) -> None:
    r = redis_async.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    qs = QdrantStore()

    # handle → best payload (the point with the most text, as a proxy for richest intel)
    handle_to_payload: dict[str, dict] = {}
    handle_freq: dict[str, int] = {}

    logger.info("Scanning %d collections for Instagram-triggered intel points...", len(_COLLECTIONS))
    for collection in _COLLECTIONS:
        offset = None
        try:
            while True:
                results = await qs._client.scroll(
                    collection_name=collection,
                    limit=200,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                pts, next_offset = results
                for pt in pts:
                    p = pt.payload or {}
                    if "instagram.com" not in str(p.get("triggered_by") or ""):
                        continue
                    text = p.get("text") or ""
                    for handle in _extract_handles(text):
                        handle_freq[handle] = handle_freq.get(handle, 0) + 1
                        # Keep payload from the most text-rich point as dossier context
                        if handle not in handle_to_payload or len(text) > len(
                            handle_to_payload[handle].get("text") or ""
                        ):
                            handle_to_payload[handle] = p
                if not next_offset:
                    break
                offset = next_offset
        except Exception as exc:
            logger.warning("Error scanning %s: %s", collection, exc)

    # Sort by frequency — highest-confidence handles first
    ranked = sorted(handle_to_payload.keys(), key=lambda h: -handle_freq[h])
    logger.info("Extracted %d candidate handles (top 20: %s)", len(ranked), ranked[:20])

    handled = 0
    async with WikiClient() as wiki:
        for handle in ranked:
            if limit and handled >= limit:
                logger.info("Reached limit of %d — stopping", limit)
                break

            freq = handle_freq[handle]
            point_payload = handle_to_payload[handle]

            if dry_run:
                logger.info("[DRY-RUN] @%-30s  freq=%d", handle, freq)
                handled += 1
                continue

            await r.sadd(_INTEL_SOURCES_KEY, handle)

            ts = point_payload.get("timestamp") or ""
            try:
                date_str = (
                    datetime.fromisoformat(ts.replace("UTC", "+00:00")).strftime("%Y-%m-%d")
                    if ts
                    else datetime.now(UTC).strftime("%Y-%m-%d")
                )
            except ValueError:
                date_str = datetime.now(UTC).strftime("%Y-%m-%d")

            wiki_path_ref = point_payload.get("wiki_path") or ""
            intsum_title = f"Intel from {date_str}"

            sa_path = social_account_wiki_path("instagram", handle)
            existing = await wiki.get_page(sa_path)
            if existing:
                logger.debug("Dossier already exists for @%s — skipping", handle)
            else:
                content = build_social_account_page(
                    handle=handle,
                    platform="instagram",
                    display_name=handle,
                    channel_url=f"https://www.instagram.com/{handle}",
                    first_seen=date_str,
                    intsum_path=wiki_path_ref,
                    intsum_title=intsum_title,
                )
                ok = await wiki.create_page(
                    sa_path,
                    f"@{handle}",
                    content,
                    description=f"Instagram — @{handle} — first intel {date_str}",
                    tags=["social-account", "instagram", "intel-source"],
                )
                if ok:
                    logger.info("Created dossier: @%-30s  (freq=%d)", handle, freq)
                else:
                    logger.warning("Failed to create dossier for @%s", handle)

            handled += 1
            await asyncio.sleep(0.3)

    await r.aclose()

    r2 = redis_async.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    total = int(await r2.scard(_INTEL_SOURCES_KEY))
    await r2.aclose()
    logger.info("Done. created=%d — osia:ig:intel_sources now has %d handles", handled, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Instagram intel source accounts from report text")
    parser.add_argument("--dry-run", action="store_true", help="Extract and print handles without writing")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Max handles to process")
    args = parser.parse_args()
    asyncio.run(backfill(args.dry_run, args.limit))


if __name__ == "__main__":
    main()
