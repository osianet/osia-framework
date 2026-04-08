"""
boost_source_quality.py — Manually up-rank intel points by source author.

Finds all points across Qdrant collections whose payload contains a matching
source/author string, then:
  - Sets ingested_at_unix to now  (removes temporal decay penalty)
  - Sets reliability_tier to "A"  (highest quality marker)

Usage:
  uv run python scripts/boost_source_quality.py --author triploi [--dry-run] [--collections col1 col2]
"""

import argparse
import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("boost_source_quality")

ALL_COLLECTIONS = [
    "collection-directorate",
    "geopolitical-and-security-desk",
    "cultural-and-theological-intelligence-desk",
    "science-technology-and-commercial-desk",
    "human-intelligence-and-profiling-desk",
    "finance-and-economics-directorate",
    "cyber-intelligence-and-warfare-desk",
    "information-warfare-desk",
    "environment-and-ecology-desk",
    "the-watch-floor",
    "osia_research_cache",
    "wikileaks-cables",
    "epstein-files",
]

PAYLOAD_AUTHOR_FIELDS = ["source", "author", "username", "handle", "entity_tags"]


def _matches_author(payload: dict, author_lower: str) -> bool:
    """Return True if any author-related field in the payload contains the author string."""
    for field in PAYLOAD_AUTHOR_FIELDS:
        val = payload.get(field)
        if val is None:
            continue
        if isinstance(val, str) and author_lower in val.lower():
            return True
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and author_lower in item.lower():
                    return True
    # Also do a loose scan across all string fields
    for val in payload.values():
        if isinstance(val, str) and author_lower in val.lower():
            return True
    return False


async def boost_author(author: str, dry_run: bool, collections: list[str]) -> None:
    client = AsyncQdrantClient(
        url=os.getenv("QDRANT_URL", "https://qdrant.osia.dev"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        port=None,
    )
    author_lower = author.lower()
    now_ts = int(time.time())
    total_updated = 0

    for col in collections:
        try:
            exists = await client.collection_exists(col)
        except Exception as exc:
            logger.warning("Cannot check collection '%s': %s", col, exc)
            continue
        if not exists:
            logger.debug("Collection '%s' does not exist — skipping.", col)
            continue

        logger.info("Scanning '%s' ...", col)
        offset = None
        col_hits: list[str] = []

        while True:
            result = await client.scroll(
                collection_name=col,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result

            for point in points:
                payload = dict(point.payload or {})
                if _matches_author(payload, author_lower):
                    col_hits.append(point.id)

            if next_offset is None:
                break
            offset = next_offset

        if not col_hits:
            logger.info("  No matches in '%s'.", col)
            continue

        logger.info("  Found %d point(s) attributed to '%s' in '%s'.", len(col_hits), author, col)

        if dry_run:
            logger.info("  [DRY RUN] Would update %d point(s) — skipping writes.", len(col_hits))
        else:
            # Batch payload updates in chunks of 100
            chunk_size = 100
            for i in range(0, len(col_hits), chunk_size):
                chunk = col_hits[i : i + chunk_size]
                await client.set_payload(
                    collection_name=col,
                    payload={
                        "ingested_at_unix": now_ts,
                        "reliability_tier": "A",
                    },
                    points=chunk,
                )
            logger.info("  Updated %d point(s): ingested_at_unix→now, reliability_tier→A", len(col_hits))

        total_updated += len(col_hits)

    label = "[DRY RUN] Would have updated" if dry_run else "Updated"
    logger.info("Done. %s %d total point(s) across %d collection(s).", label, total_updated, len(collections))


def main() -> None:
    parser = argparse.ArgumentParser(description="Up-rank intel points by source author in Qdrant.")
    parser.add_argument("--author", required=True, help="Author/handle to boost (case-insensitive substring match)")
    parser.add_argument("--dry-run", action="store_true", help="Scan only — no writes")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=ALL_COLLECTIONS,
        metavar="COL",
        help="Collections to scan (default: all desk collections)",
    )
    args = parser.parse_args()

    asyncio.run(boost_author(args.author, args.dry_run, args.collections))


if __name__ == "__main__":
    main()
