#!/usr/bin/env python3
"""
Backfill Instagram source accounts from historical Qdrant intel.

Scans all desk collections for points whose triggered_by URL is an Instagram
reel, resolves each unique URL to an uploader handle via yt-dlp, adds the
handle to the Redis intel-sources set, and creates a wiki dossier page.

Usage:
    uv run python scripts/ig_backfill_sources.py [--dry-run] [--limit N]

Environment variables:
    REDIS_URL, QDRANT_URL, QDRANT_API_KEY, WIKIJS_API_KEY
"""

import argparse
import asyncio
import logging
import os
import subprocess
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

_YTDLP_BIN = Path(".venv/bin/yt-dlp")


def _resolve_uploader(url: str) -> tuple[str | None, str | None, str | None]:
    """Return (uploader_id, uploader_display, channel_url) via yt-dlp, or (None, None, None)."""
    bin_path = str(_YTDLP_BIN) if _YTDLP_BIN.exists() else "yt-dlp"
    try:
        proc = subprocess.run(
            [
                bin_path,
                "--skip-download",
                "--print",
                "%(uploader_id)s\t%(uploader)s\t%(channel_url)s",
                "--no-warnings",
                "--quiet",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split("\t")
            uid = parts[0].strip() if len(parts) > 0 else None
            uname = parts[1].strip() if len(parts) > 1 else None
            curl = parts[2].strip() if len(parts) > 2 else None
            # Skip if yt-dlp returned a literal "NA" or empty
            return (
                uid if uid and uid != "NA" else None,
                uname if uname and uname != "NA" else None,
                curl if curl and curl != "NA" else None,
            )
    except Exception as exc:
        logger.debug("yt-dlp failed for %s: %s", url, exc)
    return None, None, None


async def backfill(dry_run: bool, limit: int | None) -> None:
    r = redis_async.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    qs = QdrantStore()

    # Collect unique IG reel URLs across all collections
    seen_urls: set[str] = set()
    url_to_first_point: dict[str, dict] = {}

    logger.info("Scanning %d collections for Instagram triggered_by URLs...", len(_COLLECTIONS))
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
                    tb = str(p.get("triggered_by") or "")
                    if "instagram.com/reel/" in tb or "instagram.com/p/" in tb:
                        # Normalise: strip query params
                        clean = tb.split("?")[0].rstrip("/")
                        if clean not in seen_urls:
                            seen_urls.add(clean)
                            url_to_first_point[clean] = p
                if not next_offset:
                    break
                offset = next_offset
        except Exception as exc:
            logger.warning("Error scanning %s: %s", collection, exc)

    logger.info("Found %d unique Instagram reel URLs", len(seen_urls))

    handled = 0
    skipped = 0

    async with WikiClient() as wiki:
        for url, point_payload in sorted(url_to_first_point.items()):
            if limit and handled >= limit:
                logger.info("Reached limit of %d — stopping", limit)
                break

            uploader_id, display_name, channel_url = _resolve_uploader(url)
            if not uploader_id:
                logger.debug("Could not resolve uploader for %s — skipping", url)
                skipped += 1
                await asyncio.sleep(1)
                continue

            logger.info("Resolved %s → @%s (%s)", url[:60], uploader_id, display_name or "")

            if dry_run:
                logger.info("[DRY-RUN] Would register @%s and create wiki dossier", uploader_id)
                handled += 1
                continue

            # Stamp Redis
            await r.sadd(_INTEL_SOURCES_KEY, uploader_id)

            # Build date/context from the Qdrant point
            ts = point_payload.get("timestamp") or ""
            try:
                date_str = (
                    datetime.fromisoformat(ts.replace("UTC", "+00:00")).strftime("%Y-%m-%d")
                    if ts
                    else datetime.now(UTC).strftime("%Y-%m-%d")
                )
            except ValueError:
                date_str = datetime.now(UTC).strftime("%Y-%m-%d")

            wiki_path_ref = point_payload.get("wiki_path", "")
            intsum_title = f"Intel from {date_str}"

            sa_path = social_account_wiki_path("instagram", uploader_id)
            existing = await wiki.get_page(sa_path)
            if existing:
                logger.debug("Dossier already exists for @%s — skipping create", uploader_id)
            else:
                content = build_social_account_page(
                    handle=uploader_id,
                    platform="instagram",
                    display_name=display_name or uploader_id,
                    channel_url=channel_url or f"https://www.instagram.com/{uploader_id}",
                    first_seen=date_str,
                    intsum_path=wiki_path_ref,
                    intsum_title=intsum_title,
                )
                ok = await wiki.create_page(
                    sa_path,
                    f"@{uploader_id}",
                    content,
                    description=f"Instagram — @{uploader_id} — first intel {date_str}",
                    tags=["social-account", "instagram", "intel-source"],
                )
                if ok:
                    logger.info("Created dossier: %s", sa_path)
                else:
                    logger.warning("Failed to create dossier for @%s", uploader_id)

            handled += 1
            await asyncio.sleep(1.5)

    await r.aclose()
    total_in_redis = await _count_redis(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    logger.info(
        "Done. resolved=%d skipped=%d — osia:ig:intel_sources now has %d handles", handled, skipped, total_in_redis
    )


async def _count_redis(url: str) -> int:
    r = redis_async.from_url(url, decode_responses=True)
    count = await r.scard(_INTEL_SOURCES_KEY)
    await r.aclose()
    return int(count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Instagram intel source accounts")
    parser.add_argument("--dry-run", action="store_true", help="Resolve handles but do not write")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Max accounts to process")
    args = parser.parse_args()
    asyncio.run(backfill(args.dry_run, args.limit))


if __name__ == "__main__":
    main()
