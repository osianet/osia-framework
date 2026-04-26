"""
rechunk_qdrant_intsums.py

One-shot cleanup: find INTSUM blob entries in desk primary collections (single
large points written by the orchestrator before chunked upserts were introduced),
delete them, and re-insert them as properly chunked points.

Blob detection heuristic: points in desk collections that
  - have no `chunk_index` payload field (not already chunked), AND
  - have `len(text) > MIN_CHARS` (default 800)

KB collections (epstein-files, wikileaks-cables, etc.) are NOT touched.

Usage:
    uv run python scripts/rechunk_qdrant_intsums.py [--dry-run] [--min-chars 800]
    uv run python scripts/rechunk_qdrant_intsums.py --execute
"""

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rechunk_intsums")

# Desk primary collections that hold INTSUM write-backs.
# collection-directorate holds RSS summaries/YouTube transcripts (different structure);
# osia_research_cache already uses chunked upserts from the research worker.
# Both are excluded from the default run — use --include-directorate to add them.
DESK_COLLECTIONS = [
    "geopolitical-and-security-desk",
    "cultural-and-theological-intelligence-desk",
    "science-technology-and-commercial-desk",
    "human-intelligence-and-profiling-desk",
    "finance-and-economics-directorate",
    "cyber-intelligence-and-warfare-desk",
    "information-warfare-desk",
    "environment-and-ecology-desk",
    "the-watch-floor",
]

EXTRA_COLLECTIONS = [
    "collection-directorate",
    "osia_research_cache",
]

DEFAULT_MIN_CHARS = 800
SCROLL_BATCH = 250


async def rechunk_collection(
    store,
    collection: str,
    min_chars: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Return (blobs_found, chunks_written)."""
    from qdrant_client.http import models as qm

    blobs_found = 0
    chunks_written = 0
    offset = None

    while True:
        result = await store._client.scroll(
            collection_name=collection,
            scroll_filter=None,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result

        blob_points = []
        for pt in points:
            payload = dict(pt.payload or {})
            text = payload.get("text", "")
            # Skip already-chunked points and short entries
            if "chunk_index" in payload:
                continue
            if len(text) < min_chars:
                continue
            blob_points.append((str(pt.id), text, payload))

        if blob_points:
            blobs_found += len(blob_points)
            logger.info(
                "%s: found %d blob(s) in this batch (total so far: %d)",
                collection,
                len(blob_points),
                blobs_found,
            )

            if not dry_run:
                for point_id, text, payload in blob_points:
                    # Build clean metadata (strip text — it's passed separately)
                    meta = {k: v for k, v in payload.items() if k not in ("text", "ingested_at_unix")}

                    # Delete the original blob
                    await store._client.delete(
                        collection_name=collection,
                        points_selector=qm.PointIdsList(points=[point_id]),
                    )

                    # Re-insert as chunks
                    ids = await store.upsert_chunks(collection, text, meta)
                    chunks_written += len(ids)
                    logger.info("%s: replaced blob %s with %d chunks", collection, point_id, len(ids))
            else:
                for point_id, text, _payload in blob_points:
                    preview = text[:120].replace("\n", " ")
                    chunks_preview = len(store._split_into_chunks(text))
                    logger.info(
                        "  [DRY RUN] %s | id=%s | %d chars → %d chunks | %.120s…",
                        collection,
                        point_id,
                        len(text),
                        chunks_preview,
                        preview,
                    )

        if next_offset is None:
            break
        offset = next_offset

    return blobs_found, chunks_written


async def main(min_chars: int, dry_run: bool, include_directorate: bool) -> None:
    # Import here so the script can be run standalone
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.intelligence.qdrant_store import QdrantStore

    store = QdrantStore()

    collections = list(DESK_COLLECTIONS)
    if include_directorate:
        collections += EXTRA_COLLECTIONS

    mode = "DRY RUN — no changes will be made" if dry_run else "EXECUTE — blobs will be deleted and rechunked"
    logger.info(
        "rechunk_qdrant_intsums starting | mode=%s | min_chars=%d | collections=%d",
        mode,
        min_chars,
        len(collections),
    )

    total_blobs = 0
    total_chunks = 0

    for collection in collections:
        try:
            exists = await store._client.collection_exists(collection)
            if not exists:
                logger.info("%s: collection not found, skipping", collection)
                continue
            blobs, chunks = await rechunk_collection(store, collection, min_chars, dry_run)
            total_blobs += blobs
            total_chunks += chunks
            if blobs:
                logger.info("%s: %d blob(s) → %d chunks", collection, blobs, chunks)
            else:
                logger.info("%s: clean (no blobs found)", collection)
        except Exception as exc:
            logger.error("%s: error — %s", collection, exc)

    logger.info(
        "Done. Total blobs found: %d | chunks written: %d%s",
        total_blobs,
        total_chunks,
        " (dry run — nothing written)" if dry_run else "",
    )
    await store.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rechunk INTSUM blob entries in desk Qdrant collections.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete blobs and write chunks. Default is dry-run.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Minimum text length to treat as a blob (default: {DEFAULT_MIN_CHARS})",
    )
    parser.add_argument(
        "--include-directorate",
        action="store_true",
        help="Also process collection-directorate and osia_research_cache (RSS/YouTube entries).",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            min_chars=args.min_chars,
            dry_run=not args.execute,
            include_directorate=args.include_directorate,
        )
    )
