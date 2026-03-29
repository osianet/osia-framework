"""
migrate_research_cache.py

One-shot migration: copy all points from osia_research_cache into the
desk-specific Qdrant collection indicated by each point's `desk` payload field.

Points whose `desk` field is missing or blank stay in osia_research_cache.
After a successful migration the source points are deleted from osia_research_cache.

Usage:
    uv run python scripts/migrate_research_cache.py [--dry-run]
"""

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_research_cache")

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
SOURCE_COLLECTION = "osia_research_cache"
EMBEDDING_DIM = 384
BATCH_SIZE = 100


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{QDRANT_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {body_text}") from e


def ensure_collection(name: str) -> None:
    try:
        _request("GET", f"/collections/{name}")
        logger.info("Collection '%s' already exists.", name)
        return
    except RuntimeError:
        pass
    _request(
        "PUT",
        f"/collections/{name}",
        {
            "vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"},
            "optimizers_config": {"indexing_threshold": 1000},
        },
    )
    logger.info("Created collection '%s'.", name)


def scroll_all(collection: str) -> list[dict]:
    """Fetch every point (with vector) from the given collection."""
    points = []
    offset = None
    while True:
        body: dict = {"limit": 250, "with_payload": True, "with_vector": True}
        if offset:
            body["offset"] = offset
        data = _request("POST", f"/collections/{collection}/points/scroll", body)
        batch = data["result"]["points"]
        points.extend(batch)
        offset = data["result"].get("next_page_offset")
        logger.info("  scrolled %d so far...", len(points))
        if not offset:
            break
    return points


def upsert_batch(collection: str, points: list[dict]) -> None:
    _request("PUT", f"/collections/{collection}/points", {"points": points})


def delete_batch(collection: str, ids: list) -> None:
    _request(
        "POST",
        f"/collections/{collection}/points/delete",
        {"points": ids},
    )


def main(dry_run: bool) -> None:
    if dry_run:
        logger.info("DRY RUN — no writes will be made.")

    logger.info("Scrolling all points from '%s'...", SOURCE_COLLECTION)
    all_points = scroll_all(SOURCE_COLLECTION)
    logger.info("Fetched %d points total.", len(all_points))

    # Group by target desk collection
    by_desk: dict[str, list[dict]] = defaultdict(list)
    stay: list[dict] = []
    for pt in all_points:
        desk = pt["payload"].get("desk", "")
        if desk and desk != SOURCE_COLLECTION:
            by_desk[desk].append(pt)
        else:
            stay.append(pt)

    logger.info("Distribution:")
    for desk, pts in sorted(by_desk.items(), key=lambda x: -len(x[1])):
        logger.info("  %-50s  %d points", desk, len(pts))
    if stay:
        logger.info("  %-50s  %d points (will remain in source)", SOURCE_COLLECTION, len(stay))

    if dry_run:
        logger.info("Dry run complete — no changes made.")
        return

    migrated_ids: list = []

    for desk, pts in by_desk.items():
        logger.info("Migrating %d points → '%s'", len(pts), desk)
        ensure_collection(desk)

        # Upsert in batches
        for i in range(0, len(pts), BATCH_SIZE):
            batch = pts[i : i + BATCH_SIZE]
            upsert_batch(desk, batch)
            logger.info("  upserted %d/%d", min(i + BATCH_SIZE, len(pts)), len(pts))
            time.sleep(0.1)  # be gentle with the API

        migrated_ids.extend(pt["id"] for pt in pts)

    if not migrated_ids:
        logger.info("Nothing to migrate.")
        return

    logger.info("Deleting %d migrated points from '%s'...", len(migrated_ids), SOURCE_COLLECTION)
    for i in range(0, len(migrated_ids), BATCH_SIZE):
        batch = migrated_ids[i : i + BATCH_SIZE]
        delete_batch(SOURCE_COLLECTION, batch)
        logger.info("  deleted %d/%d", min(i + BATCH_SIZE, len(migrated_ids)), len(migrated_ids))
        time.sleep(0.1)

    logger.info("Migration complete. %d points moved.", len(migrated_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate osia_research_cache to desk collections.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without writing.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
