"""
Generic intel reclassification and research enqueue tool.

Searches a Qdrant collection for intel on a topic, patches the classification
payload on matching points, and/or enqueues corroboration research jobs.

Usage examples:

  # Reclassify misclassified intel and enqueue research across three desks
  uv run python scripts/reclassify_intel.py \\
    --topic "Israeli settlers Patagonia forest fires land acquisition" \\
    --collection information-warfare-desk \\
    --classification settler-colonialism-land-dispossession \\
    --desks geo humint env

  # Just enqueue research, skip Qdrant patching
  uv run python scripts/reclassify_intel.py \\
    --topic "Epstein network financial flows" \\
    --desks finance humint \\
    --no-reclassify

  # Just reclassify, no new research
  uv run python scripts/reclassify_intel.py \\
    --topic "some topic" \\
    --collection cyber-intelligence-and-warfare-desk \\
    --classification confirmed-nation-state-activity \\
    --no-research

  # Always dry-run first
  uv run python scripts/reclassify_intel.py --topic "..." --desks geo --dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from dotenv import load_dotenv

from src.intelligence.qdrant_store import QdrantStore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reclassify_intel")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

DESK_ALIASES: dict[str, str] = {
    "geo": "geopolitical-and-security-desk",
    "geopolitical": "geopolitical-and-security-desk",
    "humint": "human-intelligence-and-profiling-desk",
    "cultural": "cultural-and-theological-intelligence-desk",
    "cyber": "cyber-intelligence-and-warfare-desk",
    "finance": "finance-and-economics-directorate",
    "sci": "science-technology-and-commercial-desk",
    "science": "science-technology-and-commercial-desk",
    "infowar": "information-warfare-desk",
    "env": "environment-and-ecology-desk",
    "environment": "environment-and-ecology-desk",
    "watch": "the-watch-floor",
    "watchfloor": "the-watch-floor",
}


def resolve_desk(raw: str) -> str:
    return DESK_ALIASES.get(raw.lower(), raw)


async def search_and_reclassify(
    qdrant: QdrantStore,
    collection: str,
    topic: str,
    classification: str,
    note: str,
    threshold: float,
    dry_run: bool,
) -> int:
    """Search collection for topic, patch matching points. Returns number of points patched."""
    log.info("Searching '%s' for: %s", collection, topic)
    try:
        results = await qdrant.search(collection, topic, top_k=10)
    except Exception as e:
        log.error("Search failed: %s", e)
        return 0

    hits = [r for r in results if r.score >= threshold]
    if not hits:
        log.warning("No points found above score threshold %.2f — nothing to reclassify.", threshold)
        return 0

    log.info("Found %d point(s) to reclassify:", len(hits))
    for r in hits:
        preview = r.text[:200].replace("\n", " ")
        log.info("  [%.3f] %s...", r.score, preview)

    corrections = {
        "classification": classification,
        "disinfo_flag": False,
        "misinformation": False,
        "antisemitism_flag": False,
        "influence_operation": False,
        "reclassified_by": "manual-review",
        "reclassification_note": note,
    }

    patched = 0
    for r in hits:
        point_id = QdrantStore._point_id(r.text)
        if dry_run:
            log.info("[DRY-RUN] Would patch point %s → classification: %s", point_id, classification)
        else:
            try:
                await qdrant._client.set_payload(
                    collection_name=collection,
                    payload=corrections,
                    points=[point_id],
                )
                log.info("Patched point %s", point_id)
                patched += 1
            except Exception as e:
                log.error("Failed to patch point %s: %s", point_id, e)

    return patched


def enqueue_research(
    topic: str,
    desks: list[str],
    triggered_by: str,
    dry_run: bool,
) -> None:
    """Push one research job per desk onto osia:research_queue."""
    r = redis.from_url(REDIS_URL, decode_responses=True)
    for desk in desks:
        payload = {
            "job_id": str(uuid.uuid4()),
            "topic": topic,
            "desk": desk,
            "priority": "high",
            "directives_lens": True,
            "triggered_by": triggered_by,
        }
        if dry_run:
            log.info("[DRY-RUN] Would enqueue → %s", desk)
            log.info("           topic: %s...", topic[:120])
        else:
            r.rpush("osia:research_queue", json.dumps(payload))
            log.info("Enqueued research job → %s", desk)

    if not dry_run:
        depth = r.llen("osia:research_queue")
        log.info("osia:research_queue depth now: %d", depth)


async def main(args: argparse.Namespace) -> None:
    desks = [resolve_desk(d) for d in args.desks] if args.desks else []

    do_reclassify = not args.no_reclassify and args.collection
    do_research = not args.no_research and desks

    if not do_reclassify and not do_research:
        log.error("Nothing to do — supply --collection for reclassification and/or --desks for research.")
        raise SystemExit(1)

    if do_reclassify:
        qdrant = QdrantStore()
        note = args.note or (
            f"Reclassified via intel_admin: original classification overridden to '{args.classification}'. "
            f"Rerouted for corroboration."
        )
        await search_and_reclassify(
            qdrant=qdrant,
            collection=args.collection,
            topic=args.topic,
            classification=args.classification or "reclassified",
            note=note,
            threshold=args.threshold,
            dry_run=args.dry_run,
        )

    if do_research:
        research_topic = args.research_topic or args.topic
        enqueue_research(
            topic=research_topic,
            desks=desks,
            triggered_by=f"reclassify-intel-script:{args.topic[:60]}",
            dry_run=args.dry_run,
        )

    if args.dry_run:
        log.info("DRY-RUN complete — no changes written.")
    elif do_research:
        log.info("Trigger the research worker now with:")
        log.info("  sudo systemctl start osia-research-worker.service")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topic", required=True,
        help="Topic to search for (used as both search query and research topic).",
    )
    parser.add_argument(
        "--collection", default=None,
        help="Qdrant collection to search and reclassify (e.g. information-warfare-desk). "
             "Required unless --no-reclassify is set.",
    )
    parser.add_argument(
        "--classification", default=None,
        help="New classification value to write (e.g. settler-colonialism-land-dispossession).",
    )
    parser.add_argument(
        "--note", default=None,
        help="Optional human-readable reclassification note. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--desks", nargs="+", default=None, metavar="DESK",
        help="Desk slugs or aliases to enqueue research for "
             "(e.g. geo humint env). See DESK_ALIASES in script for short forms.",
    )
    parser.add_argument(
        "--research-topic", default=None,
        help="Override the research topic for enqueued jobs (defaults to --topic).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Minimum Qdrant similarity score to consider a point a match (default: 0.35).",
    )
    parser.add_argument(
        "--no-reclassify", action="store_true",
        help="Skip the Qdrant search and payload patch.",
    )
    parser.add_argument(
        "--no-research", action="store_true",
        help="Skip enqueuing research jobs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned actions without writing anything.",
    )
    asyncio.run(main(parser.parse_args()))
