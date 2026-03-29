"""
OSIA Iran-Israel War Monitor

Periodically queues targeted research topics to the research worker
to ensure the iran-israel-war-2026 Qdrant collection is continuously
updated with the latest intelligence.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime

import redis.asyncio as aioredis
from dotenv import load_dotenv

logger = logging.getLogger("osia.iran_israel_monitor")

RESEARCH_QUEUE_KEY = "osia:research_queue"
SEEN_TOPICS_KEY = "osia:research:seen_topics"
DESK = "iran-israel-war-2026"


async def trigger_monitor():
    load_dotenv()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    topics = [
        f"Latest political developments in the Iran-Israel conflict as of {today}",
        f"Recent military strikes or operations between Iran and Israel leading up to {today}",
        f"International community responses and sanctions regarding the Iran-Israel war on {today}",
    ]

    r = aioredis.from_url(redis_url, decode_responses=True)

    enqueued = 0
    for topic in topics:
        normalised = topic.lower().strip()
        already_seen = await r.sismember(SEEN_TOPICS_KEY, normalised)
        if already_seen:
            logger.debug("Skipping already-seen topic: %r", topic)
            continue

        payload = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": DESK,
                "priority": "normal",
                "directives_lens": True,
                "triggered_by": "iran_israel_monitor",
            }
        )

        await r.rpush(RESEARCH_QUEUE_KEY, payload)
        await r.sadd(SEEN_TOPICS_KEY, normalised)

        logger.info(
            "Enqueued research job for %r → %s (triggered_by: iran_israel_monitor)",
            topic,
            DESK,
        )
        enqueued += 1

    logger.info("Enqueued %d new research jobs for the Iran-Israel monitor.", enqueued)
    await r.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(trigger_monitor())
