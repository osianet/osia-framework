"""
Daily SITREP (Situational Report) generator.

Pulls all RSS intelligence accumulated since the last SITREP from Redis,
synthesizes it with live research via the orchestrator, and delivers
the final briefing via Signal.

Triggered daily at 07:00 UTC by systemd timer.
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime

import redis.asyncio as redis
from dotenv import load_dotenv

logger = logging.getLogger("osia.sitrep")

DAILY_DIGEST_KEY = "osia:rss:daily_digest"


async def _drain_digest(r: redis.Redis) -> list[str]:
    """Pop all items from the daily digest list atomically."""
    pipe = r.pipeline()
    pipe.lrange(DAILY_DIGEST_KEY, 0, -1)
    pipe.delete(DAILY_DIGEST_KEY)
    results = await pipe.execute()
    items = results[0]  # lrange result
    return [item.decode() if isinstance(item, bytes) else item for item in items]


async def trigger_sitrep():
    load_dotenv()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    queue_name = os.getenv("OSIA_TASK_QUEUE", "osia:task_queue")

    group_id = os.getenv("SIGNAL_GROUP_ID")
    sender = os.getenv("SIGNAL_SENDER_NUMBER")
    recipient = group_id if group_id else sender

    if not recipient:
        logger.error("No SIGNAL_GROUP_ID or SIGNAL_SENDER_NUMBER set. Cannot deliver SITREP.")
        return

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    logger.info("Triggering Daily SITREP for %s targeting %s", today, recipient)

    r = redis.from_url(redis_url)

    # Pull all RSS intelligence collected since last SITREP
    digest_items = await _drain_digest(r)
    logger.info("Collected %d RSS intelligence items for today's briefing.", len(digest_items))

    # Build the digest section for the SITREP prompt
    if digest_items:
        # Cap at ~30 items to stay within token limits
        capped = digest_items[:30]
        digest_block = "\n\n---\n\n".join(capped)
        digest_section = (
            f"The following {len(capped)} intelligence reports were collected from RSS feeds "
            f"in the last 24 hours. Use these as primary source material for the SITREP:\n\n"
            f"{digest_block}"
        )
    else:
        digest_section = (
            "No RSS intelligence was collected in the last 24 hours. "
            "Rely on live research tools (Tavily web search, Wikipedia, ArXiv) to gather current events."
        )

    query = (
        f"Generate a Daily SITREP (Situational Report) for {today}.\n\n"
        f"## Pre-Collected Intelligence\n\n"
        f"{digest_section}\n\n"
        f"## Instructions\n\n"
        f"Using the pre-collected intelligence above AND your research tools, produce a formal "
        f"Intelligence Summary (INTSUM) covering:\n"
        f"1. Global Geopolitics and Security — conflicts, diplomacy, sanctions, military movements\n"
        f"2. Global Financial Markets — market moves, economic policy, trade developments\n"
        f"3. Emerging Technology and AI — breakthroughs, regulatory changes, notable papers\n\n"
        f"Structure the report with clear section headers, cite specific sources where possible, "
        f"and highlight any items requiring immediate attention with a ⚠️ prefix.\n"
        f"End with a 'WATCH LIST' section of developing situations to monitor.\n\n"
        f"## Citation Requirements\n\n"
        f"Tag every factual claim with a bracketed citation [N]. At the end of the report, "
        f"include a '## Sources' section listing each source with:\n"
        f"- Citation number\n"
        f"- Origin (RSS feed URL, tool name, or research source)\n"
        f"- Reliability rating: A (peer-reviewed/official), B (established media), "
        f"C (web/blog), D (social media), E (unverifiable)\n"
        f"Format: [N] (Rating) Origin — Description\n"
        f"End with a '## Source Confidence' line: HIGH (mostly A/B), MODERATE (mixed), or LOW (mostly C/D/E).\n"
        f"Mark any unsourced claims as [UNSOURCED] inline."
    )

    task = {
        "source": f"signal:{recipient}",
        "query": query,
    }

    await r.rpush(queue_name, json.dumps(task))
    await r.aclose()
    logger.info("SITREP task pushed to queue successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(trigger_sitrep())
