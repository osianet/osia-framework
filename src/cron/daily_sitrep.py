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

# Standing queries for Qdrant pre-seed — one per SITREP section.
# These pull accumulated OSIA intelligence into the briefing regardless of the
# day's RSS haul, giving the Watch Floor six months of context to draw on.
_SITREP_SEED_QUERIES = [
    ("geopolitics", "geopolitical conflicts military threats diplomacy sanctions state actors"),
    ("cyber", "cyber attacks threat actors nation state intrusions malware campaigns"),
    ("finance", "financial market risks economic policy trade sanctions fiscal instability"),
    ("technology", "emerging technology AI developments scientific breakthroughs dual-use"),
    (
        "gender_violence",
        "male violence against women sexual violence occupation detention femicide manosphere misogyny accountability",
    ),
    (
        "infowar",
        "information warfare influence operations propaganda manosphere radicalization disinformation psyops narrative",
    ),
]
_SEED_TOP_K = 3  # results per standing query
_SEED_MIN_SCORE = 0.45  # discard low-confidence matches


async def _pull_qdrant_seed() -> str | None:
    """
    Run standing intelligence queries against all OSIA Qdrant collections and
    return a formatted 'ACCUMULATED OSIA INTELLIGENCE' block, or None on failure.
    Uses 70-day half-life temporal decay so stale entries rank lower.
    """
    try:
        from src.intelligence.qdrant_store import QdrantStore

        async with QdrantStore() as store:
            sections: list[str] = []

            async def _query(label: str, q: str) -> tuple[str, list]:
                results = await store.cross_desk_search(q, top_k=_SEED_TOP_K, decay_half_life_days=70.0)
                filtered = [r for r in results if r.score >= _SEED_MIN_SCORE]
                return label, filtered

            pairs = await asyncio.gather(*[_query(lbl, q) for lbl, q in _SITREP_SEED_QUERIES])

        for label, hits in pairs:
            if not hits:
                continue
            block_lines = [f"### {label.upper()} — accumulated intelligence"]
            for r in hits:
                src = r.metadata.get("source", r.collection)
                ts = r.metadata.get("collected_at", r.metadata.get("timestamp", ""))
                date_str = ts[:10] if ts else ""
                block_lines.append(
                    f"[{r.collection} | {src}{' | ' + date_str if date_str else ''} | score={r.score:.2f}]\n{r.text}"
                )
            sections.append("\n\n".join(block_lines))

        if not sections:
            return None

        header = (
            "## ACCUMULATED OSIA INTELLIGENCE\n\n"
            "The following entries were retrieved from OSIA's long-term knowledge base "
            "(research archives, past INTSUMs, RSS summaries, and KB collections). "
            "Use these to provide historical context and continuity in the SITREP.\n\n"
        )
        return header + "\n\n---\n\n".join(sections)

    except Exception as exc:
        logger.warning("Qdrant SITREP pre-seed failed (non-fatal): %s", exc)
        return None


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

    # Pull RSS digest and Qdrant accumulated intelligence concurrently
    digest_items, qdrant_seed = await asyncio.gather(
        _drain_digest(r),
        _pull_qdrant_seed(),
    )
    logger.info(
        "SITREP seed: %d RSS items, Qdrant pre-seed %s",
        len(digest_items),
        "loaded" if qdrant_seed else "unavailable",
    )

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

    # Inject Qdrant accumulated intelligence if available
    qdrant_section = f"\n\n{qdrant_seed}" if qdrant_seed else ""

    query = (
        f"Generate a Daily SITREP (Situational Report) for {today}.\n\n"
        f"## Pre-Collected Intelligence\n\n"
        f"{digest_section}"
        f"{qdrant_section}\n\n"
        f"## Instructions\n\n"
        f"Using the pre-collected intelligence above AND your research tools, produce a formal "
        f"Intelligence Summary (INTSUM) covering:\n"
        f"1. Global Geopolitics and Security — conflicts, diplomacy, sanctions, military movements\n"
        f"2. Global Financial Markets — market moves, economic policy, trade developments\n"
        f"3. Emerging Technology and AI — breakthroughs, regulatory changes, notable papers\n"
        f"4. Gender Violence & Feminist Intelligence — male violence against women across scales "
        f"(conflict-zone/occupation sexual violence, elite trafficking networks, femicide trends), "
        f"feminist organising and resistance, accountability journalism. Surface trajectory indicators. "
        f"If no new intelligence exists today, note the standing monitoring status.\n"
        f"5. Information & Psychological Warfare — manosphere radicalization pipeline activity, "
        f"coordinated influence operations, narrative warfare campaigns, platform suppression of "
        f"aligned content. If no new intelligence exists today, note the standing monitoring status.\n\n"
        f"Structure the report with clear section headers, cite specific sources where possible, "
        f"and highlight any items requiring immediate attention with a ⚠️ prefix.\n"
        f"End with a 'WATCH LIST' section of developing situations to monitor, which must include "
        f"a sub-section for each OSIA Standing Monitoring Priority (gender violence, manosphere "
        f"pipeline, feminist organising) regardless of whether new intelligence arrived today.\n\n"
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
