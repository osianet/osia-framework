"""
Instagram account pool health monitor.

Runs every 4 hours (systemd timer). Tasks:
  1. Check ACTIVE pool depth — Signal alert if critically low
  2. Warm up each WARMING account in sequence, with a gap between sessions
  3. Promote accounts that have met the age + session-count thresholds
"""

import asyncio
import logging
import os

import httpx
import redis.asyncio as redis
from dotenv import load_dotenv

from src.agents.instagram_account_manager import InstagramAccountManager
from src.agents.instagram_warmup_agent import InstagramWarmupSession

logger = logging.getLogger("osia.ig_health")

# Minimum gap between back-to-back account warm-up sessions (seconds)
_INTER_SESSION_SECS = int(os.getenv("IG_INTER_SESSION_DELAY_SECS", "900"))  # 15 minutes


async def _signal_alert(message: str) -> None:
    """Best-effort Signal message to the group."""
    api_url = os.getenv("SIGNAL_API_URL", "http://localhost:8081")
    sender = os.getenv("SIGNAL_SENDER_NUMBER")
    group_id = os.getenv("SIGNAL_GROUP_ID")
    recipient = group_id if group_id else sender
    if not recipient or not sender:
        logger.warning("Signal not configured — skipping alert: %s", message)
        return
    payload = {
        "message": message,
        "number": sender,
        "recipients": [recipient] if not group_id else [],
        "groupId": group_id if group_id else "",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            await http.post(f"{api_url}/v2/send", json=payload)
    except Exception as exc:
        logger.warning("Signal alert failed: %s", exc)


async def run_health_check() -> None:
    load_dotenv()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = redis.from_url(redis_url, decode_responses=True)
    mgr = InstagramAccountManager(r)

    try:
        counts = await mgr.pool_counts()
        logger.info(
            "Pool status — ACTIVE=%d WARMING=%d FLAGGED=%d",
            counts["active"],
            counts["warming"],
            counts["flagged"],
        )

        active_min = int(os.getenv("IG_POOL_ACTIVE_MIN", "3"))

        # Alert if pool is critically low
        if counts["active"] == 0:
            await _signal_alert(
                "[IG POOL ALERT] ACTIVE pool is empty — yt-dlp Instagram ingestion is down. "
                "Run ig_pool_admin.py --create-account to add accounts."
            )
        elif counts["active"] < active_min:
            await _signal_alert(
                f"[IG POOL WARNING] Only {counts['active']} ACTIVE Instagram accounts "
                f"(min={active_min}). Consider creating more."
            )

        # Promote accounts that have met both age and session-count thresholds
        eligible = await mgr.eligible_for_promotion()
        for account_id in eligible:
            account = await mgr.get(account_id)
            if account:
                await mgr.promote(account_id)
                logger.info("Promoted @%s → ACTIVE", account.username)

        # Warm up WARMING accounts, sorted by oldest last_warmed_at first
        warming_ids = list(await r.smembers("osia:ig:pool:warming"))
        if not warming_ids:
            logger.info("No WARMING accounts — nothing to warm up")
            return

        warming_accounts = []
        for aid in warming_ids:
            acc = await mgr.get(aid)
            if acc:
                warming_accounts.append(acc)

        # Sort: accounts never warmed go first (last_warmed_at=None), then oldest
        warming_accounts.sort(key=lambda a: a.last_warmed_at or 0)

        session = InstagramWarmupSession(mgr, r, headed=False, upload_avatar=True)

        for i, acc in enumerate(warming_accounts):
            if i > 0:
                logger.info("Waiting %ds before next session…", _INTER_SESSION_SECS)
                await asyncio.sleep(_INTER_SESSION_SECS)

            logger.info(
                "Warming @%s (%d/%d) — sessions=%d",
                acc.username,
                i + 1,
                len(warming_accounts),
                acc.warmup_sessions,
            )
            success = await session.run(acc.id)
            if not success:
                logger.warning("Warm-up failed for @%s", acc.username)

        # Re-check promotion after warmup sessions
        eligible = await mgr.eligible_for_promotion()
        for account_id in eligible:
            account = await mgr.get(account_id)
            if account:
                await mgr.promote(account_id)
                logger.info("Post-warmup promotion: @%s → ACTIVE", account.username)

    finally:
        await r.aclose()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(run_health_check())
