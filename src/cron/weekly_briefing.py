"""
Weekly Department Briefing trigger.

Generates slide deck briefings with narrated video for every configured
intelligence desk. Each desk's department head presents their week's
most significant intelligence in both landscape (16:9) and portrait (9:16)
video formats.

Output is written to reports/weekly/<YYYY-Wnn>/<desk-slug>/.

Triggered weekly on Mondays at 08:00 UTC by systemd timer.
"""

import asyncio
import logging

from src.intelligence.briefing_generator import generate_all_briefings

logger = logging.getLogger("osia.weekly_briefing")


async def trigger_weekly_briefing():
    logger.info("Weekly Department Briefing pipeline starting...")

    results = await generate_all_briefings()

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if successful:
        logger.info(
            "Completed briefings: %s",
            ", ".join(r.get("desk_slug", "?") for r in successful),
        )
    if failed:
        logger.warning(
            "Failed briefings: %s",
            ", ".join(f"{r.get('desk_slug', '?')}: {r.get('error', '?')}" for r in failed),
        )

    logger.info("Weekly briefing pipeline finished.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(trigger_weekly_briefing())
