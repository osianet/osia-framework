"""
Weekly Department Briefing trigger.

Generates slide deck briefings with narrated video for every configured
intelligence desk. Each desk's department head presents their week's
most significant intelligence in both landscape (16:9) and portrait (9:16)
video formats.

Output is written to reports/weekly/<YYYY-Wnn>/<desk-slug>/.

Triggered weekly on Mondays at 08:00 UTC by systemd timer.
"""

import argparse
import asyncio
import logging

from src.intelligence.briefing_generator import generate_all_briefings

logger = logging.getLogger("osia.weekly_briefing")


async def trigger_weekly_briefing(desks: list[str] | None = None, resume: bool = False) -> None:
    logger.info("Weekly Department Briefing pipeline starting...")

    results = await generate_all_briefings(desks=desks, resume=resume)

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
    parser = argparse.ArgumentParser(description="OSIA Weekly Department Briefing")
    parser.add_argument(
        "--desks",
        nargs="+",
        metavar="DESK_SLUG",
        help="Only generate briefings for these desk slugs (e.g. --desks cyber-intelligence-and-warfare-desk geopolitical-and-security-desk)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip desks whose videos already exist; skip individual slides whose audio already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(trigger_weekly_briefing(desks=args.desks, resume=args.resume))
