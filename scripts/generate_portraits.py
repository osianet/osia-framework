"""
Generate department head portraits for OSIA weekly briefings.

Reads portrait_prompt from each desk's briefing config and generates
a headshot via Venice AI. Portraits are saved to assets/portraits/<desk-slug>.png
and reused across all future briefings.

Usage:
    uv run python scripts/generate_portraits.py              # all desks
    uv run python scripts/generate_portraits.py --desk cyber-intelligence-and-warfare-desk
    uv run python scripts/generate_portraits.py --force      # regenerate even if file exists
"""

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.desks.desk_registry import DeskRegistry
from src.intelligence.venice_image_client import VeniceImageClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("osia.generate_portraits")

PORTRAITS_DIR = Path(__file__).parent.parent / "assets" / "portraits"

# Portrait dimensions — square, high-res for crisp rendering at any slide size
PORTRAIT_SIZE = 768


async def generate_portraits(desk_filter: str | None = None, force: bool = False) -> None:
    registry = DeskRegistry()
    slugs = registry.list_slugs()

    if desk_filter:
        if desk_filter not in slugs:
            logger.error("Unknown desk: %s", desk_filter)
            return
        slugs = [desk_filter]

    client = VeniceImageClient(
        model="flux-2-pro",
        style_preset="",
        steps=25,
        cfg_scale=8.0,
    )

    PORTRAITS_DIR.mkdir(parents=True, exist_ok=True)
    generated = 0

    for slug in slugs:
        desk = registry.get(slug)
        if not desk.briefing or not desk.briefing.portrait_prompt:
            logger.info("Skipping %s — no portrait_prompt configured", slug)
            continue

        output_path = PORTRAITS_DIR / f"{slug}.png"
        if output_path.exists() and not force:
            logger.info("Skipping %s — portrait already exists (use --force to regenerate)", slug)
            continue

        logger.info("Generating portrait for %s...", desk.name)
        try:
            await client.generate(
                prompt=desk.briefing.portrait_prompt,
                width=PORTRAIT_SIZE,
                height=PORTRAIT_SIZE,
                output_path=output_path,
            )
            logger.info("✓ Saved: %s", output_path)
            generated += 1
        except Exception as e:
            logger.error("✗ Failed for %s: %s", slug, e)

    await registry.close()
    logger.info("Done — %d portrait(s) generated", generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate department head portraits via Venice AI")
    parser.add_argument("--desk", type=str, default=None, help="Generate for a single desk slug only")
    parser.add_argument("--force", action="store_true", help="Regenerate even if portrait already exists")
    args = parser.parse_args()

    asyncio.run(generate_portraits(desk_filter=args.desk, force=args.force))


if __name__ == "__main__":
    main()
