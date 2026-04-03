"""
Re-encode existing weekly briefing videos with slide fades and audio pause.

Scans all week directories under reports/weekly/, finds desks that have
both slide PNGs and audio MP3s, and re-assembles their videos using the
current _assemble_video() logic (fade in/out, silence pad between slides).

Usage:
    uv run python scripts/reencode_weekly_videos.py
    uv run python scripts/reencode_weekly_videos.py --week 2026-W14
    uv run python scripts/reencode_weekly_videos.py --week 2026-W14 --desk cyber-intelligence-and-warfare-desk
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.intelligence.briefing_generator import _assemble_video, _load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("reencode_weekly_videos")


def _collect_jobs(weeks_base: Path, week_filter: str | None, desk_filter: str | None) -> list[dict]:
    """Walk reports/weekly/ and collect re-encode jobs."""
    jobs: list[dict] = []

    if not weeks_base.exists():
        logger.error("Weekly reports directory not found: %s", weeks_base)
        return jobs

    week_dirs = sorted(weeks_base.iterdir()) if not week_filter else [weeks_base / week_filter]

    for week_dir in week_dirs:
        if not week_dir.is_dir() or not week_dir.name.startswith("20"):
            continue

        desk_dirs = [week_dir / desk_filter] if desk_filter else sorted(week_dir.iterdir())

        for desk_dir in desk_dirs:
            if not desk_dir.is_dir():
                continue

            desk_slug = desk_dir.name
            audio_dir = desk_dir / "audio"

            if not audio_dir.exists():
                continue

            audio_files = sorted(audio_dir.glob("slide_*.mp3"))
            if not audio_files:
                continue

            for orientation in ("landscape", "portrait"):
                orient_dir = desk_dir / orientation
                slides_dir = orient_dir / "slides"

                if not slides_dir.exists():
                    continue

                image_files = sorted(slides_dir.glob("slide_*.png"))
                if not image_files:
                    continue

                if len(image_files) != len(audio_files):
                    logger.warning(
                        "%s/%s/%s: slide count mismatch (%d images, %d audio) — skipping",
                        week_dir.name,
                        desk_slug,
                        orientation,
                        len(image_files),
                        len(audio_files),
                    )
                    continue

                # Determine output video path (same as original)
                week_label = week_dir.name
                video_path = orient_dir / f"{desk_slug}_{week_label}_{orientation}.mp4"

                jobs.append(
                    {
                        "week": week_label,
                        "desk_slug": desk_slug,
                        "orientation": orientation,
                        "image_paths": image_files,
                        "audio_paths": audio_files,
                        "output_path": video_path,
                    }
                )

    return jobs


async def main() -> None:
    parser = argparse.ArgumentParser(description="Re-encode weekly briefing videos with slide fades.")
    parser.add_argument("--week", help="Only re-encode a specific week (e.g. 2026-W14)")
    parser.add_argument("--desk", help="Only re-encode a specific desk slug")
    args = parser.parse_args()

    config = _load_config()
    weeks_base = _REPO_ROOT / config.get("output", {}).get("base_dir", "reports/weekly")

    jobs = _collect_jobs(weeks_base, args.week, args.desk)

    if not jobs:
        logger.info("No re-encode jobs found.")
        return

    logger.info("Found %d video(s) to re-encode.", len(jobs))

    success = 0
    failed = 0

    for job in jobs:
        label = f"{job['week']}/{job['desk_slug']}/{job['orientation']}"
        logger.info("Re-encoding: %s", label)
        try:
            await _assemble_video(
                image_paths=job["image_paths"],
                audio_paths=job["audio_paths"],
                output_path=job["output_path"],
                config=config,
            )
            success += 1
        except Exception as e:
            logger.error("Failed: %s — %s", label, e)
            failed += 1

    logger.info("Done. %d succeeded, %d failed.", success, failed)


if __name__ == "__main__":
    asyncio.run(main())
