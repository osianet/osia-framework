"""
OSIA intro webcast video generator.

Follows the briefing_generator pattern:
1. Generate slides with narration
2. Render slides to PNG via SlideRenderer
3. Generate audio via Chatterbox TTS
4. Assemble video via ffmpeg

Each desk intro slide includes the department head's portrait.
"""

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.intelligence.aesthetic import load_portrait_b64
from src.intelligence.chatterbox_tts_client import ChatterboxTTSClient
from src.intelligence.slide_renderer import SlideRenderer

logger = logging.getLogger("osia.intro_video_generator")

_REPO_ROOT = Path(__file__).parent.parent.parent
_OUT_DIR = _REPO_ROOT / "reports" / "intro_webcast"


async def generate_intro_video(
    slides: list[dict],
    orientation: str = "landscape",
    resume: bool = False,
) -> Path | None:
    """Generate full intro webcast video.

    Args:
        slides: List of slide dicts from generate_intro_video.py
        orientation: 'landscape' (16:9) or 'portrait' (9:16)
        resume: Skip existing files

    Returns:
        Path to final video file, or None if generation failed.
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = _OUT_DIR / f"intro_{orientation}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Inject portrait URIs for desk intro slides
    for slide in slides:
        if slide.get("desk_slug"):
            portrait_uri = load_portrait_b64(slide["desk_slug"])
            if portrait_uri:
                slide["portrait_data_uri"] = portrait_uri

    # Render slides to PNG
    logger.info("Rendering slides...")
    renderer = SlideRenderer()
    try:
        png_paths = renderer.render_deck(
            slides=slides,
            desk_slug="intro-webcast",
            desk_name="OSIA Intro Webcast",
            persona_name="OSIA",
            orientation=orientation,
            output_dir=work_dir / "slides",
            week_label=datetime.now(UTC).strftime("%Y-W%V"),
            bg_images=None,
        )
    except Exception as e:
        logger.error("Slide rendering failed: %s", e)
        return None

    # Generate narration audio via Chatterbox
    logger.info("Generating narration audio...")
    tts = ChatterboxTTSClient()
    audio_paths: list[Path] = []

    for i, slide in enumerate(slides):
        narration = slide.get("narration", "").strip()
        if not narration:
            logger.warning("Slide %d has no narration", i)
            audio_paths.append(None)
            continue

        audio_path = work_dir / "audio" / f"narration_{i:02d}.mp3"
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and audio_path.exists():
            logger.debug("Resume: reusing audio %s", audio_path.name)
            audio_paths.append(audio_path)
            continue

        try:
            await tts.generate_speech(narration, "default", output_path=audio_path)
            audio_paths.append(audio_path)
            logger.info("Generated audio %d/%d", i + 1, len(slides))
        except Exception as e:
            logger.error("TTS failed for slide %d: %s", i, e)
            audio_paths.append(None)

    # Assemble video
    logger.info("Assembling video...")
    video_path = _OUT_DIR / f"OSIA_Intro_Webcast_{orientation}.mp4"

    try:
        await _assemble_video(png_paths, audio_paths, video_path, orientation)
        logger.info("✓ Video saved: %s", video_path)
        return video_path
    except Exception as e:
        logger.error("Video assembly failed: %s", e)
        return None


async def _assemble_video(
    png_paths: list[Path],
    audio_paths: list[Path | None],
    output_path: Path,
    orientation: str,
) -> None:
    """Assemble PNG slides + audio into MP4 via ffmpeg."""
    # Build ffmpeg concat demuxer file
    concat_file = output_path.parent / "concat.txt"
    with open(concat_file, "w") as f:
        for i, png_path in enumerate(png_paths):
            if not png_path or not png_path.exists():
                logger.warning("PNG %d missing: %s", i, png_path)
                continue

            # Determine duration from audio if available
            duration = 5  # Default
            if i < len(audio_paths) and audio_paths[i] and audio_paths[i].exists():
                try:
                    result = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=noprint_wrappers=1:nokey=1:nokey=1",
                            str(audio_paths[i]),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    duration = float(result.stdout.strip())
                except Exception as e:
                    logger.warning("Could not get audio duration: %s", e)

            f.write(f"file '{png_path}'\n")
            f.write(f"duration {duration}\n")

    # Build audio concat file
    audio_concat_file = output_path.parent / "audio_concat.txt"
    with open(audio_concat_file, "w") as f:
        for audio_path in audio_paths:
            if audio_path and audio_path.exists():
                f.write(f"file '{audio_path}'\n")

    # ffmpeg: concat images + audio
    cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(audio_concat_file),
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(output_path),
    ]

    logger.debug("ffmpeg command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
