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

import yaml

from src.intelligence.chatterbox_tts_client import ChatterboxTTSClient
from src.intelligence.slide_renderer import SlideRenderer

logger = logging.getLogger("osia.intro_video_generator")

_REPO_ROOT = Path(__file__).parent.parent.parent
_OUT_DIR = _REPO_ROOT / "reports" / "intro_webcast"


def _load_voice_ref_map() -> dict[str, Path]:
    """Build a slug → absolute voice_ref_path mapping from all desk YAMLs."""
    voice_map: dict[str, Path] = {}
    config_dir = _REPO_ROOT / "config" / "desks"
    for yaml_file in config_dir.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                cfg = yaml.safe_load(f)
            slug = cfg.get("slug", "")
            ref = cfg.get("briefing", {}).get("voice_ref_path", "")
            if slug and ref:
                abs_ref = (_REPO_ROOT / ref).resolve()
                if abs_ref.exists():
                    voice_map[slug] = abs_ref
                else:
                    logger.warning("Voice ref not found for %s: %s", slug, abs_ref)
        except Exception as e:
            logger.warning("Could not load desk config %s: %s", yaml_file.name, e)
    return voice_map


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

    # Render slides to PNG
    logger.info("Rendering slides...")
    renderer = SlideRenderer()
    try:
        png_paths = renderer.render_deck(
            slides=slides,
            desk_slug="hero",          # fallback bg for non-desk slides
            desk_name="OSIA",
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
    voice_ref_map = _load_voice_ref_map()
    audio_paths: list[Path] = []

    for i, slide in enumerate(slides):
        narration = slide.get("narration", "").strip()
        if not narration:
            logger.warning("Slide %d has no narration", i)
            audio_paths.append(None)
            continue

        audio_path = work_dir / "audio" / f"narration_{i:02d}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and audio_path.exists():
            logger.debug("Resume: reusing audio %s", audio_path.name)
            audio_paths.append(audio_path)
            continue

        # Priority: explicit voice_ref_path on slide > desk voice map > None (default).
        desk_slug = slide.get("desk_slug")
        voice_ref = (
            slide.get("voice_ref_path")
            or (voice_ref_map.get(desk_slug) if desk_slug else None)
        )
        if desk_slug and not voice_ref:
            logger.warning("No voice ref found for desk %s — using default voice", desk_slug)

        try:
            await tts.generate_speech(narration, voice_ref, output_path=audio_path)
            audio_paths.append(audio_path)
            logger.info("Generated audio %d/%d (voice=%s)", i + 1, len(slides), desk_slug or "default")
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


def _get_audio_duration(audio_path: Path) -> float:
    """Return duration of an audio file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return float(result.stdout.strip())


_PAUSE_SECS = 0.8    # silence gap appended after each slide's audio
_FADE_SECS  = 0.4    # cross-fade duration between clips


async def _assemble_video(
    png_paths: list[Path],
    audio_paths: list[Path | None],
    output_path: Path,
    orientation: str,
) -> None:
    """Assemble PNG slides + audio into MP4 with pauses and cross-fades.

    Per-slide strategy:
      1. Append _PAUSE_SECS of silence to each audio clip so there's a
         natural breath between speakers.
      2. Encode each slide as a self-contained MP4 clip (PNG looped for
         audio+pause duration).
      3. Apply an xfade+acrossfade transition between every pair of clips
         and re-encode to a single output file.
    """
    clips_dir = output_path.parent / "clips"
    clips_dir.mkdir(exist_ok=True)
    clip_paths: list[Path] = []
    durations: list[float] = []

    for i, png_path in enumerate(png_paths):
        if not png_path or not png_path.exists():
            logger.warning("PNG %d missing, skipping slide", i)
            continue

        audio_path = audio_paths[i] if i < len(audio_paths) else None
        padded_audio = clips_dir / f"audio_{i:02d}_padded.wav"
        clip_path = clips_dir / f"clip_{i:02d}.mp4"

        # Get speech duration, then build a padded audio file with silence appended.
        speech_dur = 5.0
        if audio_path and audio_path.exists():
            try:
                speech_dur = _get_audio_duration(audio_path)
            except Exception as e:
                logger.warning("Could not probe audio duration for slide %d: %s", i, e)

            # Pad with silence: adelay puts the speech first, apad extends to total duration.
            total_dur = speech_dur + _PAUSE_SECS
            pad_cmd = [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-af", f"apad=pad_dur={_PAUSE_SECS}",
                "-t", str(total_dur),
                str(padded_audio),
            ]
            result = subprocess.run(pad_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning("Audio padding failed for slide %d, using original", i)
                padded_audio = audio_path
                total_dur = speech_dur
        else:
            total_dur = speech_dur + _PAUSE_SECS
            # Generate silent audio for slides without narration.
            pad_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-t", str(total_dur),
                str(padded_audio),
            ]
            subprocess.run(pad_cmd, capture_output=True, text=True, timeout=10)

        durations.append(total_dur)

        use_audio = padded_audio if padded_audio.exists() else None
        if use_audio:
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(png_path),
                "-i", str(use_audio),
                "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                "-c:a", "aac", "-b:a", "128k",
                "-t", str(total_dur),
                "-pix_fmt", "yuv420p",
                str(clip_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(png_path),
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                "-c:a", "aac", "-b:a", "128k",
                "-t", str(total_dur),
                "-pix_fmt", "yuv420p",
                str(clip_path),
            ]

        logger.info("Encoding clip %d/%d (%.1fs total)...", i + 1, len(png_paths), total_dur)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg clip {i} failed: {result.stderr[-500:]}")
        clip_paths.append(clip_path)

    if not clip_paths:
        raise RuntimeError("No clips were produced — cannot assemble video")

    if len(clip_paths) == 1:
        # Nothing to cross-fade — just copy.
        clip_paths[0].rename(output_path)
        return

    # Build a single ffmpeg command that cross-fades all clips together.
    # xfade offsets must account for each clip's duration minus the overlap.
    logger.info("Applying cross-fades and assembling final video (%d clips)...", len(clip_paths))

    inputs: list[str] = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    n = len(clip_paths)
    fade = _FADE_SECS

    # Build video filter chain: chain xfade between consecutive streams.
    # After each xfade the output label becomes the input for the next.
    vf_parts: list[str] = []
    af_parts: list[str] = []

    # Cumulative offset: sum of all preceding clip durations minus accumulated fade overlaps.
    offset = durations[0] - fade
    prev_v = "[0:v]"
    prev_a = "[0:a]"

    for idx in range(1, n):
        out_v = f"[vx{idx}]" if idx < n - 1 else "[vout]"
        out_a = f"[ax{idx}]" if idx < n - 1 else "[aout]"
        vf_parts.append(f"{prev_v}[{idx}:v]xfade=transition=fade:duration={fade}:offset={offset:.3f}{out_v}")
        af_parts.append(f"{prev_a}[{idx}:a]acrossfade=d={fade}{out_a}")
        offset += durations[idx] - fade
        prev_v = out_v
        prev_a = out_a

    filter_complex = ";".join(vf_parts + af_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg xfade assembly failed: {result.stderr[-800:]}")
