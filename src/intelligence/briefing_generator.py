"""
Weekly Department Briefing Generator.

Orchestrates the full pipeline for each intelligence desk:
  1. Query Qdrant for the past week's intel
  2. Generate a structured briefing via the desk's LLM
  3. Render slide decks (landscape + portrait) as PNGs and PDFs
  4. Generate narration audio via ElevenLabs TTS
  5. Assemble final videos via ffmpeg

Each desk produces its briefing independently, as if a department head
is presenting their weekly stand-up to the agency leadership.
"""

import asyncio
import json
import logging
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.intelligence.qdrant_store import QdrantStore
from src.intelligence.slide_renderer import SlideRenderer
from src.intelligence.tts_client import QuotaExceededError, TTSClient
from src.intelligence.venice_image_client import VeniceImageClient

load_dotenv()

logger = logging.getLogger("osia.briefing_generator")

_REPO_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "weekly_briefing.yaml"


def _load_config() -> dict:
    """Load weekly briefing configuration."""
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _week_label(dt: datetime) -> str:
    """Return ISO week label like '2026-W14'."""
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _extract_persona_name(persona_text: str) -> str:
    """Extract the character name from the persona description."""
    # Look for patterns like "Director Marcus Hale" or "Dr. Amara Osei"
    match = re.search(
        r"(?:Director|Dr\.|Commander|Agent|Professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        persona_text,
    )
    if match:
        return match.group(0)
    # Fallback: "You are X," pattern
    match = re.search(r"You are\s+(.+?),", persona_text)
    return match.group(1) if match else "Department Head"


async def _query_desk_intel(
    qdrant: QdrantStore,
    collection: str,
    lookback_days: int,
) -> list[dict]:
    """Pull recent intelligence from a desk's Qdrant collection.

    Returns list of dicts with text, score, source, timestamp.
    """
    # Use a broad query to surface the most significant recent entries
    queries = [
        "most important intelligence developments this week",
        "critical threats emerging risks urgent developments",
        "significant changes strategic implications key findings",
    ]

    all_results = []
    seen_texts: set[str] = set()

    for query in queries:
        try:
            results = await qdrant.search(
                collection,
                query,
                top_k=10,
                decay_half_life_days=float(lookback_days),
            )
            for r in results:
                # Deduplicate by text prefix
                fingerprint = r.text[:150]
                if fingerprint not in seen_texts:
                    seen_texts.add(fingerprint)
                    all_results.append(
                        {
                            "text": r.text,
                            "score": r.score,
                            "source": r.metadata.get("source", "unknown"),
                            "timestamp": r.metadata.get("timestamp", ""),
                            "triggered_by": r.metadata.get("triggered_by", ""),
                            "reliability_tier": r.metadata.get("reliability_tier", "?"),
                        }
                    )
        except Exception as e:
            logger.warning("Qdrant query failed for '%s': %s", collection, e)

    # Sort by score descending, take top entries
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:15]


def _build_briefing_prompt(
    desk_name: str,
    persona: str,
    intel_items: list[dict],
    week_label: str,
    directives: str,
) -> str:
    """Build the LLM prompt for generating a structured slide briefing."""
    intel_block = ""
    if intel_items:
        entries = []
        for i, item in enumerate(intel_items, 1):
            ts = item["timestamp"][:10] if item["timestamp"] else "undated"
            entries.append(
                f"[{i}] (Reliability: {item['reliability_tier']}, Score: {item['score']:.2f}) "
                f"{ts}\nSource: {item['source']}\n{item['text'][:800]}"
            )
        intel_block = "\n\n---\n\n".join(entries)
    else:
        intel_block = "No intelligence was collected this week for this desk."

    return f"""
{directives}

---

{persona}

You are delivering your weekly intelligence briefing for {week_label} to the agency leadership.
Your department is the {desk_name}.

## YOUR INTELLIGENCE FOR THIS WEEK

{intel_block}

## INSTRUCTIONS

Generate a structured briefing presentation. You must respond with VALID JSON only.
The JSON must be an array of slide objects. Each slide has these fields:
- "slide_type": one of "title", "content", or "closing"
- "title": the slide heading
- "body": markdown-formatted slide content (bullet points, bold, etc.)
- "narration": the FULL spoken script for this slide — written as natural speech,
  as if you are standing at a podium briefing senior intelligence officials.
  This will be read aloud by a text-to-speech system, so write it conversationally
  but with authority. Include pauses with "..." where appropriate.

## SLIDE STRUCTURE

1. **Title slide** (slide_type: "title"): Your department name and a one-line theme for the week.
   Narration: A confident opening greeting and preview of what you'll cover.

2. **3-5 content slides** (slide_type: "content"): Each covers one major intelligence item.
   - Title: Clear, punchy headline
   - Body: 3-5 bullet points with key facts (keep text concise for slides)
   - Narration: 30-60 seconds of spoken analysis per slide. Provide context,
     implications, and your professional assessment. Reference specific sources,
     dates, and actors. Speak with the gravitas of a seasoned intelligence professional.

3. **Closing slide** (slide_type: "closing"): "Watch List & Recommendations"
   - Body: 3-4 items to monitor in the coming week
   - Narration: Summarise key takeaways, flag items requiring immediate attention,
     and close with a professional sign-off.

## CRITICAL RULES

- Respond with ONLY the JSON array. No markdown fences, no preamble.
- Narration must be written as natural speech — no bullet points, no markdown.
- Each narration should be 2-4 paragraphs of flowing speech.
- Reference the analytical mandate: anti-imperialism, labor rights, ecological justice.
- Be specific: name actors, cite dates, reference source reliability.
- Maintain your character's voice and expertise throughout.
"""


def _parse_slides_json(raw_text: str) -> list[dict]:
    """Parse the LLM's JSON response into a list of slide dicts.

    Handles common LLM quirks: markdown fences, trailing commas, preamble text.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in LLM response")

    json_str = text[start : end + 1]

    # Remove trailing commas before ] or }
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    slides = json.loads(json_str)
    if not isinstance(slides, list):
        raise ValueError(f"Expected JSON array, got {type(slides)}")

    # Validate required fields
    for i, slide in enumerate(slides):
        if not isinstance(slide, dict):
            raise ValueError(f"Slide {i} is not a dict")
        for key in ("slide_type", "title", "narration"):
            if key not in slide:
                raise ValueError(f"Slide {i} missing required key '{key}'")
        # Ensure body exists (can be empty for title slides)
        slide.setdefault("body", "")
        # LLMs sometimes return body as a list of bullet strings
        if isinstance(slide["body"], list):
            slide["body"] = "\n".join(f"- {item}" for item in slide["body"])

    return slides


async def _assemble_video(
    image_paths: list[Path],
    audio_paths: list[Path],
    output_path: Path,
    config: dict,
) -> Path:
    """Assemble slide images + narration audio into a video using ffmpeg.

    Each slide is shown for the duration of its audio narration,
    with a minimum display time from config.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_cfg = config.get("video", {})
    slide_cfg = config.get("slides", {})
    min_duration = slide_cfg.get("min_duration_secs", 8)
    codec = video_cfg.get("codec", "libx264")
    audio_codec = video_cfg.get("audio_codec", "aac")
    crf = video_cfg.get("crf", 23)
    preset = video_cfg.get("preset", "medium")
    bitrate = video_cfg.get("bitrate", "2M")
    use_rkmpp = codec.endswith("_rkmpp")

    if len(image_paths) != len(audio_paths):
        raise ValueError(f"Mismatch: {len(image_paths)} images vs {len(audio_paths)} audio files")

    # Transition/fade config
    transition_secs = slide_cfg.get("transition_secs", 0.5)
    fade_secs = video_cfg.get("fade_duration", 0.3)
    pause_ms = int(transition_secs * 1000)

    # Get audio durations using ffprobe
    durations: list[float] = []
    for audio_path in audio_paths:
        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            dur = float(probe.stdout.strip())
            durations.append(max(dur, min_duration))
        except Exception as e:
            logger.warning("ffprobe failed for %s: %s — using min duration", audio_path, e)
            durations.append(min_duration)

    # Build ffmpeg concat demuxer input file
    concat_dir = output_path.parent / "tmp_concat"
    concat_dir.mkdir(exist_ok=True)

    # Create individual slide videos with their audio
    segment_paths: list[Path] = []
    for i, (img, audio, dur) in enumerate(zip(image_paths, audio_paths, durations, strict=True)):
        segment_path = concat_dir / f"segment_{i:02d}.mp4"

        # Total segment duration includes the silence pad so the fade tail has room
        total_dur = dur + transition_secs
        fade_out_start = total_dur - fade_secs

        # Audio: silence pad before narration, then fade in/out
        audio_filter = (
            f"adelay={pause_ms}|{pause_ms},"
            f"afade=t=in:st=0:d={fade_secs:.3f},"
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_secs:.3f}"
        )
        # Video: fade in from black, fade out to black
        video_filter = f"fade=t=in:st=0:d={fade_secs:.3f},fade=t=out:st={fade_out_start:.3f}:d={fade_secs:.3f}"

        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(img),
            "-i",
            str(audio),
            "-vf",
            video_filter,
            "-af",
            audio_filter,
            "-c:v",
            codec,
            "-c:a",
            audio_codec,
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-t",
            f"{total_dur:.3f}",
        ]
        if use_rkmpp:
            cmd += ["-b:v", bitrate]
        else:
            cmd += ["-tune", "stillimage", "-crf", str(crf), "-preset", preset]
        cmd.append(str(segment_path))

        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            logger.error("ffmpeg segment %d failed: %s", i, proc.stderr[-500:])
            raise RuntimeError(f"ffmpeg failed on segment {i}")

        segment_paths.append(segment_path)

    # Write concat list
    concat_list = concat_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for seg in segment_paths:
            f.write(f"file '{seg.resolve()}'\n")

    # Concatenate all segments
    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(output_path),
    ]

    proc = await asyncio.to_thread(
        subprocess.run,
        concat_cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        logger.error("ffmpeg concat failed: %s", proc.stderr[-500:])
        raise RuntimeError("ffmpeg concat failed")

    # Cleanup temp files
    for seg in segment_paths:
        seg.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)
    try:
        concat_dir.rmdir()
    except OSError as e:
        logger.debug("Failed to remove concat temp directory %s: %s", concat_dir, e)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Video assembled: %s (%.1f MB)", output_path.name, file_size_mb)
    return output_path


async def generate_desk_briefing(
    desk_slug: str,
    global_config: dict,
    qdrant: QdrantStore,
    directives: str,
    output_base: Path,
    week_label: str,
    resume: bool = False,
) -> dict:
    """Generate a complete weekly briefing for a single desk.

    Voice and persona are read from the desk's YAML config (briefing block).
    Returns a dict with paths to all generated artifacts.
    """
    from src.desks.desk_registry import DeskRegistry

    # Load desk metadata from registry
    registry = DeskRegistry()
    try:
        desk_cfg = registry.get(desk_slug)
        desk_name = desk_cfg.name
        collection = desk_cfg.qdrant.collection
    except KeyError:
        logger.error("Desk '%s' not found in registry", desk_slug)
        return {"error": f"Desk '{desk_slug}' not found"}

    if not desk_cfg.briefing:
        logger.error("Desk '%s' has no briefing config — skipping", desk_slug)
        return {"error": f"Desk '{desk_slug}' has no briefing config"}

    voice_id = desk_cfg.briefing.voice_id
    persona = desk_cfg.briefing.persona
    persona_name = _extract_persona_name(persona)

    logger.info("═══ Generating briefing for %s (%s) ═══", desk_name, persona_name)

    # Resume: skip this desk if all orientation videos already exist
    if resume:
        formats = global_config.get("output", {}).get("formats", ["landscape", "portrait"])
        existing_videos = [
            output_base / desk_slug / orientation / f"{desk_slug}_{week_label}_{orientation}.mp4"
            for orientation in formats
        ]
        if all(p.exists() for p in existing_videos):
            logger.info("Resume: skipping %s — all videos already exist", desk_slug)
            return {
                "desk_slug": desk_slug,
                "desk_name": desk_name,
                "persona_name": persona_name,
                "skipped": True,
                **{f"video_{orientation}": str(p) for orientation, p in zip(formats, existing_videos, strict=True)},
            }

    slides_cache = output_base / desk_slug / "slides.json"

    # Resume: reload slides from the cached JSON if it exists, skipping the LLM call
    if resume and slides_cache.exists():
        slides = json.loads(slides_cache.read_text(encoding="utf-8"))
        logger.info("Resume: loaded %d slides from cache for %s", len(slides), desk_slug)
        await registry.close()
    else:
        # 1. Query Qdrant for recent intel
        lookback = global_config.get("schedule", {}).get("lookback_days", 7)
        intel_items = await _query_desk_intel(qdrant, collection, lookback)
        logger.info("Retrieved %d intel items from '%s'", len(intel_items), collection)

        if not intel_items:
            logger.warning("No intel found for %s — skipping briefing", desk_slug)
            await registry.close()
            return {"desk_slug": desk_slug, "skipped": True, "reason": "no intel"}

        # 2. Generate structured briefing via desk LLM
        prompt = _build_briefing_prompt(desk_name, persona, intel_items, week_label, directives)

        try:
            response_text, invoke_meta = await registry.invoke(desk_slug, prompt)
            logger.info(
                "Briefing generated by %s (%s)",
                invoke_meta.get("model_id", "unknown"),
                invoke_meta.get("model_used", "primary"),
            )
        except Exception as e:
            logger.error("LLM briefing generation failed for %s: %s", desk_slug, e)
            return {"error": str(e)}
        finally:
            await registry.close()

        # 3. Parse slides JSON
        try:
            slides = _parse_slides_json(response_text)
            logger.info("Parsed %d slides for %s", len(slides), desk_slug)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse slides JSON for %s: %s", desk_slug, e)
            logger.debug("Raw LLM response:\n%s", response_text[:2000])
            return {"error": f"JSON parse failed: {e}"}

        # Persist slides so --resume can reload them without re-calling the LLM
        slides_cache.parent.mkdir(parents=True, exist_ok=True)
        slides_cache.write_text(json.dumps(slides, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4. Render slides + generate audio + assemble video for each orientation
    tts_cfg = global_config.get("tts", {})
    tts = TTSClient(
        model_id=tts_cfg.get("model_id", "eleven_v3"),
        output_format=tts_cfg.get("output_format", "mp3_44100_128"),
    )
    renderer = SlideRenderer()

    # Generate AI background images via Venice (shared between orientations)
    img_cfg = global_config.get("image_generation", {})
    bg_images_by_orientation: dict[str, list[Path | None]] = {}
    if img_cfg.get("enabled", False):
        venice_img = VeniceImageClient(
            model=img_cfg.get("model", "flux-2-pro"),
        )
        formats = global_config.get("output", {}).get("formats", ["landscape", "portrait"])
        from src.intelligence.slide_renderer import DIMENSIONS

        for orientation in formats:
            w, h = DIMENSIONS[orientation]
            img_dir = output_base / desk_slug / orientation / "bg_images"
            bg_images = await venice_img.generate_slide_images(
                slides=slides,
                desk_name=desk_name,
                width=w,
                height=h,
                output_dir=img_dir,
                resume=resume,
            )
            bg_images_by_orientation[orientation] = bg_images
    else:
        logger.debug("Image generation disabled — slides will use plain backgrounds")

    artifacts: dict = {
        "desk_slug": desk_slug,
        "desk_name": desk_name,
        "persona_name": persona_name,
        "slides_count": len(slides),
    }

    formats = global_config.get("output", {}).get("formats", ["landscape", "portrait"])

    # Generate audio once (shared between orientations)
    audio_dir = output_base / desk_slug / "audio"
    audio_paths = await tts.generate_slide_narrations(
        slides,
        voice_id,
        audio_dir,
        stability=tts_cfg.get("stability", 0.5),
        similarity_boost=tts_cfg.get("similarity_boost", 0.75),
        style=tts_cfg.get("style", 0.4),
        resume=resume,
    )
    artifacts["audio_dir"] = str(audio_dir)

    for orientation in formats:
        orient_dir = output_base / desk_slug / orientation

        # Render slide images
        bg_images = bg_images_by_orientation.get(orientation)
        image_paths = renderer.render_deck(
            slides=slides,
            desk_slug=desk_slug,
            desk_name=desk_name,
            persona_name=persona_name,
            orientation=orientation,
            output_dir=orient_dir / "slides",
            week_label=week_label,
            bg_images=bg_images,
        )

        # Render PDF deck
        pdf_path = orient_dir / f"{desk_slug}_{week_label}_{orientation}.pdf"
        renderer.render_pdf_deck(
            slides=slides,
            desk_slug=desk_slug,
            desk_name=desk_name,
            persona_name=persona_name,
            orientation=orientation,
            output_path=pdf_path,
            week_label=week_label,
            bg_images=bg_images,
        )
        artifacts[f"pdf_{orientation}"] = str(pdf_path)

        # Assemble video
        video_path = orient_dir / f"{desk_slug}_{week_label}_{orientation}.mp4"
        try:
            await _assemble_video(image_paths, audio_paths, video_path, global_config)
            artifacts[f"video_{orientation}"] = str(video_path)
        except Exception as e:
            logger.error("Video assembly failed for %s/%s: %s", desk_slug, orientation, e)
            artifacts[f"video_{orientation}_error"] = str(e)

    return artifacts


async def _upload_to_youtube(
    uploader,
    artifacts: dict,
    config: dict,
    week_label: str,
) -> None:
    """Upload a desk's landscape video to YouTube. Runs as a background task.

    Mutates artifacts dict in-place to add youtube_url / youtube_error.
    """
    desk_slug = artifacts.get("desk_slug", "unknown")
    desk_name = artifacts.get("desk_name", desk_slug)
    persona_name = artifacts.get("persona_name", "Department Head")
    landscape_video = artifacts.get("video_landscape")

    if not landscape_video:
        return

    try:
        yt_title = f"{desk_name} — Weekly Intelligence Briefing — {week_label}"
        yt_desc = (
            f"{desk_name} weekly intelligence briefing for {week_label}.\n"
            f"Presented by {persona_name}.\n\n"
            f"Generated by OSIA — Open Source Intelligence Agency.\n"
            f"https://github.com/osianet/osia-framework"
        )
        yt_tags = config.get("tags", ["OSINT", "intelligence", "briefing", "OSIA"])
        yt_privacy = config.get("privacy_status", "unlisted")
        yt_category = config.get("category_id", "25")

        result = await uploader.upload(
            video_path=Path(landscape_video),
            title=yt_title,
            description=yt_desc,
            tags=yt_tags,
            category_id=yt_category,
            privacy_status=yt_privacy,
        )
        artifacts["youtube_url"] = result["url"]
        artifacts["youtube_video_id"] = result["video_id"]
        logger.info("YouTube upload complete: %s → %s", desk_slug, result["url"])
    except Exception as e:
        logger.error("YouTube upload failed for %s: %s", desk_slug, e)
        artifacts["youtube_error"] = str(e)


async def generate_all_briefings(
    desks: list[str] | None = None,
    resume: bool = False,
) -> list[dict]:
    """Generate weekly briefings for all desks that have a briefing config.

    Discovers desks from the DeskRegistry (config/desks/*.yaml) rather than
    weekly_briefing.yaml. Only desks with a `briefing:` block are included.

    Args:
        desks: Optional list of desk slugs to restrict processing to.
        resume: If True, skip desks whose videos already exist and skip
                individual slides whose audio already exists.

    Returns list of artifact dicts (one per desk).
    """
    from src.desks.desk_registry import DeskRegistry

    config = _load_config()
    now = datetime.now(UTC)
    wl = _week_label(now)

    output_base = _REPO_ROOT / config.get("output", {}).get("base_dir", "reports/weekly") / wl

    # Load directives
    directives_path = _REPO_ROOT / "DIRECTIVES.md"
    directives = directives_path.read_text(encoding="utf-8") if directives_path.exists() else ""

    qdrant = QdrantStore()

    # Discover desks with briefing config from the registry
    registry = DeskRegistry()
    briefing_desks = [slug for slug in registry.list_slugs() if registry.get(slug).briefing is not None]
    await registry.close()

    if not briefing_desks:
        logger.error("No desks have a briefing config in config/desks/*.yaml")
        return []

    if desks:
        unknown = [d for d in desks if d not in briefing_desks]
        if unknown:
            logger.warning("Unknown or non-briefing desk slugs ignored: %s", unknown)
        briefing_desks = [d for d in briefing_desks if d in desks]
        if not briefing_desks:
            logger.error("None of the requested desks have a briefing config")
            return []

    logger.info(
        "Starting weekly briefings for %s — %d desks%s",
        wl,
        len(briefing_desks),
        " (resume mode)" if resume else "",
    )

    results: list[dict] = []
    upload_tasks: list[asyncio.Task] = []
    start_time = time.monotonic()

    # Initialise YouTube uploader once if enabled
    yt_cfg = config.get("youtube", {})
    yt_uploader = None
    if yt_cfg.get("enabled", False):
        try:
            from src.intelligence.youtube_uploader import YouTubeUploader

            yt_uploader = YouTubeUploader()
            logger.info("YouTube uploads enabled — videos will upload in background")
        except Exception as e:
            logger.warning("YouTube uploader init failed — uploads disabled: %s", e)

    # Process desks sequentially to avoid overwhelming APIs
    for desk_slug in briefing_desks:
        try:
            result = await generate_desk_briefing(
                desk_slug=desk_slug,
                global_config=config,
                qdrant=qdrant,
                directives=directives,
                output_base=output_base,
                week_label=wl,
                resume=resume,
            )
            results.append(result)

            # Fire YouTube upload as a background task (doesn't block next desk)
            landscape_video = result.get("video_landscape")
            if yt_uploader and landscape_video:
                task = asyncio.create_task(
                    _upload_to_youtube(
                        uploader=yt_uploader,
                        artifacts=result,
                        config=yt_cfg,
                        week_label=wl,
                    )
                )
                upload_tasks.append(task)

        except QuotaExceededError as e:
            logger.error("ElevenLabs quota exhausted — stopping briefing pipeline: %s", e)
            results.append({"desk_slug": desk_slug, "error": str(e)})
            break
        except Exception as e:
            logger.exception("Briefing failed for %s: %s", desk_slug, e)
            results.append({"desk_slug": desk_slug, "error": str(e)})

    # Wait for any in-flight YouTube uploads to finish
    if upload_tasks:
        logger.info("Waiting for %d YouTube upload(s) to complete...", len(upload_tasks))
        await asyncio.gather(*upload_tasks, return_exceptions=True)

    elapsed = time.monotonic() - start_time
    successful = sum(1 for r in results if "error" not in r)
    logger.info(
        "Weekly briefings complete: %d/%d successful in %.1f minutes",
        successful,
        len(results),
        elapsed / 60,
    )

    # Write manifest
    manifest_path = output_base / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "week": wl,
                "generated_at": now.isoformat(),
                "desks": results,
            },
            f,
            indent=2,
        )
    logger.info("Manifest written: %s", manifest_path)

    return results
