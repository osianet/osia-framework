"""
Venice AI image generation client for OSIA weekly briefing slides.

Uses the native /image/generate endpoint which accepts width/height directly
and supports the full Venice feature set (negative prompts, safe_mode, etc.).
Returns raw PNG bytes suitable for embedding into slide templates.
"""

import asyncio
import base64
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("osia.venice_image")

VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")
VENICE_IMAGE_URL = "https://api.venice.ai/api/v1/image/generate"

DEFAULT_MODEL = "flux-2-pro"

# Negative prompt used for all slide backgrounds — keeps images clean for text overlay.
_NEGATIVE_PROMPT = (
    "text, words, letters, numbers, watermark, signature, logo, UI elements, "
    "buttons, labels, captions, headlines, title, subtitle, banner, "
    "bright white background, overexposed, washed out, blurry, low quality, "
    "cartoon, anime, illustration, clipart, stock photo, people faces"
)

# Accent color hex values from aesthetic.yaml.
_ACCENT_COLORS: dict[str, str] = {
    "amber_alert": "#C8860A",
    "liberation_red": "#8B1A1A",
    "data_teal": "#0D4A4A",
    "earth_ochre": "#6B4C1E",
    "signal_green": "#1A3A2A",
}

# Per-desk aesthetic overrides from config/aesthetic.yaml — inlined here to avoid
# loading YAML at import time.  Keep in sync with the config file.
_DESK_AESTHETICS: dict[str, dict[str, str]] = {
    "geopolitical-and-security-desk": {
        "accent": "amber_alert",
        "motif": "ancient cartography, borders dissolving, contested territories, geopolitical maps",
        "mood": "grave, authoritative, geographically vast",
    },
    "cultural-and-theological-intelligence-desk": {
        "accent": "earth_ochre",
        "motif": "sacred geometry, manuscript illumination, oral tradition patterns, temple architecture",
        "mood": "contemplative, layered, spiritually grounded",
    },
    "science-technology-and-commercial-desk": {
        "accent": "data_teal",
        "motif": "molecular structures, circuit traces, dual-use technology shadows, laboratory glass",
        "mood": "precise, analytical, ethically questioning",
    },
    "human-intelligence-and-profiling-desk": {
        "accent": "liberation_red",
        "motif": "network graphs, shadow profiles, redacted faces, dossier pages",
        "mood": "intimate, unsettling, humanising",
    },
    "finance-and-economics-directorate": {
        "accent": "amber_alert",
        "motif": "flow diagrams, hidden ledgers, supply chain maps, currency textures",
        "mood": "forensic, cold, exposing",
    },
    "cyber-intelligence-and-warfare-desk": {
        "accent": "data_teal",
        "motif": "binary rain, exploit code fragments, dark web topology, terminal screens",
        "mood": "technical, urgent, adversarial",
    },
    "information-warfare-desk": {
        "accent": "liberation_red",
        "motif": "fractured mirrors, propaganda poster aesthetics, signal jamming, distorted broadcasts",
        "mood": "disorienting, critical, counter-narrative",
    },
    "environment-and-ecology-desk": {
        "accent": "signal_green",
        "motif": "watershed maps, deforestation overlays, mycelium networks, satellite imagery of forests",
        "mood": "urgent, reverent, ecological grief",
    },
    "the-watch-floor": {
        "accent": "amber_alert",
        "motif": "convergence point, multiple data streams merging, situation room displays, radar sweeps",
        "mood": "composed, comprehensive, decisive",
    },
}


def _clamp_dimensions(width: int, height: int) -> tuple[int, int]:
    """Scale dimensions down so neither side exceeds Venice's 1280px limit.

    Preserves aspect ratio.  If both sides are already within bounds the
    original values are returned unchanged.
    """
    max_side = 1280
    if width <= max_side and height <= max_side:
        return width, height
    scale = min(max_side / width, max_side / height)
    return int(width * scale), int(height * scale)


class VeniceImageClient:
    """Generates slide background images via Venice AI."""

    def __init__(self, model: str = DEFAULT_MODEL, **_kwargs) -> None:
        self.model = model

    async def generate(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 1080,
        output_path: Path | None = None,
        negative_prompt: str | None = None,
    ) -> bytes:
        """Generate an image and return raw PNG bytes.

        Args:
            prompt: Text description of the desired image.
            width: Image width in pixels (clamped to Venice's 1280px max).
            height: Image height in pixels (clamped to Venice's 1280px max).
            output_path: If provided, also write the image to disk.
            negative_prompt: Things to avoid in the generated image.

        Returns:
            Raw PNG image bytes.
        """
        if not VENICE_API_KEY:
            raise RuntimeError("VENICE_API_KEY not set — cannot generate images")

        clamped_w, clamped_h = _clamp_dimensions(width, height)

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "width": clamped_w,
            "height": clamped_h,
            "format": "png",
            "safe_mode": False,
            "return_binary": False,
            "hide_watermark": True,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=90.0) as http:
                    resp = await http.post(
                        VENICE_IMAGE_URL,
                        headers={
                            "Authorization": f"Bearer {VENICE_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    if resp.status_code == 429:
                        wait = 30 * (attempt + 1)
                        logger.warning("Venice image 429 — waiting %ds", wait)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status_code == 400:
                        logger.error("Venice image 400: %s", resp.text[:500])
                        raise httpx.HTTPStatusError(
                            f"400 Bad Request: {resp.text[:200]}",
                            request=resp.request,
                            response=resp,
                        )
                    resp.raise_for_status()

                data = resp.json()
                # Native endpoint returns {"images": ["<base64>", ...]}
                images = data.get("images", [])
                if not images:
                    raise ValueError("Venice returned no images")

                image_bytes = base64.b64decode(images[0])

                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(image_bytes)
                    logger.debug("Saved image: %s (%dx%d)", output_path, clamped_w, clamped_h)

                return image_bytes

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    raise  # Don't retry 400s — the request itself is wrong
                logger.warning("Venice image attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                logger.warning("Venice image attempt %d error: %s", attempt + 1, e)
                await asyncio.sleep(5 * (attempt + 1))

        raise RuntimeError(f"Venice image generation failed after 3 attempts for prompt: {prompt[:80]}")

    async def remove_background(
        self,
        image_bytes: bytes,
        output_path: Path | None = None,
    ) -> bytes:
        """Remove the background from an image via Venice AI.

        Args:
            image_bytes: Raw PNG/JPEG bytes of the source image.
            output_path: If provided, write the result PNG to disk.

        Returns:
            Raw PNG bytes with transparent background.
        """
        if not VENICE_API_KEY:
            raise RuntimeError("VENICE_API_KEY not set")

        b64 = base64.b64encode(image_bytes).decode()

        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.post(
                "https://api.venice.ai/api/v1/image/background-remove",
                headers={"Authorization": f"Bearer {VENICE_API_KEY}"},
                json={"image": b64},
            )
            resp.raise_for_status()

        result = resp.content
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(result)
            logger.debug("Saved bg-removed image: %s", output_path)
        return result

    async def generate_slide_images(
        self,
        slides: list[dict],
        desk_name: str,
        width: int,
        height: int,
        output_dir: Path,
        resume: bool = False,
        desk_slug: str | None = None,
    ) -> list[Path | None]:
        """Generate background images for each slide in a deck.

        Args:
            slides: List of slide dicts (title, body, slide_type).
            desk_name: Desk name for thematic context.
            width: Target image width.
            height: Target image height.
            output_dir: Directory to save generated images.
            resume: Skip generation if the image file already exists.
            desk_slug: Desk slug for per-desk aesthetic lookup.

        Returns:
            List of Paths (one per slide, None if generation failed).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[Path | None] = []
        desk_aesthetic = _DESK_AESTHETICS.get(desk_slug or "", {})

        for i, slide in enumerate(slides):
            img_path = output_dir / f"bg_{i:02d}.png"

            if resume and img_path.exists():
                logger.debug("Resume: reusing existing image %s", img_path.name)
                results.append(img_path)
                continue

            prompt, negative = self._build_image_prompt(slide, desk_name, desk_aesthetic)
            try:
                await self.generate(prompt, width, height, img_path, negative_prompt=negative)
                results.append(img_path)
                logger.info("Generated slide image %d/%d: %s", i + 1, len(slides), img_path.name)
            except Exception as e:
                logger.warning("Image generation failed for slide %d: %s — slide will render without image", i, e)
                results.append(None)

        return results

    @staticmethod
    def _build_image_prompt(slide: dict, desk_name: str, desk_aesthetic: dict) -> tuple[str, str]:
        """Build a cinematic image prompt from slide content and desk aesthetic.

        Returns (prompt, negative_prompt) tuple.  The prompt grounds the image
        in the specific desk's visual identity from aesthetic.yaml while keeping
        it dark and abstract enough to sit behind text.
        """
        title = slide.get("title", "")
        body = slide.get("body", "")
        slide_type = slide.get("slide_type", "content")

        # Extract topical keywords from body — strip markdown noise.
        body_clean = body[:300]
        for ch in ("*", "#", "-", ">", "`", "[", "]", "(", ")"):
            body_clean = body_clean.replace(ch, "")
        # Collapse whitespace and take first ~200 meaningful chars.
        body_hint = " ".join(body_clean.split())[:200].strip()

        # Per-desk visual identity (falls back to generic if missing).
        motif = desk_aesthetic.get("motif", "topographic contour lines, network nodes, archival textures")
        mood = desk_aesthetic.get("mood", "vigilant, grounded, purposeful")
        accent_name = desk_aesthetic.get("accent", "amber_alert")
        accent_color = _ACCENT_COLORS.get(accent_name, "#C8860A")

        # Shared style foundation.
        style_base = (
            f"Cinematic low-key lighting, single directional light source cutting through deep shadow. "
            f"Color palette: near-black (#0A0C0B) background with dark forest green (#1A3A2A) "
            f"and {accent_name.replace('_', ' ')} ({accent_color}) accents. "
            f"Textured grain, subtle noise. Mood: {mood}. "
            f"Visual motifs: {motif}."
        )

        negative = _NEGATIVE_PROMPT

        if slide_type == "title":
            prompt = (
                f"Wide-angle establishing shot for a {desk_name} intelligence briefing. "
                f"Theme: {title}. "
                f"{style_base} "
                f"Atmospheric depth, environmental storytelling, sense of scale and gravity."
            )
        elif slide_type == "closing":
            prompt = (
                f"Abstract composition suggesting strategic foresight and synthesis. "
                f"Theme: {title}. Context: {body_hint}. "
                f"{style_base} "
                f"Convergence of multiple visual threads, sense of resolution and watchfulness."
            )
        else:
            prompt = (
                f"Background image for an intelligence briefing slide about: {title}. "
                f"Subject context: {body_hint}. "
                f"Department: {desk_name}. "
                f"{style_base} "
                f"Abstract and evocative — suggests the topic without literal illustration. "
                f"Subtle depth of field, atmospheric haze."
            )

        return prompt, negative
