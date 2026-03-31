"""
Venice AI image generation client for OSIA weekly briefing slides.

Generates topic-relevant background images via Venice's native image API,
returning raw PNG bytes suitable for embedding into slide templates.
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

# Defaults — overridable via config/weekly_briefing.yaml → image_generation block
DEFAULT_MODEL = "flux-2-pro"
DEFAULT_STYLE_PRESET = "Digital Art"
DEFAULT_STEPS = 20
DEFAULT_CFG_SCALE = 7.5


class VeniceImageClient:
    """Generates slide background images via Venice AI."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        style_preset: str = DEFAULT_STYLE_PRESET,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        negative_prompt: str = "",
    ) -> None:
        self.model = model
        self.style_preset = style_preset
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.negative_prompt = negative_prompt

    async def generate(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 1080,
        output_path: Path | None = None,
    ) -> bytes:
        """Generate an image and return raw PNG bytes.

        Args:
            prompt: Text description of the desired image.
            width: Image width in pixels.
            height: Image height in pixels.
            output_path: If provided, also write the image to disk.

        Returns:
            Raw PNG image bytes.
        """
        if not VENICE_API_KEY:
            raise RuntimeError("VENICE_API_KEY not set — cannot generate images")

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "format": "png",
            "return_binary": False,
            "hide_watermark": True,
            "safe_mode": False,
            "variants": 1,
        }
        if self.style_preset:
            payload["style_preset"] = self.style_preset
        if self.negative_prompt:
            payload["negative_prompt"] = self.negative_prompt

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
                    resp.raise_for_status()

                data = resp.json()
                # Venice returns images as base64 in the response
                images = data.get("images", [])
                if not images:
                    raise ValueError("Venice returned no images")

                image_bytes = base64.b64decode(images[0])

                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(image_bytes)
                    logger.debug("Saved image: %s", output_path)

                return image_bytes

            except httpx.HTTPStatusError as e:
                logger.warning("Venice image attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                logger.warning("Venice image attempt %d error: %s", attempt + 1, e)
                await asyncio.sleep(5 * (attempt + 1))

        raise RuntimeError(f"Venice image generation failed after 3 attempts for prompt: {prompt[:80]}")

    async def generate_slide_images(
        self,
        slides: list[dict],
        desk_name: str,
        width: int,
        height: int,
        output_dir: Path,
        resume: bool = False,
    ) -> list[Path | None]:
        """Generate background images for each slide in a deck.

        Builds a cinematic prompt from each slide's title and body, tuned
        for dark-themed intelligence briefing aesthetics.

        Args:
            slides: List of slide dicts (title, body, slide_type).
            desk_name: Desk name for thematic context.
            width: Target image width.
            height: Target image height.
            output_dir: Directory to save generated images.
            resume: Skip generation if the image file already exists.

        Returns:
            List of Paths (one per slide, None if generation failed).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[Path | None] = []

        for i, slide in enumerate(slides):
            img_path = output_dir / f"bg_{i:02d}.png"

            if resume and img_path.exists():
                logger.debug("Resume: reusing existing image %s", img_path.name)
                results.append(img_path)
                continue

            prompt = self._build_image_prompt(slide, desk_name)
            try:
                await self.generate(prompt, width, height, img_path)
                results.append(img_path)
                logger.info("Generated slide image %d/%d: %s", i + 1, len(slides), img_path.name)
            except Exception as e:
                logger.warning("Image generation failed for slide %d: %s — slide will render without image", i, e)
                results.append(None)

        return results

    @staticmethod
    def _build_image_prompt(slide: dict, desk_name: str) -> str:
        """Build a cinematic image prompt from slide content.

        The prompt is designed to produce dark, atmospheric images that
        work as subtle backgrounds behind text — not literal illustrations.
        """
        title = slide.get("title", "")
        body = slide.get("body", "")
        slide_type = slide.get("slide_type", "content")

        # Extract key themes from body (first ~200 chars, strip markdown)
        body_hint = body[:200].replace("*", "").replace("#", "").replace("-", "").strip()

        if slide_type == "title":
            return (
                f"Dark cinematic wide-angle establishing shot for an intelligence agency briefing "
                f"about {desk_name}. Moody atmosphere, deep shadows, dark green and amber tones, "
                f"subtle digital overlay effects. Theme: {title}. "
                f"No text, no words, no letters, no UI elements. Photorealistic, 8k, dramatic lighting."
            )
        elif slide_type == "closing":
            return (
                f"Dark atmospheric abstract composition suggesting strategic foresight and vigilance. "
                f"Subtle radar or surveillance motifs, deep green and amber color palette, "
                f"cinematic lighting, intelligence agency aesthetic. Theme: {title}. "
                f"No text, no words, no letters. Photorealistic, 8k."
            )
        else:
            return (
                f"Dark cinematic background image for an intelligence briefing slide. "
                f"Topic: {title}. Context: {body_hint}. "
                f"Moody atmosphere, deep shadows, dark green and amber accent tones, "
                f"subtle depth of field. Evocative but not literal — abstract enough to sit "
                f"behind text. No text, no words, no letters, no UI elements. "
                f"Photorealistic, 8k, dramatic lighting."
            )
