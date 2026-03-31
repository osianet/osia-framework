"""
ElevenLabs TTS client for OSIA weekly briefings.

Wraps the ElevenLabs Python SDK to generate high-quality narration
audio for department head briefing presentations.

Environment variables:
  ELEVENLABS_API_KEY — ElevenLabs API key
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("osia.tts")


class QuotaExceededError(RuntimeError):
    """Raised when the ElevenLabs account quota is exhausted."""


class TTSClient:
    """Async wrapper around ElevenLabs TTS for briefing narration."""

    def __init__(
        self,
        model_id: str = "eleven_v3",
        output_format: str = "mp3_44100_128",
    ) -> None:
        from elevenlabs.client import ElevenLabs

        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")

        self._client = ElevenLabs(api_key=api_key)
        self._model_id = model_id
        self._output_format = output_format

    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        *,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.4,
        previous_text: str | None = None,
        next_text: str | None = None,
    ) -> Path:
        """Generate speech audio from text and write to output_path.

        Args:
            text: The narration script to speak.
            voice_id: ElevenLabs voice ID for the department head.
            output_path: Where to write the MP3 file.
            stability: Voice stability (lower = more expressive).
            similarity_boost: How closely to match the original voice.
            style: Style exaggeration (0-1).
            previous_text: Context from previous slide for continuity.
            next_text: Context from next slide for continuity.

        Returns:
            Path to the generated audio file.
        """
        from elevenlabs import VoiceSettings

        output_path.parent.mkdir(parents=True, exist_ok=True)

        kwargs: dict = {
            "text": text,
            "voice_id": voice_id,
            "model_id": self._model_id,
            "output_format": self._output_format,
            "voice_settings": VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
            ),
        }
        if previous_text:
            kwargs["previous_text"] = previous_text
        if next_text:
            kwargs["next_text"] = next_text

        # ElevenLabs SDK is synchronous — run in thread pool
        from elevenlabs.core.api_error import ApiError

        try:
            audio_iter = await asyncio.to_thread(self._client.text_to_speech.convert, **kwargs)

            # The SDK returns a generator of bytes chunks
            with open(output_path, "wb") as f:
                for chunk in audio_iter:
                    if isinstance(chunk, bytes):
                        f.write(chunk)
        except ApiError as e:
            if isinstance(e.body, dict) and e.body.get("detail", {}).get("status") == "quota_exceeded":
                detail = e.body["detail"]
                raise QuotaExceededError(detail.get("message", "ElevenLabs quota exceeded")) from e
            raise

        file_size = output_path.stat().st_size
        logger.info(
            "TTS generated: %s (%d bytes, voice=%s)",
            output_path.name,
            file_size,
            voice_id,
        )
        return output_path

    async def generate_slide_narrations(
        self,
        slides: list[dict],
        voice_id: str,
        output_dir: Path,
        *,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.4,
        resume: bool = False,
    ) -> list[Path]:
        """Generate narration audio for each slide sequentially.

        Uses previous_text/next_text for cross-slide continuity.

        Args:
            slides: List of dicts with at least a 'narration' key.
            voice_id: ElevenLabs voice ID.
            output_dir: Directory to write audio files.

        Returns:
            List of Paths to generated audio files (one per slide).
        """
        audio_paths: list[Path] = []

        for i, slide in enumerate(slides):
            narration = slide.get("narration", "")
            if not narration.strip():
                logger.warning("Slide %d has empty narration, skipping TTS", i)
                continue

            output_path = output_dir / f"slide_{i:02d}.mp3"

            if resume and output_path.exists() and output_path.stat().st_size > 1024:
                logger.info("Resume: skipping TTS for slide %d — %s already exists", i, output_path.name)
                audio_paths.append(output_path)
                continue

            await self.generate_speech(
                text=narration,
                voice_id=voice_id,
                output_path=output_path,
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
            )
            audio_paths.append(output_path)

        logger.info("Generated %d slide narrations in %s", len(audio_paths), output_dir)
        return audio_paths
