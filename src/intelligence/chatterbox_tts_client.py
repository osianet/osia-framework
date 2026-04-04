"""
Chatterbox TTS client for OSIA weekly briefings.

Local GPU inference via Resemble AI's Chatterbox (500M standard model).
Drop-in replacement for TTSClient — same generate_slide_narrations interface,
no API key or quota required.

Requirements:
    pip install chatterbox-tts
    CUDA-capable GPU strongly recommended (runs on CPU but is slow)

Voice cloning: supply a 10–30s reference clip via voice_ref_path in each
desk's briefing YAML block. Generate reference clips from existing ElevenLabs
output and commit them to config/voice_refs/.
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("osia.chatterbox_tts")


class ChatterboxTTSClient:
    """Async wrapper around Chatterbox standard TTS for briefing narration."""

    def __init__(
        self,
        exaggeration: float = 0.6,
        cfg_weight: float = 0.4,
    ) -> None:
        import torch
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Chatterbox TTS model on %s", device)
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self._exaggeration = exaggeration
        self._cfg_weight = cfg_weight
        logger.info("Chatterbox TTS ready (sr=%d Hz, device=%s)", self._model.sr, device)

    async def generate_speech(
        self,
        text: str,
        voice_ref_path: str | Path | None,
        output_path: Path,
        **_kwargs,
    ) -> Path:
        """Generate speech audio from text and write to output_path as WAV.

        Args:
            text: The narration script to speak.
            voice_ref_path: Path to a WAV/MP3 reference clip for zero-shot
                            voice cloning. None uses the model default voice.
            output_path: Destination file (extension forced to .wav).
        """
        import torchaudio as ta

        output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ref = str(voice_ref_path) if voice_ref_path else None

        wav = await asyncio.to_thread(
            self._model.generate,
            text,
            audio_prompt_path=ref,
            exaggeration=self._exaggeration,
            cfg_weight=self._cfg_weight,
        )

        await asyncio.to_thread(ta.save, str(output_path), wav, self._model.sr)

        file_size = output_path.stat().st_size
        logger.info(
            "Chatterbox TTS: %s (%d bytes, ref=%s)",
            output_path.name,
            file_size,
            Path(ref).name if ref else "default",
        )
        return output_path

    async def generate_slide_narrations(
        self,
        slides: list[dict],
        voice_ref_path: str | Path | None,
        output_dir: Path,
        *,
        resume: bool = False,
        **_kwargs,
    ) -> list[Path]:
        """Generate narration audio for each slide sequentially.

        Args:
            slides: List of dicts with at least a 'narration' key.
            voice_ref_path: Path to reference audio for zero-shot voice cloning.
            output_dir: Directory to write audio files.
            resume: Skip slides whose WAV already exists.

        Returns:
            List of Paths to generated WAV files (one per slide).
        """
        audio_paths: list[Path] = []

        for i, slide in enumerate(slides):
            narration = slide.get("narration", "")
            if not narration.strip():
                logger.warning("Slide %d has empty narration, skipping TTS", i)
                continue

            output_path = output_dir / f"slide_{i:02d}.wav"

            if resume and output_path.exists() and output_path.stat().st_size > 1024:
                logger.info("Resume: skipping TTS for slide %d — already exists", i)
                audio_paths.append(output_path)
                continue

            await self.generate_speech(
                text=narration,
                voice_ref_path=voice_ref_path,
                output_path=output_path,
            )
            audio_paths.append(output_path)

        logger.info("Generated %d slide narrations in %s", len(audio_paths), output_dir)
        return audio_paths
