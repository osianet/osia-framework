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

Voice conditioning is cached as .pt files alongside each reference clip so
that prepare_conditionals() only runs once per unique voice — subsequent
runs skip re-encoding and load the cached tensors directly.
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
        self._device = device
        logger.info("Chatterbox TTS ready (sr=%d Hz, device=%s)", self._model.sr, device)

    def _load_voice(self, voice_ref_path: str | Path | None) -> None:
        """Set voice conditionals on the model, using a .pt cache when available.

        If voice_ref_path is None, the model's built-in default conditionals
        (loaded from conds.pt at init time) are left untouched.

        Cache files are written as <ref_stem>.conds.pt next to the source clip
        so they persist across runs and are only rebuilt if the source changes.
        """
        if voice_ref_path is None:
            return

        from chatterbox.tts import Conditionals

        ref = Path(voice_ref_path)
        cache_path = ref.parent / f"{ref.stem}.conds.pt"

        if cache_path.exists() and cache_path.stat().st_mtime >= ref.stat().st_mtime:
            logger.info("Loading cached voice profile: %s", cache_path.name)
            self._model.conds = Conditionals.load(cache_path, map_location=self._device).to(self._device)
        else:
            logger.info("Encoding voice profile from %s (will cache)", ref.name)
            self._model.prepare_conditionals(str(ref), exaggeration=self._exaggeration)
            self._model.conds.save(cache_path)
            logger.info("Voice profile cached: %s", cache_path.name)

    # Chatterbox has a hardcoded max_new_tokens=1000 which caps output at ~40s
    # of speech. Any narration longer than that is silently truncated mid-sentence.
    # We split on paragraph breaks and generate each chunk separately.
    _CHUNK_CHAR_LIMIT = 600  # conservative — ~30s of speech per chunk

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split narration into paragraph-sized chunks that fit within the token limit."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if not current:
                current = para
            elif len(current) + len(para) + 2 <= self._CHUNK_CHAR_LIMIT:
                current += "\n\n" + para
            else:
                chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks or [text]

    async def generate_speech(
        self,
        text: str,
        voice_ref_path: str | Path | None,
        output_path: Path,
        **_kwargs,
    ) -> Path:
        """Generate speech audio from text and write to output_path as WAV.

        Long texts are automatically split into paragraph chunks to avoid
        Chatterbox's hardcoded 1000-token (~40s) generation limit.

        Args:
            text: The narration script to speak.
            voice_ref_path: Path to a WAV/MP3 reference clip for zero-shot
                            voice cloning. None uses the model default voice.
            output_path: Destination file (extension forced to .wav).
        """
        import torch
        import torchaudio as ta

        output_path = output_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load (or restore from cache) the voice conditionals for this ref.
        await asyncio.to_thread(self._load_voice, voice_ref_path)

        chunks = self._split_into_chunks(text)
        ref_label = Path(voice_ref_path).name if voice_ref_path else "default"

        if len(chunks) == 1:
            wav = await asyncio.to_thread(
                self._model.generate,
                chunks[0],
                audio_prompt_path=None,
                exaggeration=self._exaggeration,
                cfg_weight=self._cfg_weight,
            )
        else:
            logger.info(
                "Long narration split into %d chunks (voice=%s)", len(chunks), ref_label
            )
            wavs = []
            for idx, chunk in enumerate(chunks):
                w = await asyncio.to_thread(
                    self._model.generate,
                    chunk,
                    audio_prompt_path=None,
                    exaggeration=self._exaggeration,
                    cfg_weight=self._cfg_weight,
                )
                wavs.append(w)
                logger.debug("Chunk %d/%d done", idx + 1, len(chunks))
            wav = torch.cat(wavs, dim=-1)

        await asyncio.to_thread(ta.save, str(output_path), wav, self._model.sr)

        file_size = output_path.stat().st_size
        logger.info(
            "Chatterbox TTS: %s (%d bytes, voice=%s, chunks=%d)",
            output_path.name,
            file_size,
            ref_label,
            len(chunks),
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
                            Used as a fallback when a slide has no 'voice_ref_path' key.
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

            # Per-slide voice ref takes priority over the caller-supplied default.
            slide_voice = slide.get("voice_ref_path") or voice_ref_path

            await self.generate_speech(
                text=narration,
                voice_ref_path=slide_voice,
                output_path=output_path,
            )
            audio_paths.append(output_path)

        logger.info("Generated %d slide narrations in %s", len(audio_paths), output_dir)
        return audio_paths
