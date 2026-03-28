"""
OSIA WikiLeaks Cables Ingestion

Streams the fn5/wikileaks-cables HuggingFace dataset (124,747 US diplomatic cables
released by WikiLeaks), constructs rich cable documents from structured fields,
chunks long bodies, embeds via the HF Inference API, and upserts into a dedicated
'wikileaks-cables' Qdrant collection.

Dataset: fn5/wikileaks-cables
  - 124,747 rows, ~1 GB Parquet
  - Full dump of US diplomatic cables (1966–2010)
  - Fields: id, datetime, cable_id, origin, classification, references, body,
            parse_status, parse_error_details, original_start_line

Usage:
  uv run python scripts/ingest_wikileaks_cables.py
  uv run python scripts/ingest_wikileaks_cables.py --limit 5000 --dry-run
  uv run python scripts/ingest_wikileaks_cables.py --resume
  uv run python scripts/ingest_wikileaks_cables.py --enqueue-classified
  uv run python scripts/ingest_wikileaks_cables.py --classification SECRET CONFIDENTIAL

Options:
  --limit N             Stop after N source records (0 = no limit)
  --classification      Only ingest cables with these classification levels (space-separated).
                        Choices: UNCLASSIFIED CONFIDENTIAL SECRET TOP SECRET
                        Default: all classifications
  --enqueue-classified  Push SECRET/CONFIDENTIAL cables to Geopolitical desk research queue
  --resume              Resume from last Redis checkpoint
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for the body to be processed (default: 80)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required — for dataset access + embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.wikileaks_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "wikileaks-cables"
EMBEDDING_DIM = 384

HF_DATASET_ID = "fn5/wikileaks-cables"
SOURCE_LABEL = "WikiLeaks US Diplomatic Cables (fn5/wikileaks-cables)"

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Redis keys
CHECKPOINT_KEY = "osia:wikileaks:checkpoint"
SEEN_CABLES_KEY = "osia:wikileaks:seen_cables"  # session-local dedup set
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Classifications that are meaningful research targets when --enqueue-classified is set
ENQUEUE_CLASSIFICATIONS = {"SECRET", "CONFIDENTIAL", "TOP SECRET"}

# Chunking — paragraph-aware, sentence-level fallback, 15% overlap
CHUNK_SIZE = 1500  # characters (~375 tokens for all-MiniLM-L6-v2)
CHUNK_OVERLAP = 225  # ~15%


# ---------------------------------------------------------------------------
# Chunking (shared pattern with ingest_epstein_files.py)
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Paragraph-aware text chunker with sentence-level fallback and overlap.
    Short texts are returned as a single chunk.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush(parts: list[str]) -> tuple[list[str], int]:
        if parts:
            chunks.append("\n\n".join(parts))
            tail: list[str] = []
            tail_len = 0
            for part in reversed(parts):
                if tail_len + len(part) <= overlap:
                    tail.insert(0, part)
                    tail_len += len(part)
                else:
                    break
            return tail, tail_len
        return [], 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current_len + len(sent) > chunk_size and current_parts:
                    current_parts, current_len = flush(current_parts)
                current_parts.append(sent)
                current_len += len(sent) + 1
            continue

        if current_len + para_len > chunk_size and current_parts:
            current_parts, current_len = flush(current_parts)

        current_parts.append(para)
        current_len += para_len + 2

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if len(c.strip()) >= 50]


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def _safe(val, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "NaT", "") else default


def build_header(row: dict) -> str:
    """Build the structured header prepended to every chunk of this cable."""
    cable_id = _safe(row.get("cable_id"))
    origin = _safe(row.get("origin"))
    classification = _safe(row.get("classification"))
    dt = _safe(row.get("datetime"))
    references = row.get("references") or []

    lines: list[str] = []
    if cable_id:
        lines.append(f"Cable: {cable_id}")
    if origin:
        lines.append(f"Origin: {origin}")
    if classification:
        lines.append(f"Classification: {classification}")
    if dt:
        # Normalise timestamp to date portion only
        date_str = dt[:10] if len(dt) >= 10 else dt
        lines.append(f"Date: {date_str}")
    if references:
        ref_list = list(references)[:8]  # cap at 8 to keep header compact
        lines.append(f"References: {', '.join(str(r) for r in ref_list)}")

    return "\n".join(lines)


def build_chunks(row: dict) -> list[str]:
    """
    Return a list of embeddable text chunks for this cable.
    Each chunk is prefixed with the cable header so every embedding captures
    the classification and provenance context alongside the body content.
    """
    header = build_header(row)
    body = _safe(row.get("body"))

    if not body:
        return [header] if header else []

    body_chunks = chunk_text(body)
    if not body_chunks:
        return [header] if header else []

    # Prepend header to every chunk so cross-chunk search is self-contained
    return [f"{header}\n\n{chunk}" for chunk in body_chunks]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    chunks_produced: int = 0
    points_upserted: int = 0
    cables_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "seen=%d processed=%d skipped=%d chunks=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.chunks_produced,
            self.points_upserted,
            self.cables_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class WikileaksCablesIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_classified: bool = args.enqueue_classified
        self.classification_filter: set[str] | None = (
            {c.upper() for c in args.classification} if args.classification else None
        )
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)

        self._session_seen_cables: set[str] = set()
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self, limit: int, resume: bool) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()

            checkpoint = 0
            if resume:
                checkpoint = await self._load_checkpoint()
                if checkpoint:
                    logger.info("Resuming from record %d", checkpoint)

            logger.info(
                "Starting ingestion of %s (limit=%s resume_from=%d classification_filter=%s)",
                HF_DATASET_ID,
                limit or "none",
                checkpoint,
                self.classification_filter or "all",
            )

            stats = IngestStats()
            await self._ingest(stats, limit, checkpoint)

            await self._flush_upsert_buffer(stats)
            await self._save_checkpoint(stats.records_seen)
            stats.log_progress()
            logger.info("Ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

    async def _ensure_collection(self) -> None:
        if self.dry_run:
            logger.info("[dry-run] Skipping collection creation.")
            return
        exists = await self._qdrant.collection_exists(COLLECTION_NAME)
        if not exists:
            await self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=1000,
                ),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info(
                "Collection '%s' ready (%d points).",
                COLLECTION_NAME,
                info.points_count or 0,
            )

    # ------------------------------------------------------------------
    # Dataset ingestion
    # ------------------------------------------------------------------

    async def _ingest(self, stats: IngestStats, limit: int, checkpoint: int) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(HF_DATASET_ID, split="train", streaming=True, token=HF_TOKEN or None)

        async for row in self._stream_rows(ds, stats, limit, checkpoint):
            # Apply classification filter if set
            classification = _safe(row.get("classification")).upper()
            if self.classification_filter and classification not in self.classification_filter:
                stats.records_skipped += 1
                continue

            # Skip cables that failed parsing
            if _safe(row.get("parse_status")).lower().startswith("error"):
                stats.records_skipped += 1
                continue

            body = _safe(row.get("body"))
            if len(body) < self.min_text_len:
                stats.records_skipped += 1
                continue

            cable_id = _safe(row.get("cable_id"))
            if not cable_id:
                # Fall back to the integer id field
                cable_id = f"cable_{_safe(row.get('id'))}"

            try:
                await self._process_record(cable_id, classification, row, stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing cable %s: %s", cable_id, exc)

        await self._save_checkpoint(stats.records_seen)

    async def _process_record(
        self,
        cable_id: str,
        classification: str,
        row: dict,
        stats: IngestStats,
    ) -> None:
        stats.records_processed += 1

        chunks = build_chunks(row)
        if not chunks:
            stats.records_skipped += 1
            return

        stats.chunks_produced += len(chunks)

        # Stable base ID from cable_id
        base_id = str(uuid.UUID(bytes=hashlib.sha256(f"wl:{cable_id}".encode()).digest()[:16]))

        # Shared metadata for all chunks of this cable
        dt_raw = _safe(row.get("datetime"))
        references = row.get("references") or []
        base_metadata: dict = {
            "source": SOURCE_LABEL,
            "dataset": HF_DATASET_ID,
            "document_type": "diplomatic_cable",
            "provenance": "wikileaks_disclosure",
            "ingest_date": TODAY,
            "cable_id": cable_id,
        }
        origin = _safe(row.get("origin"))
        if origin:
            base_metadata["origin"] = origin
        if classification:
            base_metadata["classification"] = classification
        if dt_raw:
            base_metadata["cable_date"] = dt_raw[:10] if len(dt_raw) >= 10 else dt_raw
        if references:
            base_metadata["references"] = [str(r) for r in list(references)[:20]]
        chunk_count = len(chunks)
        base_metadata["chunk_count"] = chunk_count

        for idx, chunk_text_val in enumerate(chunks):
            # For single-chunk cables use the base ID; multi-chunk get a suffix
            if chunk_count == 1:
                point_id = base_id
            else:
                suffix = f"{cable_id}:chunk{idx}"
                point_id = str(uuid.UUID(bytes=hashlib.sha256(suffix.encode()).digest()[:16]))

            payload = {
                "text": chunk_text_val,
                "chunk_index": idx,
                **base_metadata,
            }
            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload=payload,
                )
            )

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        # Optionally enqueue classified cables for Geopolitical desk research
        if self.enqueue_classified and classification in ENQUEUE_CLASSIFICATIONS:
            enqueued = await self._maybe_enqueue_cable(cable_id, classification, row)
            stats.cables_enqueued += enqueued

        if stats.records_processed % 500 == 0:
            stats.log_progress()

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _maybe_enqueue_cable(self, cable_id: str, classification: str, row: dict) -> int:
        """
        Enqueue a Geopolitical desk research job for a notable classified cable.
        Deduplicates by cable_id within the session and via Redis to avoid
        flooding the queue across runs.
        """
        if not self._redis or self.dry_run:
            return 0

        if cable_id in self._session_seen_cables:
            return 0

        redis_key = f"osia:wikileaks:enqueued:{cable_id}"
        already_queued = await self._redis.exists(redis_key)
        if already_queued:
            self._session_seen_cables.add(cable_id)
            return 0

        origin = _safe(row.get("origin"))
        topic = f"WikiLeaks cable {cable_id}"
        if origin:
            topic += f" ({origin})"

        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "geopolitical-and-security-desk",
                "priority": "low",
                "directives_lens": True,
                "triggered_by": "wikileaks_cables_ingest",
                "metadata": {
                    "cable_id": cable_id,
                    "classification": classification,
                    "origin": origin,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        # TTL 30 days — don't re-enqueue same cable next run
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)
        self._session_seen_cables.add(cable_id)

        logger.debug("Research job enqueued: %r → Geopolitical desk", topic)
        return 1

    # ------------------------------------------------------------------
    # Flush + embed
    # ------------------------------------------------------------------

    async def _flush_upsert_buffer(self, stats: IngestStats | None = None) -> None:
        if not self._upsert_buffer:
            return

        points = list(self._upsert_buffer)
        self._upsert_buffer.clear()
        texts = [p.payload["text"] for p in points]

        vectors = await self._embed_all(texts)

        for point, vector in zip(points, vectors, strict=True):
            point.vector = vector

        if not self.dry_run:
            await self._qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            logger.debug("Upserted %d points to '%s'.", len(points), COLLECTION_NAME)

        if stats is not None:
            stats.points_upserted += len(points)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.embed_batch_size] for i in range(0, len(texts), self.embed_batch_size)]
        results: list[list[float]] = []
        for group_start in range(0, len(batches), self.embed_concurrency):
            group = batches[group_start : group_start + self.embed_concurrency]
            group_results = await asyncio.gather(*[self._embed_batch(b) for b in group])
            for batch_vecs in group_results:
                results.extend(batch_vecs)
        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._embed_semaphore:
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=45.0) as http:
                        resp = await http.post(
                            HF_EMBEDDING_URL,
                            headers={"Authorization": f"Bearer {HF_TOKEN}"},
                            json={"inputs": texts, "options": {"wait_for_model": True}},
                        )
                        if resp.status_code == 429:
                            wait = 30 * (attempt + 1)
                            logger.warning("HF embedding 429 — waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        result = resp.json()
                        if isinstance(result, list) and result and isinstance(result[0], list):
                            return result
                        if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                            return [result]
                        logger.error("Unexpected HF embedding shape: %s", type(result))
                        break
                except Exception as exc:
                    logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(5 * (attempt + 1))
            logger.error("Embedding failed for batch of %d — using zero vectors.", len(texts))
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    async def _stream_rows(self, dataset, stats: IngestStats, limit: int, checkpoint: int):
        loop = asyncio.get_event_loop()
        skipped_ff = 0

        def _iter():
            yield from dataset

        it = _iter()

        _sentinel = object()
        _exhausted = False
        while True:
            row = await loop.run_in_executor(None, next, it, _sentinel)
            if row is _sentinel:
                _exhausted = True
                break

            stats.records_seen += 1

            if stats.records_seen <= checkpoint:
                skipped_ff += 1
                if skipped_ff % 2_000 == 0:
                    logger.info("Fast-forwarding checkpoint: %d/%d", skipped_ff, checkpoint)
                continue

            yield row

            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("Reached --limit %d — stopping.", limit)
                break

        if _exhausted:
            logger.info("Dataset exhausted at %d total records — ingestion complete.", stats.records_seen)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    async def _save_checkpoint(self, cursor: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(CHECKPOINT_KEY, cursor)
        logger.info("Checkpoint saved: %d records", cursor)

    async def _load_checkpoint(self) -> int:
        if not self._redis:
            return 0
        val = await self._redis.get(CHECKPOINT_KEY)
        return int(val) if val else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest fn5/wikileaks-cables dataset into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N source records (0=no limit)")
    p.add_argument(
        "--classification",
        nargs="+",
        metavar="LEVEL",
        help="Only ingest cables with these classifications (e.g. SECRET CONFIDENTIAL). Default: all.",
    )
    p.add_argument(
        "--enqueue-classified",
        action="store_true",
        dest="enqueue_classified",
        help="Enqueue SECRET/CONFIDENTIAL cables to Geopolitical desk research queue",
    )
    p.add_argument("--resume", action="store_true", help="Resume from Redis checkpoint")
    p.add_argument("--dry-run", action="store_true", help="Parse and chunk but skip Qdrant writes and Redis updates")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=80, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set in environment — required for dataset access and embeddings.")

    logger.info(
        "Starting WikiLeaks cables ingest | dataset=%s limit=%s classification=%s enqueue_classified=%s dry_run=%s",
        HF_DATASET_ID,
        args.limit or "none",
        args.classification or "all",
        args.enqueue_classified,
        args.dry_run,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant or Redis.")

    ingestor = WikileaksCablesIngestor(args)
    asyncio.run(ingestor.run(limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
