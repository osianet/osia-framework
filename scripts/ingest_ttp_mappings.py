"""
OSIA TTP Mappings Ingestion

Streams the tumeteor/Security-TTP-Mapping HuggingFace dataset (20,736 threat
report text snippets with expert-labelled MITRE ATT&CK technique IDs), embeds
via the HF Inference API, and upserts into a dedicated 'ttp-mappings' Qdrant
collection.

The ATT&CK technique IDs (T1059, T1566.001, etc.) are stored as a metadata
array on each point, enabling Qdrant payload filtering queries like:
  "all documents containing T1566" or cross-referencing with mitre-attack collection.

Dataset: tumeteor/Security-TTP-Mapping
  - 14,900 train / 2,630 validation / 3,170 test rows
  - Sources: MITRE CTID TRAM dataset, expert-annotated threat reports, derived procedures
  - Fields: text1 (report snippet), labels (list of ATT&CK technique IDs), split

Usage:
  uv run python scripts/ingest_ttp_mappings.py
  uv run python scripts/ingest_ttp_mappings.py --splits train
  uv run python scripts/ingest_ttp_mappings.py --limit 1000 --dry-run
  uv run python scripts/ingest_ttp_mappings.py --resume

Options:
  --splits              Space-separated splits (default: train validation test)
  --limit N             Stop after N records per split (0 = no limit)
  --resume              Resume each split from its Redis checkpoint
  --dry-run             Parse but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars to process (default: 40)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
"""

import argparse
import asyncio
import hashlib
import logging
import os
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
logger = logging.getLogger("osia.ttp_mappings_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "ttp-mappings"
EMBEDDING_DIM = 384

HF_DATASET_ID = "tumeteor/Security-TTP-Mapping"
SOURCE_LABEL = "Security TTP Mapping Corpus (tumeteor/Security-TTP-Mapping)"
ALL_SPLITS = ["train", "validation", "test"]

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY_PREFIX = "osia:ttp_mappings:checkpoint:"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def _safe(val, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "") else default


def _parse_labels(raw) -> list[str]:
    """
    Normalise the labels field to a list of ATT&CK technique ID strings.
    The dataset may return them as a list, a stringified list, or a single string.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    # Stringified Python list: "['T1059', 'T1082']"
    if s.startswith("["):
        import ast

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [s] if s else []


def build_document_text(row: dict, ttp_ids: list[str]) -> str:
    """
    Build an embeddable document: the threat report snippet followed by the
    labelled ATT&CK techniques, so retrieval captures both the text context
    and the technique attribution in a single embedding.
    """
    text = _safe(row.get("text1") or row.get("text") or row.get("sentence") or "")
    parts: list[str] = []
    if text:
        parts.append(text)
    if ttp_ids:
        parts.append(f"ATT&CK Techniques: {', '.join(ttp_ids)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    split: str
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    points_upserted: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "[%s] seen=%d processed=%d skipped=%d upserted=%d errors=%d elapsed=%s",
            self.split,
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.points_upserted,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------


class TtpMappingsIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self, splits: list[str], limit: int, resume: bool) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()
            for split in splits:
                stats = IngestStats(split=split)
                checkpoint = 0
                if resume:
                    checkpoint = await self._load_checkpoint(split)
                    if checkpoint:
                        logger.info("[%s] Resuming from record %d", split, checkpoint)
                logger.info("[%s] Starting ingestion (limit=%s)", split, limit or "none")
                await self._ingest_split(split, stats, limit, checkpoint)
                await self._flush(stats)
                await self._save_checkpoint(split, stats.records_seen)
                stats.log_progress()
                logger.info("[%s] Done.", split)
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    async def _ensure_collection(self) -> None:
        if self.dry_run:
            return
        exists = await self._qdrant.collection_exists(COLLECTION_NAME)
        if not exists:
            await self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(size=EMBEDDING_DIM, distance=qdrant_models.Distance.COSINE),
                optimizers_config=qdrant_models.OptimizersConfigDiff(indexing_threshold=1000),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info("Collection '%s' ready (%d points).", COLLECTION_NAME, info.points_count or 0)

    async def _ingest_split(self, split: str, stats: IngestStats, limit: int, checkpoint: int) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(HF_DATASET_ID, split=split, streaming=True, token=HF_TOKEN or None)

        async for row in self._stream_rows(ds, stats, limit, checkpoint):
            # Flexible text field — dataset uses 'text1' but guard against renames
            raw_text = _safe(row.get("text1") or row.get("text") or row.get("sentence") or "")
            if len(raw_text) < self.min_text_len:
                stats.records_skipped += 1
                continue

            ttp_ids = _parse_labels(row.get("labels") or row.get("label") or [])
            text = build_document_text(row, ttp_ids)

            doc_id = str(uuid.UUID(bytes=hashlib.sha256(f"ttp:{split}:{raw_text[:128]}".encode()).digest()[:16]))

            metadata: dict = {
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "split": split,
                "document_type": "ttp_mapping",
                "provenance": "expert_annotation",
                "ingest_date": TODAY,
            }
            if ttp_ids:
                metadata["ttp_ids"] = ttp_ids

            try:
                stats.records_processed += 1
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(
                        id=doc_id, vector=[0.0] * EMBEDDING_DIM, payload={"text": text, **metadata}
                    )
                )
                if len(self._upsert_buffer) >= self.upsert_batch_size:
                    await self._flush(stats)
                if stats.records_processed % 1000 == 0:
                    stats.log_progress()
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing record: %s", exc)

    async def _flush(self, stats: IngestStats | None = None) -> None:
        if not self._upsert_buffer:
            return
        points = list(self._upsert_buffer)
        self._upsert_buffer.clear()
        vectors = await self._embed_all([p.payload["text"] for p in points])
        for point, vector in zip(points, vectors, strict=True):
            point.vector = vector
        if not self.dry_run:
            await self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        if stats:
            stats.points_upserted += len(points)

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.embed_batch_size] for i in range(0, len(texts), self.embed_batch_size)]
        results: list[list[float]] = []
        for group_start in range(0, len(batches), self.embed_concurrency):
            group = batches[group_start : group_start + self.embed_concurrency]
            group_results = await asyncio.gather(*[self._embed_batch(b) for b in group])
            for vecs in group_results:
                results.extend(vecs)
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
                            await asyncio.sleep(30 * (attempt + 1))
                            continue
                        resp.raise_for_status()
                        result = resp.json()
                        if isinstance(result, list) and result and isinstance(result[0], list):
                            return result
                        if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                            return [result]
                        break
                except Exception as exc:
                    logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(5 * (attempt + 1))
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    async def _stream_rows(self, dataset, stats: IngestStats, limit: int, checkpoint: int):
        loop = asyncio.get_event_loop()
        skipped_ff = 0

        def _iter():
            yield from dataset

        it = _iter()
        _sentinel = object()
        while True:
            row = await loop.run_in_executor(None, next, it, _sentinel)
            if row is _sentinel:
                break
            stats.records_seen += 1
            if stats.records_seen <= checkpoint:
                skipped_ff += 1
                if skipped_ff % 1000 == 0:
                    logger.info("[%s] Fast-forwarding: %d/%d", stats.split, skipped_ff, checkpoint)
                continue
            yield row
            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("[%s] Reached --limit %d.", stats.split, limit)
                break

    async def _save_checkpoint(self, split: str, cursor: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(f"{CHECKPOINT_KEY_PREFIX}{split}", cursor)

    async def _load_checkpoint(self, split: str) -> int:
        if not self._redis:
            return 0
        val = await self._redis.get(f"{CHECKPOINT_KEY_PREFIX}{split}")
        return int(val) if val else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest tumeteor/Security-TTP-Mapping into OSIA Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=40, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not HF_TOKEN:
        parser.error("HF_TOKEN not set.")
    logger.info(
        "Starting TTP mappings ingest | splits=%s limit=%s dry_run=%s", args.splits, args.limit or "none", args.dry_run
    )
    if args.dry_run:
        logger.warning("DRY RUN — no writes.")
    asyncio.run(TtpMappingsIngestor(args).run(splits=args.splits, limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
