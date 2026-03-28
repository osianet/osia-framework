"""
OSIA CTI Reports Ingestion

Streams all splits of the mrmoor/cyber-threat-intelligence-splited HuggingFace
dataset (9,732 NER-annotated CTI report texts), extracts structured entity
metadata (malware families, threat actors, IOCs — IPs, domains, hashes, filepaths),
embeds via the HF Inference API, and upserts into a dedicated 'cti-reports'
Qdrant collection.

Dataset: mrmoor/cyber-threat-intelligence-splited
  - 6,810 train / 1,460 validation / 1,460 test rows
  - Publicly disclosed CTI text with expert NER annotations
  - Fields: text, entities (list of {start, end, type, text}), relations

Entity types in the dataset:
  Semantic: malware, threat-actor, attack-pattern, identity, location, campaign
  IOC: IPV4, DOMAIN, URL, SHA1, SHA256, MD5, FILEPATH, EMAIL, CVE
  Context: TIME, SOFTWARE, TOOL, VULNERABILITY

Usage:
  uv run python scripts/ingest_cti_reports.py
  uv run python scripts/ingest_cti_reports.py --splits train
  uv run python scripts/ingest_cti_reports.py --limit 500 --dry-run
  uv run python scripts/ingest_cti_reports.py --resume

Options:
  --splits              Space-separated splits (default: train validation test)
  --limit N             Stop after N records per split (0 = no limit)
  --resume              Resume each split from its Redis checkpoint
  --dry-run             Parse but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars to process (default: 60)

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
logger = logging.getLogger("osia.cti_reports_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "cti-reports"
EMBEDDING_DIM = 384

HF_DATASET_ID = "mrmoor/cyber-threat-intelligence-splited"
SOURCE_LABEL = "CTI Report Corpus (mrmoor/cyber-threat-intelligence-splited)"
ALL_SPLITS = ["train", "validation", "test"]

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY_PREFIX = "osia:cti_reports:checkpoint:"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Entity types that are meaningful semantic categories (vs IOC types)
SEMANTIC_ENTITY_TYPES = {"malware", "threat-actor", "attack-pattern", "identity", "campaign", "vulnerability", "tool"}
IOC_ENTITY_TYPES = {"IPV4", "DOMAIN", "URL", "SHA1", "SHA256", "MD5", "FILEPATH", "EMAIL", "CVE"}


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _safe(val, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "") else default


def extract_entities(entities_field) -> dict[str, list[str]]:
    """
    Parse the entities field (list of dicts or already-parsed objects) into
    per-type deduplicated lists for metadata and document augmentation.
    """
    result: dict[str, list[str]] = {}
    if not entities_field:
        return result

    raw = entities_field
    # Dataset may return a dict-of-lists (Datasets Arrow format) or list-of-dicts
    if isinstance(raw, dict):
        # Arrow-style: {"start": [...], "end": [...], "type": [...], "text": [...]}
        types = raw.get("type", []) or []
        texts = raw.get("text", []) or []
        for etype, etext in zip(types, texts, strict=False):
            if etype and etext:
                key = str(etype).lower()
                val = str(etext).strip()
                if val:
                    result.setdefault(key, [])
                    if val not in result[key]:
                        result[key].append(val)
    elif isinstance(raw, list):
        for ent in raw:
            if isinstance(ent, dict):
                etype = _safe(ent.get("type")).lower()
                etext = _safe(ent.get("text"))
                if etype and etext:
                    result.setdefault(etype, [])
                    if etext not in result[etype]:
                        result[etype].append(etext)

    return result


def build_document_text(row: dict, entities: dict[str, list[str]]) -> str:
    """
    Build an embeddable document: the raw CTI text augmented with an entity
    summary block so semantic search captures both the narrative and the IOCs.
    """
    text = _safe(row.get("text"))
    parts: list[str] = []

    if text:
        parts.append(text)

    # Entity summary appended after the main text
    summary_lines: list[str] = []
    for etype in ("malware", "threat-actor", "attack-pattern", "campaign", "identity"):
        vals = entities.get(etype, [])
        if vals:
            label = etype.replace("-", " ").title()
            summary_lines.append(f"{label}: {', '.join(vals[:10])}")

    # IOC types
    for etype in ("IPV4", "DOMAIN", "SHA1", "SHA256", "MD5", "CVE"):
        vals = entities.get(etype.lower(), entities.get(etype, []))
        if vals:
            summary_lines.append(f"{etype}: {', '.join(vals[:10])}")

    if summary_lines:
        parts.append("Entities:\n" + "\n".join(summary_lines))

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


class CtiReportsIngestor:
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
                logger.info("[%s] Starting ingestion of %s (limit=%s)", split, HF_DATASET_ID, limit or "none")
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
            raw_text = _safe(row.get("text"))
            if len(raw_text) < self.min_text_len:
                stats.records_skipped += 1
                continue

            entities = extract_entities(row.get("entities"))
            text = build_document_text(row, entities)

            doc_id = str(uuid.UUID(bytes=hashlib.sha256(f"cti:{split}:{raw_text[:128]}".encode()).digest()[:16]))

            # Build per-type metadata arrays for payload filtering
            metadata: dict = {
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "split": split,
                "document_type": "cti_report",
                "provenance": "expert_annotation",
                "ingest_date": TODAY,
            }
            for etype in ("malware", "threat-actor", "attack-pattern", "campaign", "identity"):
                vals = entities.get(etype, [])
                if vals:
                    metadata[etype.replace("-", "_")] = vals[:20]
            for ioc in ("IPV4", "DOMAIN", "SHA1", "SHA256", "MD5", "CVE", "FILEPATH"):
                vals = entities.get(ioc.lower(), entities.get(ioc, []))
                if vals:
                    metadata[ioc.lower()] = vals[:20]

            try:
                stats.records_processed += 1
                payload = {"text": text, **metadata}
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(id=doc_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
                )
                if len(self._upsert_buffer) >= self.upsert_batch_size:
                    await self._flush(stats)
                if stats.records_processed % 500 == 0:
                    stats.log_progress()
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing record: %s", exc)

    async def _flush(self, stats: IngestStats | None = None) -> None:
        if not self._upsert_buffer:
            return
        points = list(self._upsert_buffer)
        self._upsert_buffer.clear()
        texts = [p.payload["text"] for p in points]
        vectors = await self._embed_all(texts)
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
        _exhausted = False
        while True:
            row = await loop.run_in_executor(None, next, it, _sentinel)
            if row is _sentinel:
                _exhausted = True
                break
            stats.records_seen += 1
            if stats.records_seen <= checkpoint:
                skipped_ff += 1
                if skipped_ff % 500 == 0:
                    logger.info("[%s] Fast-forwarding: %d/%d", stats.split, skipped_ff, checkpoint)
                continue
            yield row
            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("[%s] Reached --limit %d.", stats.split, limit)
                break

        if _exhausted:
            logger.info(
                "[%s] Dataset exhausted at %d total records — ingestion complete.",
                stats.split,
                stats.records_seen,
            )

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
        description="Ingest mrmoor/cyber-threat-intelligence-splited into OSIA Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=60, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not HF_TOKEN:
        parser.error("HF_TOKEN not set.")
    logger.info(
        "Starting CTI reports ingest | splits=%s limit=%s dry_run=%s", args.splits, args.limit or "none", args.dry_run
    )
    if args.dry_run:
        logger.warning("DRY RUN — no writes.")
    asyncio.run(CtiReportsIngestor(args).run(splits=args.splits, limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
