"""
OSIA CVE Database Ingestion

Streams the stasvinokur/cve-and-cwe-dataset-1999-2025 HuggingFace dataset
(280,694 NVD CVE records from 1999 through May 2025), constructs rich
vulnerability documents from structured fields, embeds via the HF Inference
API, and upserts into a dedicated 'cve-database' Qdrant collection.

Dataset: stasvinokur/cve-and-cwe-dataset-1999-2025
  - 280,694 rows, ~103 MB CSV / 38 MB Parquet
  - License: CC0-1.0 (public domain)
  - Fields: CVE-ID, CVSS-V4, CVSS-V3, CVSS-V2, SEVERITY, DESCRIPTION, CWE-ID

Usage:
  uv run python scripts/ingest_cve_database.py
  uv run python scripts/ingest_cve_database.py --limit 5000 --dry-run
  uv run python scripts/ingest_cve_database.py --severity CRITICAL HIGH
  uv run python scripts/ingest_cve_database.py --resume

Options:
  --limit N             Stop after N records (0 = no limit)
  --severity            Only ingest CVEs with these severity levels (space-separated).
                        Choices: CRITICAL HIGH MEDIUM LOW NONE. Default: all.
  --resume              Resume from Redis checkpoint
  --dry-run             Parse but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-desc-len        Minimum description length to process (default: 20)

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
logger = logging.getLogger("osia.cve_database_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "cve-database"
EMBEDDING_DIM = 384

HF_DATASET_ID = "stasvinokur/cve-and-cwe-dataset-1999-2025"
SOURCE_LABEL = "NVD CVE Database 1999–2025 (stasvinokur/cve-and-cwe-dataset-1999-2025)"

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY = "osia:cve_database:checkpoint"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def _safe(val, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "NaN", "") else default


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def build_document_text(row: dict) -> str:
    """
    Construct an embeddable vulnerability document combining the CVE ID,
    severity/score metadata, and description. Structured so semantic search
    captures both the classification context and the vulnerability narrative.
    """
    cve_id = _safe(row.get("CVE-ID"))
    description = _safe(row.get("DESCRIPTION"))
    severity = _safe(row.get("SEVERITY"))
    cwe_id = _safe(row.get("CWE-ID"))
    cvss_v3 = _safe_float(row.get("CVSS-V3"))
    cvss_v4 = _safe_float(row.get("CVSS-V4"))

    parts: list[str] = []

    header = f"Vulnerability: {cve_id}" if cve_id else "Vulnerability"
    if severity:
        header += f" ({severity})"
    parts.append(header)

    meta_lines: list[str] = []
    if cwe_id:
        meta_lines.append(f"Weakness: {cwe_id}")
    score_parts = []
    if cvss_v4 is not None:
        score_parts.append(f"CVSSv4: {cvss_v4:.1f}")
    if cvss_v3 is not None:
        score_parts.append(f"CVSSv3: {cvss_v3:.1f}")
    if score_parts:
        meta_lines.append(" | ".join(score_parts))
    if meta_lines:
        parts.append("\n".join(meta_lines))

    if description:
        parts.append(description)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
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
            "seen=%d processed=%d skipped=%d upserted=%d errors=%d elapsed=%s",
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


class CveDatabaseIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.severity_filter: set[str] | None = {s.upper() for s in args.severity} if args.severity else None
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_desc_len: int = args.min_desc_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

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
                "Starting CVE ingest (limit=%s severity_filter=%s resume_from=%d)",
                limit or "none",
                self.severity_filter or "all",
                checkpoint,
            )
            stats = IngestStats()
            await self._ingest(stats, limit, checkpoint)
            await self._flush(stats)
            await self._save_checkpoint(stats.records_seen)
            stats.log_progress()
            logger.info("CVE ingestion complete.")
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

    async def _ingest(self, stats: IngestStats, limit: int, checkpoint: int) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(HF_DATASET_ID, split="train", streaming=True, token=HF_TOKEN or None)

        async for row in self._stream_rows(ds, stats, limit, checkpoint):
            severity = _safe(row.get("SEVERITY")).upper()
            if self.severity_filter and severity not in self.severity_filter:
                stats.records_skipped += 1
                continue

            description = _safe(row.get("DESCRIPTION"))
            if len(description) < self.min_desc_len:
                stats.records_skipped += 1
                continue

            cve_id = _safe(row.get("CVE-ID"))
            if not cve_id:
                stats.records_skipped += 1
                continue

            text = build_document_text(row)
            doc_id = str(uuid.UUID(bytes=hashlib.sha256(f"cve:{cve_id}".encode()).digest()[:16]))

            metadata: dict = {
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "document_type": "cve",
                "provenance": "nvd",
                "ingest_date": TODAY,
                "cve_id": cve_id,
            }
            if severity:
                metadata["severity"] = severity
                metadata["severity_score"] = SEVERITY_ORDER.get(severity, -1)
            cwe_id = _safe(row.get("CWE-ID"))
            if cwe_id:
                metadata["cwe_id"] = cwe_id
            for cvss_field, dest_key in [("CVSS-V4", "cvss_v4"), ("CVSS-V3", "cvss_v3"), ("CVSS-V2", "cvss_v2")]:
                score = _safe_float(row.get(cvss_field))
                if score is not None:
                    metadata[dest_key] = score

            try:
                stats.records_processed += 1
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(
                        id=doc_id, vector=[0.0] * EMBEDDING_DIM, payload={"text": text, **metadata}
                    )
                )
                if len(self._upsert_buffer) >= self.upsert_batch_size:
                    await self._flush(stats)
                if stats.records_processed % 2000 == 0:
                    stats.log_progress()
                    await self._save_checkpoint(stats.records_seen)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing %s: %s", cve_id, exc)

        await self._save_checkpoint(stats.records_seen)

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
                if skipped_ff % 5000 == 0:
                    logger.info("Fast-forwarding checkpoint: %d/%d", skipped_ff, checkpoint)
                continue
            yield row
            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("Reached --limit %d.", limit)
                break

    async def _save_checkpoint(self, cursor: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(CHECKPOINT_KEY, cursor)

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
        description="Ingest stasvinokur/cve-and-cwe-dataset-1999-2025 into OSIA Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--severity",
        nargs="+",
        metavar="LEVEL",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"],
        help="Only ingest CVEs with these severity levels.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-desc-len", type=int, default=20, dest="min_desc_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not HF_TOKEN:
        parser.error("HF_TOKEN not set.")
    logger.info(
        "Starting CVE database ingest | limit=%s severity=%s dry_run=%s",
        args.limit or "none",
        args.severity or "all",
        args.dry_run,
    )
    if args.dry_run:
        logger.warning("DRY RUN — no writes.")
    asyncio.run(CveDatabaseIngestor(args).run(limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
