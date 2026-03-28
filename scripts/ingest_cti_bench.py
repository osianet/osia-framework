"""
OSIA CTI-Bench Ingestion

Streams all six task configurations of the AI4Sec/cti-bench HuggingFace dataset
(5,610 analyst-grade CTI evaluation scenarios), builds embeddable documents from
each task format, and upserts into a dedicated 'cti-bench' Qdrant collection.

Dataset: AI4Sec/cti-bench
  - License: CC BY-NC-SA 4.0 (non-commercial — internal intelligence use only)
  - Paper: arxiv 2406.07599

Task configurations and their intel value:
  cti-mcq      2,500 rows  Cybersecurity knowledge Q&A (MCQ format with answer)
  cti-rcm      1,000 rows  CVE description → CWE mapping (2024 CVEs)
  cti-rcm-2021 1,000 rows  CVE description → CWE mapping (2021 CVEs)
  cti-vsp      1,000 rows  Vulnerability description → CVSS severity prediction
  cti-ate         60 rows  Malware report → ATT&CK technique IDs (expert ground truth)
  cti-taa         50 rows  Threat intelligence report → threat actor attribution

The cti-ate and cti-taa tasks are highest intel value: real malware reports and
threat actor attribution scenarios with expert-labelled ground truth.

Usage:
  uv run python scripts/ingest_cti_bench.py
  uv run python scripts/ingest_cti_bench.py --tasks cti-ate cti-taa
  uv run python scripts/ingest_cti_bench.py --dry-run
  uv run python scripts/ingest_cti_bench.py --resume

Options:
  --tasks               Task configs to ingest (default: all six)
  --resume              Resume each task from its Redis checkpoint
  --dry-run             Parse but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 32)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)

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
logger = logging.getLogger("osia.cti_bench_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "cti-bench"
EMBEDDING_DIM = 384

HF_DATASET_ID = "AI4Sec/cti-bench"
SOURCE_LABEL = "CTI-Bench Analyst Evaluation Scenarios (AI4Sec/cti-bench)"

ALL_TASKS = ["cti-mcq", "cti-rcm", "cti-rcm-2021", "cti-vsp", "cti-ate", "cti-taa"]

# All tasks use the test split
TASK_SPLIT = "test"

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY_PREFIX = "osia:cti_bench:checkpoint:"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Per-task document builders
# ---------------------------------------------------------------------------


def _safe(val, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "") else default


def _first(*keys, row: dict, default: str = "") -> str:
    """Return the first non-empty value found among the given keys in row."""
    for k in keys:
        v = _safe(row.get(k))
        if v:
            return v
    return default


def _parse_list(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        import ast

        s = raw.strip()
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except (ValueError, SyntaxError):
                pass
        return [s] if s else []
    return []


def build_mcq_doc(row: dict) -> tuple[str, dict]:
    """cti-mcq: multiple-choice cybersecurity knowledge."""
    question = _first("question", "Question", "query", row=row)
    # Options may be separate fields (A/B/C/D) or a single options field
    opts: list[str] = []
    for opt_key in ("A", "B", "C", "D", "option_a", "option_b", "option_c", "option_d"):
        v = _safe(row.get(opt_key))
        if v:
            opts.append(v)
    if not opts:
        raw_opts = row.get("options") or row.get("choices") or []
        opts = _parse_list(raw_opts)

    answer = _first("answer", "Answer", "correct_answer", "label", row=row)

    parts = ["Cybersecurity Knowledge Q&A"]
    if question:
        parts.append(f"Question: {question}")
    if opts:
        for i, opt in enumerate(opts):
            parts.append(f"  {chr(65 + i)}) {opt}")
    if answer:
        parts.append(f"Answer: {answer}")

    extra: dict = {"task": "cti-mcq"}
    if question:
        extra["question"] = question[:200]
    if answer:
        extra["answer"] = answer

    return "\n\n".join(parts), extra


def build_rcm_doc(row: dict, task: str) -> tuple[str, dict]:
    """cti-rcm / cti-rcm-2021: CVE description → CWE mapping."""
    cve_id = _first("cve_id", "CVE-ID", "cve", "id", row=row)
    description = _first("description", "Description", "text", "vulnerability_description", row=row)
    cwe_id = _first("cwe_id", "CWE-ID", "cwe", "label", "answer", row=row)

    year = "2021" if "2021" in task else "2024"
    parts = [f"CVE → CWE Mapping ({year})"]
    if cve_id:
        parts.append(f"CVE: {cve_id}")
    if description:
        parts.append(f"Description: {description}")
    if cwe_id:
        parts.append(f"CWE Classification: {cwe_id}")

    extra: dict = {"task": task}
    if cve_id:
        extra["cve_id"] = cve_id
    if cwe_id:
        extra["cwe_id"] = cwe_id

    return "\n\n".join(parts), extra


def build_vsp_doc(row: dict) -> tuple[str, dict]:
    """cti-vsp: vulnerability severity prediction."""
    description = _first("description", "Description", "text", "vulnerability_description", row=row)
    severity = _first("severity", "Severity", "label", "answer", "cvss_severity", row=row)
    cve_id = _first("cve_id", "CVE-ID", "cve", row=row)

    parts = ["Vulnerability Severity Assessment"]
    if cve_id:
        parts.append(f"CVE: {cve_id}")
    if description:
        parts.append(f"Description: {description}")
    if severity:
        parts.append(f"Severity: {severity}")

    extra: dict = {"task": "cti-vsp"}
    if severity:
        extra["severity"] = severity
    if cve_id:
        extra["cve_id"] = cve_id

    return "\n\n".join(parts), extra


def build_ate_doc(row: dict) -> tuple[str, dict]:
    """cti-ate: malware report → ATT&CK technique extraction."""
    report = _first("text", "report", "content", "description", row=row)
    techniques_raw = row.get("techniques") or row.get("labels") or row.get("answer") or row.get("ttps") or []
    techniques = _parse_list(techniques_raw)
    if not techniques:
        # Sometimes stored as a comma-separated string
        t_str = _first("techniques", "labels", "answer", row=row)
        if t_str:
            techniques = [t.strip() for t in t_str.split(",") if t.strip()]

    parts = ["Malware Analysis Report — ATT&CK Technique Extraction"]
    if report:
        parts.append(report[:4000])  # cap long reports
    if techniques:
        parts.append(f"ATT&CK Techniques (Ground Truth): {', '.join(techniques)}")

    extra: dict = {"task": "cti-ate"}
    if techniques:
        extra["ttp_ids"] = techniques

    return "\n\n".join(parts), extra


def build_taa_doc(row: dict) -> tuple[str, dict]:
    """cti-taa: threat intelligence report → threat actor attribution."""
    report = _first("text", "report", "content", "description", row=row)
    actor = _first("actor", "threat_actor", "label", "answer", "attribution", row=row)

    parts = ["Threat Intelligence Report — Actor Attribution"]
    if report:
        parts.append(report[:4000])
    if actor:
        parts.append(f"Attributed Threat Actor: {actor}")

    extra: dict = {"task": "cti-taa"}
    if actor:
        extra["threat_actor"] = actor

    return "\n\n".join(parts), extra


TASK_BUILDERS = {
    "cti-mcq": lambda row: build_mcq_doc(row),
    "cti-rcm": lambda row: build_rcm_doc(row, "cti-rcm"),
    "cti-rcm-2021": lambda row: build_rcm_doc(row, "cti-rcm-2021"),
    "cti-vsp": lambda row: build_vsp_doc(row),
    "cti-ate": lambda row: build_ate_doc(row),
    "cti-taa": lambda row: build_taa_doc(row),
}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    task: str
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
            self.task,
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


class CtiBenchIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self, tasks: list[str], resume: bool) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()
            for task in tasks:
                stats = IngestStats(task=task)
                checkpoint = 0
                if resume:
                    checkpoint = await self._load_checkpoint(task)
                    if checkpoint:
                        logger.info("[%s] Resuming from record %d", task, checkpoint)
                logger.info("[%s] Starting ingestion.", task)
                await self._ingest_task(task, stats, checkpoint)
                await self._flush(stats)
                await self._save_checkpoint(task, stats.records_seen)
                stats.log_progress()
                logger.info("[%s] Done.", task)
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

    async def _ingest_task(self, task: str, stats: IngestStats, checkpoint: int) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        builder = TASK_BUILDERS.get(task)
        if not builder:
            logger.warning("No builder for task '%s' — skipping.", task)
            return

        try:
            ds = load_dataset(HF_DATASET_ID, name=task, split=TASK_SPLIT, token=HF_TOKEN or None)
        except Exception as exc:
            logger.error("[%s] Failed to load dataset: %s", task, exc)
            return

        rows = list(ds)
        logger.info("[%s] Loaded %d rows.", task, len(rows))

        for i, row in enumerate(rows):
            stats.records_seen += 1
            if stats.records_seen <= checkpoint:
                continue

            try:
                text, extra_meta = builder(row)
                if not text.strip():
                    stats.records_skipped += 1
                    continue

                doc_id = str(uuid.UUID(bytes=hashlib.sha256(f"ctibench:{task}:{i}".encode()).digest()[:16]))
                metadata: dict = {
                    "source": SOURCE_LABEL,
                    "dataset": HF_DATASET_ID,
                    "document_type": "cti_benchmark",
                    "provenance": "expert_annotation",
                    "ingest_date": TODAY,
                    **extra_meta,
                }
                stats.records_processed += 1
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(
                        id=doc_id, vector=[0.0] * EMBEDDING_DIM, payload={"text": text, **metadata}
                    )
                )
                if len(self._upsert_buffer) >= self.upsert_batch_size:
                    await self._flush(stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("[%s] Error processing row %d: %s", task, i, exc)

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

    async def _save_checkpoint(self, task: str, cursor: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(f"{CHECKPOINT_KEY_PREFIX}{task}", cursor)

    async def _load_checkpoint(self, task: str) -> int:
        if not self._redis:
            return 0
        val = await self._redis.get(f"{CHECKPOINT_KEY_PREFIX}{task}")
        return int(val) if val else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest AI4Sec/cti-bench into OSIA Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS, help="Task configs to ingest.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=32, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not HF_TOKEN:
        parser.error("HF_TOKEN not set.")
    logger.info("Starting CTI-Bench ingest | tasks=%s dry_run=%s", args.tasks, args.dry_run)
    if args.dry_run:
        logger.warning("DRY RUN — no writes.")
    asyncio.run(CtiBenchIngestor(args).run(tasks=args.tasks, resume=args.resume))


if __name__ == "__main__":
    main()
