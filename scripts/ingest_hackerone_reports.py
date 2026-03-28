"""
OSIA HackerOne Disclosed Reports Ingestion

Streams all three splits of the Hacker0x01/hackerone_disclosed_reports HuggingFace
dataset (12,618 publicly disclosed vulnerability reports), constructs rich report
documents from structured fields, embeds via the HF Inference API, and upserts into
a dedicated 'hackerone-reports' Qdrant collection.

Dataset: Hacker0x01/hackerone_disclosed_reports
  - 10,094 train / 1,262 validation / 1,262 test rows
  - Publicly disclosed bug bounty reports from HackerOne
  - Fields: id, title, created_at, substate, vulnerability_information, reporter,
            team, has_bounty?, visibility, disclosed_at, weakness, structured_scope,
            original_report_id, vote_count

Usage:
  uv run python scripts/ingest_hackerone_reports.py
  uv run python scripts/ingest_hackerone_reports.py --splits train
  uv run python scripts/ingest_hackerone_reports.py --limit 500 --dry-run
  uv run python scripts/ingest_hackerone_reports.py --skip-no-content
  uv run python scripts/ingest_hackerone_reports.py --resume

Options:
  --splits              Space-separated splits to ingest: train validation test (default: all)
  --limit N             Stop after N source records per split (0 = no limit)
  --skip-no-content     Skip reports with visibility=no-content (no vuln text available)
  --resume              Resume each split from its last Redis checkpoint
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a record to be processed (default: 80)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required — for dataset access + embeddings)
  QDRANT_URL            Qdrant URL (default: http://localhost:6333)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
"""

import argparse
import ast
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
logger = logging.getLogger("osia.hackerone_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "hackerone-reports"
EMBEDDING_DIM = 384

HF_DATASET_ID = "Hacker0x01/hackerone_disclosed_reports"
SOURCE_LABEL = "HackerOne Disclosed Bug Bounty Reports"
ALL_SPLITS = ["train", "validation", "test"]

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Redis keys
CHECKPOINT_KEY_PREFIX = "osia:hackerone:checkpoint:"  # + split name

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Dict field parser
# ---------------------------------------------------------------------------


def _parse_dict_field(val) -> dict:
    """
    Dataset dict fields are serialised as Python repr strings with single quotes.
    Parse safely via ast.literal_eval; return {} on any failure.
    """
    if isinstance(val, dict):
        return val
    if not val or str(val).strip() in ("None", "nan", ""):
        return {}
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _safe(val, default: str = "") -> str:
    s = str(val).strip() if val is not None else ""
    return s if s not in ("None", "nan", "NaT", "") else default


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document_text(row: dict) -> str:
    """
    Construct a rich, embeddable document from a HackerOne report row.
    Combines structured metadata with the full vulnerability narrative so a
    single embedding captures both the classification and the technical content.
    """
    report_id = _safe(row.get("id"))
    title = _safe(row.get("title"))
    substate = _safe(row.get("substate"))
    vuln_info = _safe(row.get("vulnerability_information"))
    disclosed_at = _safe(row.get("disclosed_at"))
    vote_count = _safe(row.get("vote_count"))
    has_bounty = str(row.get("has_bounty?", "")).strip()

    reporter = _parse_dict_field(row.get("reporter"))
    team = _parse_dict_field(row.get("team"))
    weakness = _parse_dict_field(row.get("weakness"))
    scope = _parse_dict_field(row.get("structured_scope"))

    # Header
    header = f"Report #{report_id}: {title}" if report_id and title else title or f"Report #{report_id}"

    parts: list[str] = [header] if header else []

    # Classification block
    meta_lines: list[str] = []
    team_handle = _safe(team.get("handle"))
    team_name = _safe(team.get("profile", {}).get("name") if isinstance(team.get("profile"), dict) else "")
    org = team_name or team_handle
    if org:
        meta_lines.append(f"Organization: {org}" + (f" ({team_handle})" if team_name and team_handle else ""))

    weakness_name = _safe(weakness.get("name"))
    if weakness_name:
        meta_lines.append(f"Weakness: {weakness_name}")

    asset_id = _safe(scope.get("asset_identifier"))
    asset_type = _safe(scope.get("asset_type"))
    severity = _safe(scope.get("max_severity"))
    if asset_id or severity:
        asset_line = "Asset: "
        if asset_id:
            asset_line += asset_id
        if asset_type:
            asset_line += f" ({asset_type})"
        if severity:
            asset_line += f" — Severity: {severity}"
        meta_lines.append(asset_line)

    reporter_name = _safe(reporter.get("username"))
    if reporter_name:
        meta_lines.append(f"Reporter: {reporter_name}")

    status_parts = []
    if substate:
        status_parts.append(f"Status: {substate}")
    if has_bounty in ("True", "true", "1"):
        status_parts.append("Bounty: Yes")
    elif has_bounty in ("False", "false", "0"):
        status_parts.append("Bounty: No")
    if vote_count and vote_count != "0":
        status_parts.append(f"Votes: {vote_count}")
    if status_parts:
        meta_lines.append(" | ".join(status_parts))

    if disclosed_at:
        meta_lines.append(f"Disclosed: {disclosed_at[:10]}")

    if meta_lines:
        parts.append("\n".join(meta_lines))

    # Vulnerability narrative (primary intel content)
    if vuln_info:
        parts.append(f"Vulnerability Details:\n{vuln_info}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Data models
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
# Main ingestor
# ---------------------------------------------------------------------------


class HackerOneIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.skip_no_content: bool = args.skip_no_content
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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

                logger.info(
                    "[%s] Starting ingestion of %s (limit=%s, resume_from=%d)",
                    split,
                    HF_DATASET_ID,
                    limit or "none",
                    checkpoint,
                )

                await self._ingest_split(split, stats, limit, checkpoint)
                await self._flush_upsert_buffer(stats)
                await self._save_checkpoint(split, stats.records_seen)
                stats.log_progress()
                logger.info("[%s] Ingestion complete.", split)
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
    # Split ingestion
    # ------------------------------------------------------------------

    async def _ingest_split(
        self,
        split: str,
        stats: IngestStats,
        limit: int,
        checkpoint: int,
    ) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(HF_DATASET_ID, split=split, streaming=True, token=HF_TOKEN or None)

        async for row in self._stream_rows(ds, stats, limit, checkpoint):
            # Skip no-content reports if requested (no vuln text available)
            if self.skip_no_content and _safe(row.get("visibility")) == "no-content":
                stats.records_skipped += 1
                continue

            text = build_document_text(row)
            if len(text) < self.min_text_len:
                stats.records_skipped += 1
                continue

            report_id = _safe(row.get("id"))
            if report_id:
                doc_id = str(uuid.UUID(bytes=hashlib.sha256(f"h1:{report_id}".encode()).digest()[:16]))
            else:
                doc_id = str(uuid.UUID(bytes=hashlib.sha256(text[:256].encode()).digest()[:16]))

            # Parse structured sub-fields for metadata
            weakness = _parse_dict_field(row.get("weakness"))
            team = _parse_dict_field(row.get("team"))
            scope = _parse_dict_field(row.get("structured_scope"))
            reporter = _parse_dict_field(row.get("reporter"))

            metadata: dict = {
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "split": split,
                "doc_id": doc_id,
                "document_type": "bug_bounty_report",
                "provenance": "public_disclosure",
                "ingest_date": TODAY,
            }
            if report_id:
                metadata["report_id"] = report_id
            for src, dest in [
                ("title", "title"),
                ("substate", "substate"),
                ("visibility", "visibility"),
                ("vote_count", "vote_count"),
            ]:
                val = _safe(row.get(src))
                if val:
                    metadata[dest] = val
            bounty_val = str(row.get("has_bounty?", "")).strip()
            if bounty_val:
                metadata["has_bounty"] = bounty_val in ("True", "true", "1")
            for dt_field, dest in [("created_at", "created_at"), ("disclosed_at", "disclosed_at")]:
                val = _safe(row.get(dt_field))
                if val:
                    metadata[dest] = val[:10]
            weakness_name = _safe(weakness.get("name"))
            if weakness_name:
                metadata["weakness"] = weakness_name
            weakness_id = weakness.get("id")
            if weakness_id is not None:
                metadata["weakness_id"] = int(weakness_id)
            team_handle = _safe(team.get("handle"))
            if team_handle:
                metadata["team_handle"] = team_handle
            asset_id = _safe(scope.get("asset_identifier"))
            if asset_id:
                metadata["asset_identifier"] = asset_id
            severity = _safe(scope.get("max_severity"))
            if severity:
                metadata["max_severity"] = severity
            reporter_name = _safe(reporter.get("username"))
            if reporter_name:
                metadata["reporter"] = reporter_name

            try:
                await self._process_record(doc_id, text, metadata, stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing report %s: %s", report_id, exc)

    async def _process_record(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
        stats: IngestStats,
    ) -> None:
        stats.records_processed += 1
        payload = {"text": text, **metadata}
        self._upsert_buffer.append(qdrant_models.PointStruct(id=doc_id, vector=[0.0] * EMBEDDING_DIM, payload=payload))
        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)
        if stats.records_processed % 500 == 0:
            stats.log_progress()

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
            await self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
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
        while True:
            row = await loop.run_in_executor(None, next, it, _sentinel)
            if row is _sentinel:
                break

            stats.records_seen += 1

            if stats.records_seen <= checkpoint:
                skipped_ff += 1
                if skipped_ff % 1_000 == 0:
                    logger.info(
                        "[%s] Fast-forwarding checkpoint: %d/%d",
                        stats.split,
                        skipped_ff,
                        checkpoint,
                    )
                continue

            yield row

            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("[%s] Reached --limit %d — stopping.", stats.split, limit)
                break

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    async def _save_checkpoint(self, split: str, cursor: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(f"{CHECKPOINT_KEY_PREFIX}{split}", cursor)
        logger.info("[%s] Checkpoint saved: %d records", split, cursor)

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
        description="Ingest Hacker0x01/hackerone_disclosed_reports into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=ALL_SPLITS,
        choices=ALL_SPLITS,
        help="Which dataset splits to ingest.",
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N records per split (0=no limit)")
    p.add_argument(
        "--skip-no-content",
        action="store_true",
        help="Skip reports with visibility=no-content (no vulnerability text)",
    )
    p.add_argument("--resume", action="store_true", help="Resume each split from its Redis checkpoint")
    p.add_argument("--dry-run", action="store_true", help="Parse but skip Qdrant writes and Redis updates")
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
        "Starting HackerOne ingest | splits=%s limit=%s skip_no_content=%s dry_run=%s",
        args.splits,
        args.limit or "none",
        args.skip_no_content,
        args.dry_run,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant or Redis.")

    ingestor = HackerOneIngestor(args)
    asyncio.run(ingestor.run(splits=args.splits, limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
