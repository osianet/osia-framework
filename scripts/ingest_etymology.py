"""
OSIA Etymology Database Ingestion

Streams the Nickmancol/etymology HuggingFace dataset (4.2+ million etymological
relationships across 3,300+ languages from Wiktionary), constructs rich etymology
documents from relational fields, embeds via the HF Inference API, and upserts
into a dedicated 'etymology-database' Qdrant collection.

Dataset: Nickmancol/etymology
  - 4,222,599 rows
  - 2.0+ million unique terms across 3,300+ languages
  - Fields: term, lang, related_term, related_lang, reltype (30+ types)
  - Relation types: inherited_from, borrowed_from, derived_from, cognate_of, etc.

Usage:
  uv run python scripts/ingest_etymology.py
  uv run python scripts/ingest_etymology.py --limit 10000 --dry-run
  uv run python scripts/ingest_etymology.py --lang eng fra deu
  uv run python scripts/ingest_etymology.py --resume

Options:
  --limit N             Stop after N records (0 = no limit)
  --lang                Only ingest terms from these language codes (space-separated)
  --reltype             Only ingest relations of these types (space-separated)
  --resume              Resume from Redis checkpoint
  --dry-run             Parse but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-doc-len         Minimum document length to process (default: 40)

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
logger = logging.getLogger("osia.etymology_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "etymology-database"
EMBEDDING_DIM = 384

HF_DATASET_ID = "Nickmancol/etymology"
SOURCE_LABEL = "Wiktionary Etymology Database (Nickmancol/etymology)"

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY = "osia:etymology:checkpoint"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Common relation types for metadata tagging
RELATION_TYPES = {
    "inherited_from",
    "borrowed_from",
    "derived_from",
    "learned_borrowing_from",
    "semi_learned_borrowing_from",
    "has_prefix",
    "has_suffix",
    "has_affix",
    "compound_of",
    "cognate_of",
    "doublet_with",
    "calque_of",
    "back_from",
    "alteration_of",
    "blend_of",
    "clipping_of",
    "contraction_of",
    "reanalysis_of",
    "reduced_from",
    "shortening_of",
    "slang_term_from",
    "metanalysis_from",
    "onomatopoeia",
    "acronym_of",
    "reduplication_of",
    "transposition_of",
}


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def _safe(val, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "NaN", "") else default


def build_document_text(row: dict) -> str:
    """
    Construct an embeddable etymology document combining the term, its etymology,
    and relationship metadata. Structured so semantic search captures both the
    linguistic form and the etymological narrative.
    """
    term = _safe(row.get("term"))
    lang = _safe(row.get("lang"))
    related_term = _safe(row.get("related_term"))
    related_lang = _safe(row.get("related_lang"))
    reltype = _safe(row.get("reltype"))

    parts: list[str] = []

    # Header with term and language
    if term and lang:
        parts.append(f"Term: {term} ({lang})")
    elif term:
        parts.append(f"Term: {term}")

    # Etymology relationship
    if reltype and related_term and related_lang:
        rel_display = reltype.replace("_", " ").title()
        parts.append(f"Etymology: {rel_display} from {related_term} ({related_lang})")
    elif reltype and related_term:
        rel_display = reltype.replace("_", " ").title()
        parts.append(f"Etymology: {rel_display} from {related_term}")
    elif reltype:
        rel_display = reltype.replace("_", " ").title()
        parts.append(f"Relation Type: {rel_display}")

    # Additional context
    position = row.get("position")
    if position is not None:
        parts.append(f"Position in compound: {position}")

    return "\n".join(parts)


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


class EtymologyIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.lang_filter: set[str] | None = {lang_code.lower() for lang_code in args.lang} if args.lang else None
        self.reltype_filter: set[str] | None = {r.lower() for r in args.reltype} if args.reltype else None
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_doc_len: int = args.min_doc_len

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
                "Starting etymology ingest (limit=%s lang_filter=%s reltype_filter=%s resume_from=%d)",
                limit or "none",
                self.lang_filter or "all",
                self.reltype_filter or "all",
                checkpoint,
            )
            stats = IngestStats()
            await self._ingest(stats, limit, checkpoint)
            await self._flush(stats)
            await self._save_checkpoint(stats.records_seen)
            stats.log_progress()
            logger.info("Etymology ingestion complete.")
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
            term = _safe(row.get("term"))
            lang = _safe(row.get("lang"))

            # Apply language filter
            if self.lang_filter and lang.lower() not in self.lang_filter:
                stats.records_skipped += 1
                continue

            reltype = _safe(row.get("reltype")).lower()

            # Apply relation type filter
            if self.reltype_filter and reltype not in self.reltype_filter:
                stats.records_skipped += 1
                continue

            # Skip empty terms
            if not term or not lang:
                stats.records_skipped += 1
                continue

            text = build_document_text(row)
            if len(text) < self.min_doc_len:
                stats.records_skipped += 1
                continue

            # Generate stable ID from term + lang + reltype
            doc_id = str(uuid.UUID(bytes=hashlib.sha256(f"etym:{lang}:{term}:{reltype}".encode()).digest()[:16]))

            # Build metadata
            metadata: dict = {
                "source": SOURCE_LABEL,
                "dataset": HF_DATASET_ID,
                "document_type": "etymology",
                "ingest_date": TODAY,
                "term": term,
                "lang": lang,
                "reltype": reltype,
            }

            related_term = _safe(row.get("related_term"))
            related_lang = _safe(row.get("related_lang"))
            if related_term:
                metadata["related_term"] = related_term
            if related_lang:
                metadata["related_lang"] = related_lang

            position = row.get("position")
            if position is not None:
                metadata["position"] = int(position)

            group_tag = _safe(row.get("group_tag"))
            if group_tag:
                metadata["group_tag"] = group_tag

            parent_tag = _safe(row.get("parent_tag"))
            if parent_tag:
                metadata["parent_tag"] = parent_tag

            try:
                stats.records_processed += 1
                self._upsert_buffer.append(
                    qdrant_models.PointStruct(
                        id=doc_id, vector=[0.0] * EMBEDDING_DIM, payload={"text": text, **metadata}
                    )
                )
                if len(self._upsert_buffer) >= self.upsert_batch_size:
                    await self._flush(stats)
                if stats.records_processed % 5000 == 0:
                    stats.log_progress()
                    await self._save_checkpoint(stats.records_seen)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error processing %s (%s): %s", term, lang, exc)

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
        _exhausted = False
        while True:
            row = await loop.run_in_executor(None, next, it, _sentinel)
            if row is _sentinel:
                _exhausted = True
                break
            stats.records_seen += 1
            if stats.records_seen <= checkpoint:
                skipped_ff += 1
                if skipped_ff % 10000 == 0:
                    logger.info("Fast-forwarding checkpoint: %d/%d", skipped_ff, checkpoint)
                continue
            yield row
            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("Reached --limit %d.", limit)
                break

        if _exhausted:
            logger.info("Dataset exhausted at %d total records — ingestion complete.", stats.records_seen)

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
        description="Ingest Nickmancol/etymology into OSIA Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lang", nargs="+", metavar="CODE", help="Only ingest terms from these language codes.")
    p.add_argument("--reltype", nargs="+", metavar="TYPE", help="Only ingest relations of these types.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-doc-len", type=int, default=40, dest="min_doc_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not HF_TOKEN:
        parser.error("HF_TOKEN not set.")
    logger.info(
        "Starting etymology ingest | limit=%s lang=%s reltype=%s dry_run=%s",
        args.limit or "none",
        args.lang or "all",
        args.reltype or "all",
        args.dry_run,
    )
    if args.dry_run:
        logger.warning("DRY RUN — no writes.")
    asyncio.run(EtymologyIngestor(args).run(limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
