"""
OSIA Yahoo Finance Ingestion

Streams three high-value text tables from the defeatbeta/yahoo-finance-data
HuggingFace dataset, builds rich documents, and upserts into the
'yahoo-finance' Qdrant collection for Finance desk RAG retrieval.

Tables ingested (selected via --tables):
  stock_news                — news articles with full paragraph text
  stock_earning_call_transcripts — verbatim earnings call transcripts
  stock_profile             — company profiles and business summaries

Deliberately excluded (quantitative only, not useful for text RAG):
  stock_prices, exchange_rate, daily_treasury_yield, stock_dividend_events,
  stock_split_events, stock_shares_outstanding, stock_tailing_eps,
  stock_earning_calendar, stock_sec_filing, stock_revenue_breakdown,
  stock_statement

Usage:
  uv run python scripts/ingest_yahoo_finance.py
  uv run python scripts/ingest_yahoo_finance.py --tables stock_news stock_profile
  uv run python scripts/ingest_yahoo_finance.py --limit 10000 --dry-run
  uv run python scripts/ingest_yahoo_finance.py --resume
  uv run python scripts/ingest_yahoo_finance.py --symbol-filter AAPL TSLA NVDA

Options:
  --tables              Tables to ingest (default: all three text tables)
  --limit N             Stop after N source records per table (0 = no limit)
  --symbol-filter       Only ingest records for these ticker symbols
  --resume              Resume each table from its last Redis checkpoint
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a document body (default: 80)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings API)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
"""

import argparse
import asyncio
import hashlib
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
logger = logging.getLogger("osia.yahoo_finance_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "yahoo-finance"
EMBEDDING_DIM = 384
SOURCE_LABEL = "Yahoo Finance (defeatbeta/yahoo-finance-data)"

HF_DATASET_ID = "defeatbeta/yahoo-finance-data"
HF_PARQUET_BASE = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data"

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

AVAILABLE_TABLES = ["stock_news", "stock_earning_call_transcripts", "stock_profile"]
DEFAULT_TABLES = AVAILABLE_TABLES

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Chunking — for long transcripts and articles
CHUNK_SIZE = 1500  # characters
CHUNK_OVERLAP = 225  # ~15%


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunker with sentence-level fallback and overlap."""
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
        if len(para) > chunk_size:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if current_len + len(sent) > chunk_size and current_parts:
                    current_parts, current_len = flush(current_parts)
                current_parts.append(sent)
                current_len += len(sent) + 1
            continue
        if current_len + len(para) > chunk_size and current_parts:
            current_parts, current_len = flush(current_parts)
        current_parts.append(para)
        current_len += len(para) + 2

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if len(c.strip()) >= 50]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(val: object, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "NaT", "") else default


def _join_paragraphs(paragraphs_field: object) -> str:
    """
    Reconstruct article/transcript body from the HuggingFace nested struct.
    Handles both list-of-dicts and dict-of-lists layouts that pyarrow may produce.
    """
    if not paragraphs_field:
        return ""

    # List of dicts: [{"paragraph_number": 1, "paragraph": "..."}]
    if isinstance(paragraphs_field, list):
        items = paragraphs_field
        try:
            items = sorted(items, key=lambda x: int(x.get("paragraph_number", 0)))
        except (TypeError, ValueError):
            pass
        return "\n\n".join(
            _safe(item.get("paragraph") or item.get("content", "")) for item in items if isinstance(item, dict)
        )

    # Dict-of-lists: {"paragraph_number": [...], "paragraph": [...]}
    if isinstance(paragraphs_field, dict):
        paras = paragraphs_field.get("paragraph") or paragraphs_field.get("content") or []
        numbers = paragraphs_field.get("paragraph_number") or list(range(len(paras)))
        pairs = sorted(zip(numbers, paras, strict=False), key=lambda x: x[0])
        return "\n\n".join(_safe(p) for _, p in pairs)

    return _safe(paragraphs_field)


def _join_transcript(transcripts_field: object) -> str:
    """
    Reconstruct an earnings call transcript with speaker attribution.
    Format: "Speaker: content"
    """
    if not transcripts_field:
        return ""

    if isinstance(transcripts_field, list):
        items = transcripts_field
        try:
            items = sorted(items, key=lambda x: int(x.get("paragraph_number", 0)))
        except (TypeError, ValueError):
            pass
        lines = []
        for item in items:
            if not isinstance(item, dict):
                continue
            speaker = _safe(item.get("speaker", ""))
            content = _safe(item.get("content", ""))
            if content:
                lines.append(f"{speaker}: {content}" if speaker else content)
        return "\n\n".join(lines)

    if isinstance(transcripts_field, dict):
        speakers = transcripts_field.get("speaker") or []
        contents = transcripts_field.get("content") or []
        numbers = transcripts_field.get("paragraph_number") or list(range(len(contents)))
        triples = sorted(zip(numbers, speakers, contents, strict=False), key=lambda x: x[0])
        lines = []
        for _, spk, cnt in triples:
            cnt_s = _safe(cnt)
            spk_s = _safe(spk)
            if cnt_s:
                lines.append(f"{spk_s}: {cnt_s}" if spk_s else cnt_s)
        return "\n\n".join(lines)

    return _safe(transcripts_field)


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------


def build_news_doc(row: dict) -> tuple[str, str, dict]:
    """
    Returns (header, body, metadata) for a news article row.
    Header is prepended to every chunk; body is chunked separately.
    """
    title = _safe(row.get("title"))
    publisher = _safe(row.get("publisher"))
    date = _safe(row.get("report_date"))
    article_type = _safe(row.get("type"))
    doc_uuid = _safe(row.get("uuid"))

    related_raw = row.get("related_symbols") or []
    if isinstance(related_raw, list):
        symbols = ", ".join(_safe(s) for s in related_raw if _safe(s))
    else:
        symbols = _safe(related_raw)

    header_parts = ["Source: Yahoo Finance News"]
    if title:
        header_parts.append(f"Title: {title}")
    if publisher:
        header_parts.append(f"Publisher: {publisher}")
    if date:
        header_parts.append(f"Date: {date}")
    if article_type:
        header_parts.append(f"Type: {article_type}")
    if symbols:
        header_parts.append(f"Related Symbols: {symbols}")
    header = "\n".join(header_parts)

    body = _join_paragraphs(row.get("news"))

    metadata = {
        "source": SOURCE_LABEL,
        "document_type": "financial_news",
        "provenance": "yahoo_finance_news",
        "table": "stock_news",
        "ingest_date": TODAY,
        "ingested_at_unix": int(time.time()),
    }
    if doc_uuid:
        metadata["original_uuid"] = doc_uuid
    if title:
        metadata["title"] = title
    if publisher:
        metadata["publisher"] = publisher
    if date:
        metadata["report_date"] = date
    if symbols:
        metadata["related_symbols"] = symbols
        metadata["entity_tags"] = [s.strip() for s in symbols.split(",") if s.strip()]
    if article_type:
        metadata["article_type"] = article_type

    return header, body, metadata


def build_transcript_doc(row: dict) -> tuple[str, str, dict]:
    """
    Returns (header, body, metadata) for an earnings call transcript row.
    """
    symbol = _safe(row.get("symbol"))
    fiscal_year = _safe(row.get("fiscal_year"))
    fiscal_quarter = _safe(row.get("fiscal_quarter"))
    date = _safe(row.get("report_date"))

    period = ""
    if fiscal_year:
        period = f"FY{fiscal_year}"
        if fiscal_quarter:
            period += f" Q{fiscal_quarter}"

    header_parts = ["Source: Yahoo Finance Earnings Call Transcript"]
    if symbol:
        header_parts.append(f"Symbol: {symbol}")
    if period:
        header_parts.append(f"Period: {period}")
    if date:
        header_parts.append(f"Date: {date}")
    header = "\n".join(header_parts)

    body = _join_transcript(row.get("transcripts"))

    metadata = {
        "source": SOURCE_LABEL,
        "document_type": "earnings_call_transcript",
        "provenance": "yahoo_finance_transcripts",
        "table": "stock_earning_call_transcripts",
        "ingest_date": TODAY,
        "ingested_at_unix": int(time.time()),
    }
    if symbol:
        metadata["symbol"] = symbol
        metadata["entity_tags"] = [symbol]
    if fiscal_year:
        metadata["fiscal_year"] = fiscal_year
    if fiscal_quarter:
        metadata["fiscal_quarter"] = fiscal_quarter
    if date:
        metadata["report_date"] = date
    if period:
        metadata["period"] = period

    return header, body, metadata


def build_profile_doc(row: dict) -> tuple[str, str, dict]:
    """
    Returns (header, body, metadata) for a company profile row.
    """
    symbol = _safe(row.get("symbol"))
    industry = _safe(row.get("industry"))
    sector = _safe(row.get("sector"))
    country = _safe(row.get("country"))
    city = _safe(row.get("city"))
    employees = _safe(row.get("full_time_employees"))
    date = _safe(row.get("report_date"))
    summary = _safe(row.get("long_business_summary"))

    location_parts = [p for p in [city, country] if p]
    location = ", ".join(location_parts)

    header_parts = ["Source: Yahoo Finance Company Profile"]
    if symbol:
        header_parts.append(f"Symbol: {symbol}")
    if sector:
        header_parts.append(f"Sector: {sector}")
    if industry:
        header_parts.append(f"Industry: {industry}")
    if location:
        header_parts.append(f"Location: {location}")
    if employees:
        header_parts.append(f"Full-time Employees: {employees}")
    if date:
        header_parts.append(f"As of: {date}")
    header = "\n".join(header_parts)

    metadata = {
        "source": SOURCE_LABEL,
        "document_type": "company_profile",
        "provenance": "yahoo_finance_profile",
        "table": "stock_profile",
        "ingest_date": TODAY,
        "ingested_at_unix": int(time.time()),
    }
    if symbol:
        metadata["symbol"] = symbol
        metadata["entity_tags"] = [symbol]
    if sector:
        metadata["sector"] = sector
    if industry:
        metadata["industry"] = industry
    if country:
        metadata["country"] = country
    if employees:
        metadata["full_time_employees"] = employees
    if date:
        metadata["report_date"] = date

    return header, summary, metadata


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    table: str = ""
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    chunks_produced: int = 0
    points_upserted: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "[%s] seen=%d processed=%d skipped=%d chunks=%d upserted=%d errors=%d elapsed=%s",
            self.table,
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.chunks_produced,
            self.points_upserted,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class YahooFinanceIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.tables: list[str] = args.tables
        self.symbol_filter: set[str] | None = {s.upper() for s in args.symbol_filter} if args.symbol_filter else None
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self, limit: int, resume: bool) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()
            for table in self.tables:
                logger.info("=== Starting table: %s ===", table)
                checkpoint = 0
                if resume:
                    checkpoint = await self._load_checkpoint(table)
                    if checkpoint:
                        logger.info("[%s] Resuming from record %d", table, checkpoint)
                stats = IngestStats(table=table)
                await self._ingest_table(table, stats, limit, checkpoint)
                await self._flush_upsert_buffer(stats)
                await self._save_checkpoint(table, stats.records_seen)
                stats.log_progress()
                logger.info("=== Table %s complete ===", table)
            logger.info("All tables ingested.")
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
    # Table ingestion
    # ------------------------------------------------------------------

    async def _ingest_table(self, table: str, stats: IngestStats, limit: int, checkpoint: int) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        parquet_url = f"{HF_PARQUET_BASE}/{table}.parquet"
        logger.info("[%s] Loading from %s", table, parquet_url)

        ds = load_dataset(
            "parquet",
            data_files={"train": parquet_url},
            split="train",
            streaming=True,
        )

        loop = asyncio.get_event_loop()

        def _iter():
            yield from ds

        it = _iter()
        _sentinel = object()

        while True:
            row = await loop.run_in_executor(None, next, it, _sentinel)
            if row is _sentinel:
                logger.info("[%s] Dataset exhausted at %d records.", table, stats.records_seen)
                break

            stats.records_seen += 1

            if stats.records_seen <= checkpoint:
                if stats.records_seen % 50_000 == 0:
                    logger.info("[%s] Fast-forwarding: %d/%d", table, stats.records_seen, checkpoint)
                continue

            # Symbol filter
            if self.symbol_filter:
                sym = _safe(row.get("symbol") or "").upper()
                syms_raw = row.get("related_symbols") or []
                row_symbols = {sym} if sym else set()
                if isinstance(syms_raw, list):
                    row_symbols |= {_safe(s).upper() for s in syms_raw}
                if not row_symbols & self.symbol_filter:
                    stats.records_skipped += 1
                    continue

            try:
                await self._process_row(table, row, stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("[%s] Error processing row %d: %s", table, stats.records_seen, exc)

            if stats.records_processed % 5000 == 0 and stats.records_processed > 0:
                stats.log_progress()
                await self._save_checkpoint(table, stats.records_seen)

            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("[%s] Reached --limit %d — stopping.", table, limit)
                break

        await self._save_checkpoint(table, stats.records_seen)

    async def _process_row(self, table: str, row: dict, stats: IngestStats) -> None:
        if table == "stock_news":
            header, body, metadata = build_news_doc(row)
            dedup_key = _safe(row.get("uuid")) or f"news:{stats.records_seen}"
        elif table == "stock_earning_call_transcripts":
            header, body, metadata = build_transcript_doc(row)
            symbol = _safe(row.get("symbol"))
            fy = _safe(row.get("fiscal_year"))
            fq = _safe(row.get("fiscal_quarter"))
            dedup_key = f"transcript:{symbol}:{fy}:{fq}"
        elif table == "stock_profile":
            header, body, metadata = build_profile_doc(row)
            dedup_key = f"profile:{_safe(row.get('symbol'))}"
        else:
            return

        if len(body) < self.min_text_len and len(header) < self.min_text_len:
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        # Build chunks — header prepended to every chunk for self-contained retrieval
        body_chunks = chunk_text(body) if body else []
        if not body_chunks:
            # Header-only document (e.g. profile with no summary)
            chunks = [header]
        else:
            chunks = [f"{header}\n\n{chunk}" for chunk in body_chunks]

        stats.chunks_produced += len(chunks)

        base_id = str(uuid.UUID(bytes=hashlib.sha256(f"yf:{dedup_key}".encode()).digest()[:16]))
        chunk_count = len(chunks)
        metadata["chunk_count"] = chunk_count

        for idx, chunk_text_val in enumerate(chunks):
            if chunk_count == 1:
                point_id = base_id
            else:
                suffix = f"{dedup_key}:chunk{idx}"
                point_id = str(uuid.UUID(bytes=hashlib.sha256(f"yf:{suffix}".encode()).digest()[:16]))

            payload = {"text": chunk_text_val, "chunk_index": idx, **metadata}
            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload=payload,
                )
            )

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

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
    # Checkpointing (per-table)
    # ------------------------------------------------------------------

    async def _save_checkpoint(self, table: str, cursor: int) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(f"osia:yahoo_finance:{table}:checkpoint", cursor)
        logger.info("[%s] Checkpoint saved: %d records", table, cursor)

    async def _load_checkpoint(self, table: str) -> int:
        if not self._redis:
            return 0
        val = await self._redis.get(f"osia:yahoo_finance:{table}:checkpoint")
        return int(val) if val else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest Yahoo Finance text tables into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tables",
        nargs="+",
        default=DEFAULT_TABLES,
        choices=AVAILABLE_TABLES,
        metavar="TABLE",
        help=f"Tables to ingest. Choices: {', '.join(AVAILABLE_TABLES)}",
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N records per table (0=no limit)")
    p.add_argument(
        "--symbol-filter",
        nargs="+",
        metavar="SYMBOL",
        dest="symbol_filter",
        help="Only ingest records for these ticker symbols (e.g. AAPL TSLA NVDA)",
    )
    p.add_argument("--resume", action="store_true", help="Resume each table from Redis checkpoint")
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Parse and chunk but skip Qdrant writes and Redis updates",
    )
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=80, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set in environment — required for the embeddings API.")

    logger.info(
        "Yahoo Finance ingest | tables=%s limit=%s symbol_filter=%s dry_run=%s",
        args.tables,
        args.limit or "none",
        args.symbol_filter or "all",
        args.dry_run,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant or Redis.")

    ingestor = YahooFinanceIngestor(args)
    asyncio.run(ingestor.run(limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
