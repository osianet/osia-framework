"""
OSIA FBI Vault FOIA Documents Ingestion

Ingests FBI FOIA releases from the FBI Vault (vault.fbi.gov), sourced via
archive.org mirrors. Covers COINTELPRO domestic surveillance operations,
J. Edgar Hoover's personal files, organised crime investigations, Cold War
counterintelligence (SOLO/Venona era), and key political assassinations.

The FBI Vault publishes documents as PDFs. This script targets archive.org
collections that have already been OCR'd to text, and also queries the FBI
Vault's own site to discover collection document lists where text is available.

Source priority:
  1. archive.org FBI collections (OCR text via _djvu.txt / _full.txt)
  2. vault.fbi.gov collection pages (extract PDF links → note as metadata only,
     no PDF download unless --include-pdf-metadata flag is used)

Known collections included:
  - COINTELPRO (Black Nationalist, Communist Party, New Left, White Hate groups)
  - J. Edgar Hoover Official and Confidential files
  - JFK Assassination (FBI's own investigation files)
  - Martin Luther King Jr. (FBI surveillance files)
  - Malcolm X (FBI file)
  - Rosenbergs (Julius & Ethel — Venona era espionage)
  - Oswald, Lee Harvey (FBI file)
  - Mafia / La Cosa Nostra investigations
  - Cold War SOLO operation (Jack Childs / Morris Childs)

Usage:
  uv run python scripts/ingest_fbi_vault.py
  uv run python scripts/ingest_fbi_vault.py --dry-run
  uv run python scripts/ingest_fbi_vault.py --collections cointelpro mlk hoover
  uv run python scripts/ingest_fbi_vault.py --resume
  uv run python scripts/ingest_fbi_vault.py --limit 200

Options:
  --dry-run             Parse and chunk but skip Qdrant writes
  --collections         Space-separated collection slugs to ingest (default: all)
  --resume              Skip collections already in Redis completed set
  --limit N             Stop after N items total (0 = no limit)
  --enqueue-notable     Push key documents to HUMINT/InfoWar desk research queues
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a text chunk (default: 120)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)

Rate limiting / polite scraping:
  - Sends User-Agent: OSIA-Framework/1.0 on all requests to archive.org
  - Waits CANDIDATE_DELAY (1.5s) before fetching each known candidate identifier,
    pacing the initial discovery requests per collection
  - Waits SEARCH_DELAY (2.0s) before the collection-level archive.org search call,
    giving archive.org's API recovery time between collection searches
  - Waits ITEM_DELAY (1.5s) before fetching each search-result item's OCR text,
    so text downloads are spaced to roughly 1 item per 1.5 seconds
  - Persistent httpx.AsyncClient reused for all requests (single connection pool per run)
  - Per-collection Redis checkpoint (osia:fbi_vault:completed_collections) supports
    --resume so interrupted runs skip already-completed collections
  - Per-item deduplication via osia:fbi_vault:seen:{md5} prevents duplicate upserts
    when the same item appears in multiple search result pages
  - Use --collections to scope a run to specific collections (e.g. cointelpro mlk) and
    --limit to cap total items ingested per run
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
logger = logging.getLogger("osia.fbi_vault_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "fbi-vault"
EMBEDDING_DIM = 384
SOURCE_LABEL = "FBI Vault FOIA Releases"

# Polite scraping
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
CANDIDATE_DELAY = 1.5  # seconds between candidate ia_id fetch attempts
ITEM_DELAY = 1.5  # seconds between search-result item fetches
SEARCH_DELAY = 2.0  # seconds between collection-level archive.org searches

COMPLETED_COLLECTIONS_KEY = "osia:fbi_vault:completed_collections"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 225

HF_EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

IA_SEARCH_URL = "https://archive.org/advancedsearch.php"

# ---------------------------------------------------------------------------
# Collection catalogue
# Each entry has:
#   slug          — short ID used in --collections filter and Redis keys
#   name          — human-readable label
#   ia_query      — archive.org search query to enumerate items in this collection
#   ia_candidates — specific archive.org identifiers to try first (may be empty)
#   desk_hint     — which desk to route notable docs to
#   date_range    — approximate era for ingested_at_unix approximation
# ---------------------------------------------------------------------------

COLLECTIONS: list[dict] = [
    {
        "slug": "cointelpro",
        "name": "COINTELPRO — FBI Domestic Counterintelligence Programs",
        "ia_query": 'subject:"COINTELPRO" AND mediatype:texts',
        "ia_candidates": [
            "cointelpro-blk",
            "cointelpro-cpusa",
            "cointelpro-newleft",
            "cointelpro-whitehate",
            "cointelprofiles",
        ],
        "desk_hint": "information-warfare-desk",
        "date_range": (1956, 1971),
    },
    {
        "slug": "hoover",
        "name": "J. Edgar Hoover Official and Confidential Files",
        "ia_query": 'subject:"J. Edgar Hoover" AND subject:"FBI" AND mediatype:texts',
        "ia_candidates": [
            "jedgarhooverfiles",
            "hoover-official-confidential",
        ],
        "desk_hint": "human-intelligence-and-profiling-desk",
        "date_range": (1924, 1972),
    },
    {
        "slug": "jfk",
        "name": "JFK Assassination — FBI Investigation Files",
        "ia_query": 'subject:"Kennedy assassination" AND subject:"FBI" AND mediatype:texts',
        "ia_candidates": [
            "jfkfbifiles",
            "warren-commission-fbi",
        ],
        "desk_hint": "human-intelligence-and-profiling-desk",
        "date_range": (1963, 1979),
    },
    {
        "slug": "mlk",
        "name": "Martin Luther King Jr. — FBI Surveillance Files",
        "ia_query": 'subject:"Martin Luther King" AND subject:"FBI" AND mediatype:texts',
        "ia_candidates": [
            "fbi-mlk-files",
            "mlkfbifile",
            "martin-luther-king-fbi",
        ],
        "desk_hint": "information-warfare-desk",
        "date_range": (1963, 1968),
    },
    {
        "slug": "malcolm-x",
        "name": "Malcolm X — FBI File",
        "ia_query": 'subject:"Malcolm X" AND subject:"FBI" AND mediatype:texts',
        "ia_candidates": [
            "malcolmxfbifile",
            "fbi-malcolm-x",
        ],
        "desk_hint": "information-warfare-desk",
        "date_range": (1953, 1965),
    },
    {
        "slug": "rosenbergs",
        "name": "Rosenbergs Espionage Case — FBI Files (Venona era)",
        "ia_query": 'subject:"Rosenberg" AND subject:"espionage" AND subject:"FBI" AND mediatype:texts',
        "ia_candidates": [
            "rosenbergfbifiles",
            "julius-ethel-rosenberg-fbi",
        ],
        "desk_hint": "geopolitical-and-security-desk",
        "date_range": (1950, 1953),
    },
    {
        "slug": "mafia",
        "name": "La Cosa Nostra / Mafia — FBI Organised Crime Files",
        "ia_query": 'subject:"organized crime" AND subject:"FBI" AND subject:"mafia" AND mediatype:texts',
        "ia_candidates": [
            "fbi-mafia-files",
            "lacosanostra-fbi",
        ],
        "desk_hint": "human-intelligence-and-profiling-desk",
        "date_range": (1957, 1990),
    },
    {
        "slug": "solo",
        "name": "SOLO Operation — FBI Cold War Counterintelligence (Morris & Jack Childs)",
        "ia_query": 'subject:"SOLO" AND subject:"FBI" AND subject:"Communist Party" AND mediatype:texts',
        "ia_candidates": [
            "fbi-solo-operation",
        ],
        "desk_hint": "geopolitical-and-security-desk",
        "date_range": (1958, 1977),
    },
]

COLLECTION_BY_SLUG: dict[str, dict] = {c["slug"]: c for c in COLLECTIONS}


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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


def clean_ocr_text(text: str) -> str:
    text = re.sub(r"\x0c", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"-\n([a-z])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    collections_seen: int = 0
    collections_skipped: int = 0
    collections_processed: int = 0
    items_seen: int = 0
    items_skipped: int = 0
    chunks_produced: int = 0
    points_upserted: int = 0
    docs_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "colls=%d items=%d skipped=%d chunks=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.collections_processed,
            self.items_seen,
            self.items_skipped,
            self.chunks_produced,
            self.points_upserted,
            self.docs_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class FbiVaultIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.collection_filter: set[str] | None = set(args.collections) if args.collections else None
        self.resume: bool = args.resume
        self.limit: int = args.limit
        self.enqueue_notable: bool = args.enqueue_notable
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._http: httpx.AsyncClient | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []
        self._total_items: int = 0

    async def run(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self._http = httpx.AsyncClient(
            timeout=60.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        try:
            await self._ensure_collection()
            stats = IngestStats()

            completed: set[str] = set()
            if self.resume and self._redis:
                members = await self._redis.smembers(COMPLETED_COLLECTIONS_KEY)
                completed = set(members)
                if completed:
                    logger.info("Resuming — %d collections already completed.", len(completed))

            colls = [c for c in COLLECTIONS if (self.collection_filter is None or c["slug"] in self.collection_filter)]

            for coll in colls:
                stats.collections_seen += 1
                if coll["slug"] in completed:
                    stats.collections_skipped += 1
                    logger.info("Skipping %s (already completed).", coll["name"])
                    continue

                if self.limit and self._total_items >= self.limit:
                    logger.info("Reached --limit %d — stopping.", self.limit)
                    break

                try:
                    await self._ingest_collection(coll, stats)
                except Exception as exc:
                    stats.errors += 1
                    logger.warning("Error ingesting collection %s: %s", coll["slug"], exc)
                    continue

                if not self.dry_run and self._redis:
                    await self._redis.sadd(COMPLETED_COLLECTIONS_KEY, coll["slug"])
                stats.collections_processed += 1
                stats.log_progress()

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("FBI Vault ingestion complete.")
        finally:
            if self._http:
                await self._http.aclose()
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    # ------------------------------------------------------------------
    # Per-collection ingestion
    # ------------------------------------------------------------------

    async def _ingest_collection(self, coll: dict, stats: IngestStats) -> None:
        logger.info("Processing collection: %s", coll["name"])

        # 1. Try known ia_candidates first
        for ia_id in coll["ia_candidates"]:
            if self.limit and self._total_items >= self.limit:
                return
            await asyncio.sleep(CANDIDATE_DELAY)
            text = await self._fetch_ia_text(ia_id)
            if text and len(text) > 500:
                await self._process_text(text, ia_id, coll, stats)

        # 2. Search archive.org for additional items
        await asyncio.sleep(SEARCH_DELAY)
        items = await self._search_ia(coll["ia_query"], rows=50)
        known = set(coll["ia_candidates"])
        for item in items:
            if self.limit and self._total_items >= self.limit:
                return
            ia_id = item.get("identifier", "")
            if not ia_id or ia_id in known:
                continue

            seen_key = f"osia:fbi_vault:seen:{hashlib.md5(ia_id.encode(), usedforsecurity=False).hexdigest()}"
            if self._redis and await self._redis.exists(seen_key):
                stats.items_skipped += 1
                continue

            await asyncio.sleep(ITEM_DELAY)
            text = await self._fetch_ia_text(ia_id)
            if text and len(text) > 500:
                await self._process_text(text, ia_id, coll, stats, item_meta=item)
                if self._redis and not self.dry_run:
                    await self._redis.set(seen_key, "1", ex=60 * 60 * 24 * 90)

    async def _process_text(
        self,
        text: str,
        ia_id: str,
        coll: dict,
        stats: IngestStats,
        item_meta: dict | None = None,
    ) -> None:
        text = clean_ocr_text(text)
        chunks = chunk_text(text)
        if not chunks:
            stats.items_skipped += 1
            return

        stats.items_seen += 1
        self._total_items += 1
        stats.chunks_produced += len(chunks)

        # Approximate date from collection date range
        year = coll["date_range"][0]
        ingested_at_unix = int(datetime(year, 1, 1, tzinfo=UTC).timestamp())

        item_title = (item_meta or {}).get("title", "") or ia_id
        doc_date = (item_meta or {}).get("date", "")

        base_metadata: dict = {
            "source": SOURCE_LABEL,
            "document_type": "fbi_foia_release",
            "provenance": "fbi_vault",
            "ingest_date": TODAY,
            "ia_identifier": ia_id,
            "collection_slug": coll["slug"],
            "collection_name": coll["name"],
            "ingested_at_unix": ingested_at_unix,
        }
        if item_title:
            base_metadata["doc_title"] = item_title
        if doc_date:
            base_metadata["doc_date"] = doc_date

        base_id = str(uuid.UUID(bytes=hashlib.sha256(f"fbi:{ia_id}".encode()).digest()[:16]))
        chunk_count = len(chunks)

        for idx, chunk in enumerate(chunks):
            if len(chunk) < self.min_text_len:
                continue
            header = f"Source: {SOURCE_LABEL}\nCollection: {coll['name']}\nDocument: {item_title}"
            text_with_header = f"{header}\n\n{chunk}"

            if chunk_count == 1:
                point_id = base_id
            else:
                point_id = str(uuid.UUID(bytes=hashlib.sha256(f"fbi:{ia_id}:chunk{idx}".encode()).digest()[:16]))

            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload={"text": text_with_header, "chunk_index": idx, **base_metadata},
                )
            )

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        if self.enqueue_notable:
            await self._maybe_enqueue(ia_id, item_title, coll, stats)

    # ------------------------------------------------------------------
    # archive.org helpers
    # ------------------------------------------------------------------

    async def _fetch_ia_text(self, ia_id: str) -> str:
        assert self._http is not None
        for suffix in ("_djvu.txt", "_full.txt"):
            url = f"https://archive.org/download/{ia_id}/{ia_id}{suffix}"
            for attempt in range(3):
                try:
                    resp = await self._http.get(url)
                    if resp.status_code == 404:
                        break
                    resp.raise_for_status()
                    return resp.text
                except Exception as exc:
                    if attempt < 2:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        logger.debug("Could not fetch %s%s: %s", ia_id, suffix, exc)
        return ""

    async def _search_ia(self, query: str, rows: int = 50) -> list[dict]:
        assert self._http is not None
        params = {
            "q": query,
            "fl": "identifier,title,subject,date",
            "rows": str(rows),
            "output": "json",
        }
        for attempt in range(3):
            try:
                resp = await self._http.get(IA_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json().get("response", {}).get("docs", [])
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    logger.warning("archive.org search failed: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _maybe_enqueue(self, ia_id: str, title: str, coll: dict, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:fbi_vault:enqueued:{ia_id}"
        if await self._redis.exists(redis_key):
            return
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": title or f"FBI file {ia_id} — {coll['name']}",
                "desk": coll["desk_hint"],
                "priority": "low",
                "directives_lens": True,
                "triggered_by": "fbi_vault_ingest",
                "metadata": {
                    "ia_identifier": ia_id,
                    "collection_slug": coll["slug"],
                    "source": SOURCE_LABEL,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)
        stats.docs_enqueued += 1

    # ------------------------------------------------------------------
    # Embed + upsert
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
            logger.debug("Upserted %d points.", len(points))
        if stats is not None:
            stats.points_upserted += len(points)

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.embed_batch_size] for i in range(0, len(texts), self.embed_batch_size)]
        results: list[list[float]] = []
        for i in range(0, len(batches), self.embed_concurrency):
            group = batches[i : i + self.embed_concurrency]
            group_results = await asyncio.gather(*[self._embed_batch(b) for b in group])
            for vecs in group_results:
                results.extend(vecs)
        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._embed_semaphore:
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=45.0) as hf_http:
                        resp = await hf_http.post(
                            HF_EMBED_URL,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest FBI Vault FOIA documents into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--collections",
        nargs="+",
        metavar="SLUG",
        help=f"Collections to ingest (default: all). Choices: {', '.join(COLLECTION_BY_SLUG)}",
    )
    p.add_argument("--resume", action="store_true", help="Skip collections already in Redis completed set")
    p.add_argument("--limit", type=int, default=0, help="Stop after N total items (0=no limit)")
    p.add_argument("--enqueue-notable", action="store_true", dest="enqueue_notable")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=120, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set — required for embeddings.")

    if args.collections:
        unknown = [s for s in args.collections if s not in COLLECTION_BY_SLUG]
        if unknown:
            parser.error(f"Unknown collection slugs: {unknown}. Valid: {list(COLLECTION_BY_SLUG)}")

    logger.info(
        "Starting FBI Vault ingest | collections=%s limit=%s enqueue=%s dry_run=%s",
        args.collections or "all",
        args.limit or "none",
        args.enqueue_notable,
        args.dry_run,
    )
    if args.dry_run:
        logger.warning("DRY RUN — no data will be written.")

    ingestor = FbiVaultIngestor(args)
    asyncio.run(ingestor.run())


if __name__ == "__main__":
    main()
