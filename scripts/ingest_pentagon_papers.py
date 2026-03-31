"""
OSIA Pentagon Papers Ingestion

Downloads and ingests the Pentagon Papers — the classified DoD history of US
involvement in Vietnam (1945–1967), leaked by Daniel Ellsberg in 1971. The full
47-volume study documents the systematic deception of Congress and the public
across four presidential administrations.

Sources (in priority order):
  1. Gravel Edition (5 vols) — archive.org: senatepapers* / gravel-edition identifiers
  2. National Security Archive / George Washington University hosted editions
  3. govinfo.gov GPO Gravel Edition packages

The script searches archive.org for each volume and tries multiple known identifiers,
falling back to a general search query per volume if needed.

Usage:
  uv run python scripts/ingest_pentagon_papers.py
  uv run python scripts/ingest_pentagon_papers.py --dry-run
  uv run python scripts/ingest_pentagon_papers.py --volumes 1 2 3
  uv run python scripts/ingest_pentagon_papers.py --resume
  uv run python scripts/ingest_pentagon_papers.py --enqueue-notable

Options:
  --dry-run             Parse and chunk but skip Qdrant writes
  --volumes             Space-separated volume numbers to ingest (default: all 1–5)
  --resume              Skip volumes already in Redis completed set
  --enqueue-notable     Push volumes to Geopolitical/InfoWar desk research queues
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a chunk (default: 120)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)

Rate limiting / polite scraping:
  - Sends User-Agent: OSIA-Framework/1.0 on all requests to archive.org
  - Waits REQUEST_DELAY (1.5s) before each candidate identifier fetch and before each
    search fallback attempt, spacing requests to archive.org naturally
  - 3-attempt retry with 10s backoff on archive.org search API calls
  - Persistent httpx.AsyncClient reused for all requests (single connection pool per run)
  - Per-volume Redis checkpoint (osia:pentagon_papers:completed_volumes) supports --resume
    so interrupted runs do not re-fetch already ingested volumes
  - ingested_at_unix set to the 1971 publication date, not ingest time, so temporal
    decay scoring reflects the document's true age
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
logger = logging.getLogger("osia.pentagon_papers_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "pentagon-papers"
EMBEDDING_DIM = 384
SOURCE_LABEL = "Pentagon Papers — The Pentagon Papers: The Defense Department History of United States Decisionmaking on Vietnam (Gravel Edition)"

COMPLETED_VOLUMES_KEY = "osia:pentagon_papers:completed_volumes"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Polite scraping
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
REQUEST_DELAY = 1.5  # seconds between requests to archive.org
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 225

HF_EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# archive.org identifier candidates per volume (tried in order)
VOLUMES: list[dict] = [
    {
        "number": 1,
        "title": "Volume I: Vietnam and the US, 1940–1950 / Kennedy Commitments and Programs, 1961",
        "ia_candidates": [
            "pentagonpapersde00beac",
            "pentagonpapers_vol1",
            "thepentagonpapersvolume1",
            "pentagon-papers-vol-1",
        ],
        "search_query": "pentagon papers gravel edition volume 1 vietnam",
        "desk_hint": "geopolitical-and-security-desk",
        "date": "1971",
    },
    {
        "number": 2,
        "title": "Volume II: The Kennedy Counterinsurgency Infrastructure, 1961–1963",
        "ia_candidates": [
            "pentagonpapersde01beac",
            "pentagonpapers_vol2",
            "thepentagonpapersvolume2",
            "pentagon-papers-vol-2",
        ],
        "search_query": "pentagon papers gravel edition volume 2 vietnam counterinsurgency",
        "desk_hint": "geopolitical-and-security-desk",
        "date": "1971",
    },
    {
        "number": 3,
        "title": "Volume III: The Johnson Policy, 1963–1968",
        "ia_candidates": [
            "pentagonpapersde02beac",
            "pentagonpapers_vol3",
            "thepentagonpapersvolume3",
            "pentagon-papers-vol-3",
        ],
        "search_query": "pentagon papers gravel edition volume 3 johnson vietnam",
        "desk_hint": "geopolitical-and-security-desk",
        "date": "1971",
    },
    {
        "number": 4,
        "title": "Volume IV: The Air War in the North, 1965–1968 / Pacification",
        "ia_candidates": [
            "pentagonpapersde03beac",
            "pentagonpapers_vol4",
            "thepentagonpapersvolume4",
            "pentagon-papers-vol-4",
        ],
        "search_query": "pentagon papers gravel edition volume 4 air war vietnam",
        "desk_hint": "geopolitical-and-security-desk",
        "date": "1971",
    },
    {
        "number": 5,
        "title": "Volume V: Justification of the War — Public Statements and the Historical Record",
        "ia_candidates": [
            "pentagonpapersde04beac",
            "thepentagonpapers",
            "pentagonpapers025197",
            "pentagonpapers035197",
        ],
        "search_query": "pentagon papers gravel edition volume 5 justification deception",
        "desk_hint": "information-warfare-desk",
        "date": "1971",
    },
]

# ---------------------------------------------------------------------------
# Text utilities (shared pattern)
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
    """Clean OCR artefacts from djvu/pdf text."""
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
    volumes_seen: int = 0
    volumes_skipped: int = 0
    volumes_processed: int = 0
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
            "vols=%d processed=%d skipped=%d chunks=%d upserted=%d errors=%d elapsed=%s",
            self.volumes_seen,
            self.volumes_processed,
            self.volumes_skipped,
            self.chunks_produced,
            self.points_upserted,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class PentagonPapersIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.volume_filter: set[int] | None = set(args.volumes) if args.volumes else None
        self.resume: bool = args.resume
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
                members = await self._redis.smembers(COMPLETED_VOLUMES_KEY)
                completed = set(members)

            volumes = [v for v in VOLUMES if self.volume_filter is None or v["number"] in self.volume_filter]

            for vol in volumes:
                stats.volumes_seen += 1
                vol_key = f"vol{vol['number']}"
                if vol_key in completed:
                    stats.volumes_skipped += 1
                    logger.info("Skipping %s (already completed).", vol["title"])
                    continue

                try:
                    await self._ingest_volume(vol, stats)
                except Exception as exc:
                    stats.errors += 1
                    logger.warning("Error ingesting %s: %s", vol["title"], exc)
                    continue

                if not self.dry_run and self._redis:
                    await self._redis.sadd(COMPLETED_VOLUMES_KEY, vol_key)
                stats.volumes_processed += 1
                stats.log_progress()

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("Pentagon Papers ingestion complete.")
        finally:
            if self._http:
                await self._http.aclose()
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    # ------------------------------------------------------------------
    # Volume fetching
    # ------------------------------------------------------------------

    async def _ingest_volume(self, vol: dict, stats: IngestStats) -> None:
        logger.info("Fetching %s …", vol["title"])
        text, ia_id = await self._fetch_volume_text(vol)
        if not text or len(text) < 500:
            logger.warning("Could not fetch text for %s — skipping.", vol["title"])
            stats.errors += 1
            return

        text = clean_ocr_text(text)
        logger.info("Fetched %d chars (id=%s) — chunking …", len(text), ia_id or "unknown")
        chunks = chunk_text(text)
        if not chunks:
            return

        stats.chunks_produced += len(chunks)

        base_metadata: dict = {
            "source": SOURCE_LABEL,
            "document_type": "declassified_history",
            "provenance": "pentagon_papers_gravel_edition",
            "ingest_date": TODAY,
            "volume_number": vol["number"],
            "volume_title": vol["title"],
            "date": vol["date"],
            "ingested_at_unix": int(datetime(1971, 6, 13, tzinfo=UTC).timestamp()),
        }
        if ia_id:
            base_metadata["ia_identifier"] = ia_id

        for idx, chunk in enumerate(chunks):
            if len(chunk) < self.min_text_len:
                continue
            header = f"Source: {SOURCE_LABEL}\nVolume: {vol['title']}\nDate: {vol['date']}"
            text_with_header = f"{header}\n\n{chunk}"
            id_key = f"pentagon:{vol['number']}:chunk{idx}"
            point_id = str(uuid.UUID(bytes=hashlib.sha256(id_key.encode()).digest()[:16]))
            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload={"text": text_with_header, "chunk_index": idx, **base_metadata},
                )
            )
            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        await self._flush_upsert_buffer(stats)

        if self.enqueue_notable:
            await self._enqueue_volume(vol, stats)

    async def _fetch_volume_text(self, vol: dict) -> tuple[str, str]:
        """Try candidate archive.org identifiers, then fall back to search. Returns (text, identifier)."""
        for ia_id in vol["ia_candidates"]:
            for suffix in ("_djvu.txt", "_full.txt"):
                await asyncio.sleep(REQUEST_DELAY)
                url = f"https://archive.org/download/{ia_id}/{ia_id}{suffix}"
                text = await self._fetch_url(url)
                if text and len(text) > 500:
                    return text, ia_id

        # Search fallback
        await asyncio.sleep(REQUEST_DELAY)
        found_id = await self._search_archive_org(vol["search_query"])
        if found_id:
            for suffix in ("_djvu.txt", "_full.txt"):
                await asyncio.sleep(REQUEST_DELAY)
                url = f"https://archive.org/download/{found_id}/{found_id}{suffix}"
                text = await self._fetch_url(url)
                if text and len(text) > 500:
                    return text, found_id

        return "", ""

    async def _search_archive_org(self, query: str) -> str | None:
        assert self._http is not None
        url = "https://archive.org/advancedsearch.php"
        params = {
            "q": f"({query}) AND mediatype:texts",
            "fl": "identifier,title",
            "rows": "5",
            "output": "json",
        }
        for attempt in range(3):
            try:
                resp = await self._http.get(url, params=params)
                resp.raise_for_status()
                docs = resp.json().get("response", {}).get("docs", [])
                if docs:
                    return docs[0]["identifier"]
                return None
            except Exception as exc:
                wait = 10 * (attempt + 1)
                logger.debug("archive.org search attempt %d failed: %s — retry in %ds", attempt + 1, exc, wait)
                await asyncio.sleep(wait)
        return None

    async def _fetch_url(self, url: str) -> str:
        assert self._http is not None
        for attempt in range(3):
            try:
                resp = await self._http.get(url)
                # 404/403/503 on archive.org download URLs all mean the item
                # is unavailable or doesn't exist — skip without retrying.
                if resp.status_code in (403, 404, 503):
                    return ""
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(10 * (attempt + 1))
                else:
                    logger.debug("Failed to fetch %s: %s", url, exc)
        return ""

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _enqueue_volume(self, vol: dict, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:pentagon_papers:enqueued:vol{vol['number']}"
        if await self._redis.exists(redis_key):
            return
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": vol["title"],
                "desk": vol["desk_hint"],
                "priority": "low",
                "directives_lens": True,
                "triggered_by": "pentagon_papers_ingest",
                "metadata": {"volume_number": vol["number"], "source": SOURCE_LABEL},
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
        description="Ingest Pentagon Papers (Gravel Edition) into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--volumes", nargs="+", type=int, metavar="N", help="Volume numbers to ingest (default: all 1–5)")
    p.add_argument("--resume", action="store_true")
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

    logger.info(
        "Starting Pentagon Papers ingest | volumes=%s enqueue=%s dry_run=%s",
        args.volumes or "all",
        args.enqueue_notable,
        args.dry_run,
    )
    if args.dry_run:
        logger.warning("DRY RUN — no data will be written.")

    ingestor = PentagonPapersIngestor(args)
    asyncio.run(ingestor.run())


if __name__ == "__main__":
    main()
