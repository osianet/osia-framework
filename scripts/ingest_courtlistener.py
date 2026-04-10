"""
OSIA CourtListener Federal Court Opinions Ingestion

Fetches US federal court opinions from the CourtListener API (PACER data)
into the 'courtlistener-cases' Qdrant collection for Human Intelligence
& Profiling desk RAG retrieval.

Federal court records are primary-source intelligence on: prosecutions of
political actors and corporate criminals, national security proceedings,
civil rights enforcement, immigration and asylum rulings, and corporate
litigation revealing internal misconduct. CourtListener covers SCOTUS,
all federal circuits, and district courts.

Source: https://www.courtlistener.com/api/rest/v4/
Free REST API. Optional API token for higher rate limits (free registration).
Covers opinions from the late 1700s to present.

Usage:
  uv run python scripts/ingest_courtlistener.py
  uv run python scripts/ingest_courtlistener.py --dry-run
  uv run python scripts/ingest_courtlistener.py --resume
  uv run python scripts/ingest_courtlistener.py --date-from 2023-01-01
  uv run python scripts/ingest_courtlistener.py --enqueue-notable
  uv run python scripts/ingest_courtlistener.py --court-types f  (federal only)

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Resume from last Redis checkpoint (last opinion ID)
  --date-from YYYY-MM-DD  Start date (default: 2 years ago)
  --date-to YYYY-MM-DD    End date (default: today)
  --court-ids           Specific court slugs (e.g. scotus ca9 ca2). Default: all federal
  --enqueue-notable     Push high-profile cases to HUMINT research queue
  --limit N             Stop after N opinions (0 = no limit)
  --embed-batch-size    Texts per HF embedding call (default: 24)
  --embed-concurrency   Parallel embedding calls (default: 2)
  --upsert-batch-size   Points per Qdrant upsert call (default: 32)

Environment variables (from .env):
  COURTLISTENER_API_KEY  CourtListener API token (required; free at courtlistener.com)
  HF_TOKEN               HuggingFace token (required for embeddings)
  QDRANT_URL             Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY         Qdrant API key
  REDIS_URL              Redis URL (default: redis://localhost:6379)
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
from datetime import UTC, datetime, timedelta
from urllib.parse import unquote

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
logger = logging.getLogger("osia.courtlistener_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY", "") or None
HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "courtlistener-cases"
EMBEDDING_DIM = 384
SOURCE_LABEL = "CourtListener Federal Court Opinions"

CL_BASE = "https://www.courtlistener.com/api/rest/v4"
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
PAGE_SIZE = 20  # opinions can be large; keep batches small
REQUEST_DELAY = 2.0  # CourtListener is strict about rate limits

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Per-court checkpoint: osia:courtlistener:last_id:{court_id} → last ingested opinion ID
CHECKPOINT_KEY_PREFIX = "osia:courtlistener:last_id"
# Fields to fetch from cluster endpoint (docket excluded — separate URL, too expensive)
CLUSTER_FIELDS = "case_name,date_filed,judges,precedential_status,citation_count"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Court slugs for high-value federal courts
HIGH_VALUE_COURTS = {
    "scotus",           # Supreme Court
    "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11",
    "cadc", "cafc",     # DC Circuit, Federal Circuit
}

# Keywords in case name suggesting high HUMINT value
NOTABLE_CASE_KEYWORDS = {
    "united states v.", "conspiracy", "espionage", "terrorism", "bribery",
    "corruption", "fraud", "rico", "trafficking", "assassination", "treason",
    "sanctions", "laundering", "surveillance", "wiretap", "classified",
    "national security", "cia", "nsa", "fbi", "doj", "sec v.", "ftc v.",
}

CHUNK_SIZE = 600
CHUNK_OVERLAP_WORDS = 80
MAX_TEXT_WORDS = 3000  # cap opinion length before chunking to avoid very long docs


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    words = text.split()
    # Truncate very long opinions to keep embedding costs reasonable
    if len(words) > MAX_TEXT_WORDS:
        words = words[:MAX_TEXT_WORDS]
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) >= 80]


def _parse_date_unix(date_str: str) -> int | None:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return int(dt.replace(tzinfo=UTC).timestamp())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


def build_document(opinion: dict, cluster: dict, court_id: str = "") -> tuple[str, int | None]:
    """Build a narrative document from a CourtListener opinion + cluster."""
    case_name = cluster.get("case_name", "")
    date_filed = cluster.get("date_filed", "")
    # court_id passed explicitly — cluster.docket is a URL, not fetched
    judges = cluster.get("judges", "")
    precedential = cluster.get("precedential_status", "")
    citations = cluster.get("citation_count", 0)

    opinion_type = opinion.get("type", "")
    opinion_id = str(opinion.get("id", ""))

    # Get opinion text: prefer plain_text, fall back to html
    text = opinion.get("plain_text", "") or ""
    if not text:
        text = _strip_html(opinion.get("html", "") or "")

    filed_unix = _parse_date_unix(date_filed)

    if not case_name and not text:
        return "", filed_unix

    lines: list[str] = []
    if case_name:
        lines.append(f"Case: {case_name}")
    if date_filed:
        lines.append(f"Date Filed: {date_filed}")
    if court_id:
        lines.append(f"Court: {court_id.upper()}")
    if opinion_type:
        lines.append(f"Opinion Type: {opinion_type}")
    if precedential:
        lines.append(f"Precedential Status: {precedential}")
    if judges:
        lines.append(f"Judge(s): {judges}")
    if citations:
        lines.append(f"Times Cited: {citations}")
    if opinion_id:
        lines.append(f"Source: https://www.courtlistener.com/opinion/{opinion_id}/")

    if text:
        lines.append(f"\nOpinion Text:\n{text}")

    if len(lines) < 3:
        return "", filed_unix

    return "\n".join(lines), filed_unix


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    chunks_produced: int = 0
    points_upserted: int = 0
    events_enqueued: int = 0
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
            self.events_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class CourtListenerIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.limit: int = args.limit
        self.court_ids: list[str] = args.court_ids or []
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.resume: bool = args.resume

        today_dt = datetime.now(UTC)
        self.date_to: str = args.date_to or TODAY
        self.date_from_default: str = args.date_from or (today_dt - timedelta(days=730)).strftime("%Y-%m-%d")

        headers = {"User-Agent": USER_AGENT}
        if COURTLISTENER_API_KEY:
            headers["Authorization"] = f"Token {COURTLISTENER_API_KEY}"
        self._headers = headers

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self) -> None:
        if not COURTLISTENER_API_KEY:
            raise RuntimeError(
                "COURTLISTENER_API_KEY is not set — free token required. "
                "Register at https://www.courtlistener.com/sign-in/ and add to .env"
            )
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()
            logger.info("Using authenticated CourtListener API key.")
            courts = self.court_ids or sorted(HIGH_VALUE_COURTS)
            logger.info("Ingesting %d courts: %s", len(courts), ", ".join(courts))

            stats = IngestStats()
            await self._ingest(stats)
            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("CourtListener ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    async def _resolve_start_id(self, court_id: str) -> int:
        """Return the last ingested opinion ID for this court (0 = start from beginning)."""
        if self.resume and self._redis:
            val = await self._redis.get(f"{CHECKPOINT_KEY_PREFIX}:{court_id}")
            if val:
                logger.info("[%s] Resuming from opinion ID %s", court_id.upper(), val)
                return int(val)
        return 0

    async def _save_court_checkpoint(self, court_id: str, last_id: int) -> None:
        if self.dry_run or not self._redis or not last_id:
            return
        await self._redis.set(f"{CHECKPOINT_KEY_PREFIX}:{court_id}", str(last_id))

    async def _ensure_collection(self) -> None:
        if self.dry_run:
            return
        for attempt in range(5):
            try:
                exists = await self._qdrant.collection_exists(COLLECTION_NAME)
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 15 * (attempt + 1)
                logger.warning("Qdrant not ready (attempt %d/5): %s — retrying in %ds", attempt + 1, exc, wait)
                await asyncio.sleep(wait)
        if not exists:
            await self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(indexing_threshold=1000),
            )
            logger.info("Created Qdrant collection '%s'.", COLLECTION_NAME)
        else:
            info = await self._qdrant.get_collection(COLLECTION_NAME)
            logger.info("Collection '%s' ready (%d points).", COLLECTION_NAME, info.points_count or 0)

    async def _ingest(self, stats: IngestStats) -> None:
        courts = self.court_ids or sorted(HIGH_VALUE_COURTS)
        async with httpx.AsyncClient(headers=self._headers, timeout=60.0) as http:
            for court_id in courts:
                logger.info("--- Court: %s ---", court_id.upper())
                await self._ingest_court(http, court_id, stats)
                if self.limit and stats.records_processed >= self.limit:
                    return

    async def _ingest_court(self, http: httpx.AsyncClient, court_id: str, stats: IngestStats) -> None:
        start_id = await self._resolve_start_id(court_id)
        last_id = start_id
        cursor = None
        page = 1

        while True:
            params: dict = {
                "cluster__docket__court": court_id,
                "order_by": "id",
                "page_size": PAGE_SIZE,
            }
            if start_id:
                params["id__gte"] = start_id
            if cursor:
                params["cursor"] = cursor

            for attempt in range(4):
                try:
                    resp = await http.get(f"{CL_BASE}/opinions/", params=params)
                    if resp.status_code == 401:
                        raise RuntimeError(
                            "CourtListener API returned 401 — set COURTLISTENER_API_KEY in .env "
                            "(free token at https://www.courtlistener.com/sign-in/)"
                        )
                    if resp.status_code == 429:
                        wait = 60 * (attempt + 1)
                        logger.warning("[%s] 429 — waiting %ds", court_id, wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except RuntimeError:
                    raise
                except Exception as exc:
                    logger.warning("[%s] Attempt %d failed (page=%d): %s", court_id, attempt + 1, page, exc)
                    await asyncio.sleep(15 * (attempt + 1))
            else:
                logger.error("[%s] Giving up at page %d.", court_id, page)
                break

            results = data.get("results", [])
            if not results:
                logger.info("[%s] No more opinions at page %d — done.", court_id.upper(), page)
                break

            logger.info("[%s] Page %d: %d opinions", court_id.upper(), page, len(results))

            # Fetch cluster metadata concurrently for the whole page
            cluster_urls = [op.get("cluster", "") for op in results if isinstance(op.get("cluster"), str)]
            clusters_fetched = await asyncio.gather(*[self._fetch_cluster(http, url) for url in cluster_urls])
            cluster_map = {url: cl for url, cl in zip(cluster_urls, clusters_fetched)}

            for opinion in results:
                stats.records_seen += 1
                opinion_id = int(opinion.get("id", 0))
                if opinion_id > last_id:
                    last_id = opinion_id
                try:
                    cluster = cluster_map.get(opinion.get("cluster", ""), {})
                    await self._process_opinion(opinion, cluster, court_id, stats)
                except Exception as exc:
                    stats.errors += 1
                    logger.debug("[%s] Opinion error: %s", court_id, exc)

                if self.limit and stats.records_processed >= self.limit:
                    await self._save_court_checkpoint(court_id, last_id)
                    return

                if stats.records_processed % 200 == 0 and stats.records_processed > 0:
                    stats.log_progress()

            await self._save_court_checkpoint(court_id, last_id)

            next_url = data.get("next")
            if not next_url:
                break

            m = re.search(r"cursor=([^&]+)", next_url)
            # Decode once — httpx will re-encode when building the request URL
            cursor = unquote(m.group(1)) if m else None
            if not cursor:
                break

            page += 1
            await asyncio.sleep(REQUEST_DELAY)

    async def _fetch_cluster(self, http: httpx.AsyncClient, url: str) -> dict:
        """Fetch cluster metadata (case name, date filed, judges, etc.) by URL."""
        if not url:
            return {}
        fetch_url = url if url.endswith("/") else url + "/"
        fetch_url += f"?fields={CLUSTER_FIELDS}"
        for attempt in range(3):
            try:
                resp = await http.get(fetch_url)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.debug("Cluster fetch attempt %d failed (%s): %s", attempt + 1, url, exc)
                await asyncio.sleep(5 * (attempt + 1))
        return {}

    async def _process_opinion(self, opinion: dict, cluster: dict, court_id: str, stats: IngestStats) -> None:
        opinion_id = str(opinion.get("id", ""))
        case_name = cluster.get("case_name", "")

        doc, filed_unix = build_document(opinion, cluster, court_id)
        if not doc.strip():
            stats.records_skipped += 1
            return

        chunks = chunk_text(doc)
        if not chunks:
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        date_filed = cluster.get("date_filed", "")
        judges = cluster.get("judges", "")
        entity_tags = [t for t in [case_name, court_id, judges] if t]
        ingest_unix = filed_unix or int(time.time())

        for i, chunk in enumerate(chunks):
            point_id = str(uuid.UUID(bytes=hashlib.sha256(f"cl:{opinion_id}:{i}".encode()).digest()[:16]))
            payload: dict = {
                "text": chunk,
                "source": SOURCE_LABEL,
                "document_type": "court_opinion",
                "provenance": "courtlistener",
                "ingest_date": TODAY,
                "opinion_id": opinion_id,
                "case_name": case_name,
                "court_id": court_id,
                "pub_date": date_filed,
                "entity_tags": entity_tags,
                "ingested_at_unix": ingest_unix,
            }
            if judges:
                payload["judges"] = judges
            if len(chunks) > 1:
                payload["chunk_index"] = i
                payload["total_chunks"] = len(chunks)

            self._upsert_buffer.append(
                qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
            )
            stats.chunks_produced += 1

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        if self.enqueue_notable and self._is_notable(case_name, court_id, cluster):
            await self._maybe_enqueue(opinion_id, case_name, court_id, date_filed, cluster, stats)

    def _is_notable(self, case_name: str, court_id: str, cluster: dict) -> bool:
        name_lower = case_name.lower()
        if any(kw in name_lower for kw in NOTABLE_CASE_KEYWORDS):
            return True
        if court_id == "scotus":
            return True
        citations = cluster.get("citation_count", 0) or 0
        return citations >= 50

    async def _maybe_enqueue(
        self, opinion_id: str, case_name: str, court_id: str, date_filed: str,
        cluster: dict, stats: IngestStats
    ) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:cl:enqueued:{opinion_id}"
        if await self._redis.exists(redis_key):
            return
        citations = cluster.get("citation_count", 0) or 0
        topic = f"CourtListener notable case: {case_name} ({court_id.upper()}, {date_filed})"
        if citations:
            topic += f" — {citations} citations"
        job = json.dumps({
            "job_id": str(uuid.uuid4()),
            "topic": topic,
            "desk": "human-intelligence-and-profiling-desk",
            "priority": "normal",
            "triggered_by": "courtlistener_ingest",
            "metadata": {
                "opinion_id": opinion_id,
                "case_name": case_name,
                "court_id": court_id,
                "date_filed": date_filed,
            },
        })
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 60)
        stats.events_enqueued += 1
        logger.debug("Enqueued notable case: %r", case_name[:80])

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



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest CourtListener federal court opinions into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Resume from last Redis checkpoint date")
    p.add_argument("--date-from", help="Start date YYYY-MM-DD (default: 2 years ago)")
    p.add_argument("--date-to", help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--court-ids", nargs="+", help="Court slugs to include (default: all federal)")
    p.add_argument("--enqueue-notable", action="store_true",
                   help="Push notable cases to HUMINT research queue")
    p.add_argument("--limit", type=int, default=0, help="Stop after N opinions (0=no limit)")
    p.add_argument("--embed-batch-size", type=int, default=24, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=2, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=32, help="Points per Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = CourtListenerIngestor(args)
    asyncio.run(ingestor.run())
