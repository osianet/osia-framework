"""
OSIA FEC Campaign Finance Ingestion

Fetches US federal campaign finance records from the OpenFEC API into the
'fec-campaign-finance' Qdrant collection for Human Intelligence & Profiling
desk RAG retrieval.

FEC data is the primary public record of who funds US political campaigns —
individual donors, PACs, corporations, lobbyists, and foreign-adjacent entities.
Essential for profiling political actors, mapping donor networks, and tracing
the money behind policy decisions.

Ingests:
  - Federal candidates (office sought, party, state, total raised)
  - Individual contribution records (donor → committee, amount, employer)
  - Committees and PACs (type, designation, party affiliation)

Source: https://api.open.fec.gov/
Free API key from https://api.data.gov/. Covers 1979-present.

Usage:
  uv run python scripts/ingest_fec.py
  uv run python scripts/ingest_fec.py --dry-run
  uv run python scripts/ingest_fec.py --resume
  uv run python scripts/ingest_fec.py --cycle 2024
  uv run python scripts/ingest_fec.py --enqueue-notable
  uv run python scripts/ingest_fec.py --min-amount 5000

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Resume from last Redis checkpoint
  --cycle Y             Election cycle year (default: two most recent cycles)
  --min-amount N        Minimum contribution amount to ingest (default: 1000)
  --enqueue-notable     Push large donors / key committees to HUMINT research queue
  --limit N             Stop after N records (0 = no limit)
  --embed-batch-size    Texts per HF embedding call (default: 32)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)

Environment variables (from .env):
  FEC_API_KEY           OpenFEC API key (required; free from api.data.gov)
  HF_TOKEN              HuggingFace token (required for embeddings)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
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
logger = logging.getLogger("osia.fec_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEC_API_KEY = os.getenv("FEC_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "fec-campaign-finance"
EMBEDDING_DIM = 384
SOURCE_LABEL = "FEC Campaign Finance Records"

FEC_BASE = "https://api.open.fec.gov/v1"
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
PAGE_SIZE = 100
REQUEST_DELAY = 1.2  # FEC rate limit: 1000/hour for registered keys

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Redis checkpoint keys per cycle
CHECKPOINT_KEY_TEMPLATE = "osia:fec:last_page:{cycle}:{endpoint}"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")
CURRENT_YEAR = datetime.now(UTC).year
# FEC cycles are even years
DEFAULT_CYCLES = [
    CURRENT_YEAR if CURRENT_YEAR % 2 == 0 else CURRENT_YEAR + 1,
    (CURRENT_YEAR if CURRENT_YEAR % 2 == 0 else CURRENT_YEAR + 1) - 2,
]

# Large individual contribution threshold for notable enqueue
NOTABLE_AMOUNT = 50_000

CHUNK_SIZE = 400
CHUNK_OVERLAP_WORDS = 50


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) >= 60]


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------


def build_candidate_doc(rec: dict) -> str:
    """Build a text document from an FEC candidate record."""
    name = rec.get("name", "")
    office = rec.get("office_full", rec.get("office", ""))
    party = rec.get("party_full", rec.get("party", ""))
    state = rec.get("state", "")
    district = rec.get("district", "")
    cycle = rec.get("election_years", [])
    status = rec.get("candidate_status", "")
    incumbent = rec.get("incumbent_challenge_full", "")
    candidate_id = rec.get("candidate_id", "")

    lines: list[str] = []
    if name:
        lines.append(f"Candidate: {name}")
    if office:
        office_str = office
        if state:
            office_str += f" — {state}"
        if district and district != "00":
            office_str += f" District {district}"
        lines.append(f"Office: {office_str}")
    if party:
        lines.append(f"Party: {party}")
    if cycle:
        lines.append(f"Election Cycles: {', '.join(str(y) for y in sorted(cycle))}")
    if status:
        lines.append(f"Status: {status}")
    if incumbent:
        lines.append(f"Incumbent/Challenger: {incumbent}")
    if candidate_id:
        lines.append(f"FEC Candidate ID: {candidate_id}")
        lines.append(f"FEC Profile: https://www.fec.gov/data/candidate/{candidate_id}/")
    return "\n".join(lines)


def build_contribution_doc(rec: dict) -> str:
    """Build a text document from an FEC individual contribution record."""
    contributor = rec.get("contributor_name", "")
    amount = rec.get("contribution_receipt_amount")
    date = rec.get("contribution_receipt_date", "")
    employer = rec.get("contributor_employer", "")
    occupation = rec.get("contributor_occupation", "")
    city = rec.get("contributor_city", "")
    state = rec.get("contributor_state", "")
    zip_code = rec.get("contributor_zip", "")
    committee_name = rec.get("committee", {}).get("name", "") if rec.get("committee") else ""
    committee_id = rec.get("committee_id", "")
    candidate_name = ""
    if rec.get("candidate"):
        candidate_name = rec["candidate"].get("name", "")
    memo = rec.get("memo_text", "")
    receipt_type = rec.get("receipt_type_full", "")
    transaction_id = rec.get("transaction_id", "")

    lines: list[str] = []
    if contributor:
        lines.append(f"Contributor: {contributor}")
    if amount is not None:
        lines.append(f"Amount: ${float(amount):,.2f}")
    if date:
        lines.append(f"Date: {date[:10]}")
    if employer:
        lines.append(f"Employer: {employer}")
    if occupation:
        lines.append(f"Occupation: {occupation}")
    loc_parts = [p for p in [city, state, zip_code] if p]
    if loc_parts:
        lines.append(f"Location: {', '.join(loc_parts)}")
    if committee_name:
        lines.append(f"Recipient Committee: {committee_name}")
        if committee_id:
            lines.append(f"Committee ID: {committee_id}")
    if candidate_name:
        lines.append(f"Candidate: {candidate_name}")
    if receipt_type:
        lines.append(f"Receipt Type: {receipt_type}")
    if memo:
        lines.append(f"Memo: {memo}")
    if transaction_id:
        lines.append(f"Transaction ID: {transaction_id}")
    return "\n".join(lines)


def build_committee_doc(rec: dict) -> str:
    """Build a text document from an FEC committee record."""
    name = rec.get("name", "")
    committee_type = rec.get("committee_type_full", "")
    designation = rec.get("designation_full", "")
    party = rec.get("party_full", rec.get("party", ""))
    state = rec.get("state", "")
    treasurer = rec.get("treasurer_name", "")
    committee_id = rec.get("committee_id", "")
    cycles = rec.get("cycles", [])
    org_type = rec.get("organization_type_full", "")
    candidate_ids = rec.get("candidate_ids", [])

    lines: list[str] = []
    if name:
        lines.append(f"Committee: {name}")
    if committee_type:
        lines.append(f"Type: {committee_type}")
    if designation:
        lines.append(f"Designation: {designation}")
    if org_type:
        lines.append(f"Organisation Type: {org_type}")
    if party:
        lines.append(f"Party: {party}")
    if state:
        lines.append(f"State: {state}")
    if treasurer:
        lines.append(f"Treasurer: {treasurer}")
    if cycles:
        lines.append(f"Active Cycles: {', '.join(str(c) for c in sorted(cycles))}")
    if candidate_ids:
        lines.append(f"Associated Candidates: {', '.join(candidate_ids[:10])}")
    if committee_id:
        lines.append(f"FEC Committee ID: {committee_id}")
        lines.append(f"FEC Profile: https://www.fec.gov/data/committee/{committee_id}/")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    points_upserted: int = 0
    events_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "seen=%d processed=%d skipped=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.points_upserted,
            self.events_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class FecIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.limit: int = args.limit
        self.cycles: list[int] = [args.cycle] if args.cycle else DEFAULT_CYCLES
        self.min_amount: float = args.min_amount
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.resume: bool = args.resume

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []
        self._http: httpx.AsyncClient | None = None

    async def run(self) -> None:
        if not FEC_API_KEY:
            logger.error("FEC_API_KEY must be set in .env (get free key at api.data.gov)")
            return

        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self._http = httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30.0)

        try:
            await self._ensure_collection()
            stats = IngestStats()

            for cycle in self.cycles:
                logger.info("Ingesting FEC cycle %d...", cycle)
                await self._ingest_candidates(cycle, stats)
                await self._ingest_committees(cycle, stats)
                await self._ingest_contributions(cycle, stats)
                stats.log_progress()

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("FEC campaign finance ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()
            if self._http:
                await self._http.aclose()

    async def _ensure_collection(self) -> None:
        if self.dry_run:
            return
        exists = await self._qdrant.collection_exists(COLLECTION_NAME)
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

    async def _fec_get(self, endpoint: str, params: dict) -> dict:
        """GET from FEC API with retry and rate-limit handling."""
        params["api_key"] = FEC_API_KEY
        for attempt in range(4):
            try:
                resp = await self._http.get(f"{FEC_BASE}{endpoint}", params=params)
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning("FEC 429 — waiting %ds", wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.warning("FEC fetch attempt %d failed (%s): %s", attempt + 1, endpoint, exc)
                await asyncio.sleep(10 * (attempt + 1))
        return {}

    async def _ingest_candidates(self, cycle: int, stats: IngestStats) -> None:
        ck = CHECKPOINT_KEY_TEMPLATE.format(cycle=cycle, endpoint="candidates")
        start_page = 1
        if self.resume and self._redis:
            val = await self._redis.get(ck)
            if val:
                start_page = int(val)
                logger.info("Candidates cycle %d: resuming from page %d", cycle, start_page)

        page = start_page
        while True:
            data = await self._fec_get(
                "/candidates/",
                {
                    "election_year": cycle,
                    "per_page": PAGE_SIZE,
                    "page": page,
                    "sort": "name",
                },
            )
            results = data.get("results", [])
            if not results:
                break

            for rec in results:
                stats.records_seen += 1
                doc = build_candidate_doc(rec)
                if not doc.strip():
                    stats.records_skipped += 1
                    continue
                candidate_id = rec.get("candidate_id", "")
                entity_tags = [
                    t
                    for t in [
                        rec.get("name", ""),
                        rec.get("party_full", ""),
                        rec.get("state", ""),
                        rec.get("office_full", ""),
                    ]
                    if t
                ]
                await self._buffer_point(
                    f"fec:candidate:{candidate_id}:{cycle}",
                    doc,
                    {
                        "source": SOURCE_LABEL,
                        "document_type": "fec_candidate",
                        "provenance": "fec_api",
                        "ingest_date": TODAY,
                        "cycle": cycle,
                        "candidate_id": candidate_id,
                        "pub_date": str(cycle),
                        "entity_tags": entity_tags,
                        "ingested_at_unix": int(time.time()),
                    },
                    stats,
                )

            if self._redis and not self.dry_run:
                await self._redis.set(ck, page + 1)

            pagination = data.get("pagination", {})
            if page >= pagination.get("pages", 1):
                break
            page += 1
            await asyncio.sleep(REQUEST_DELAY)

            if self.limit and stats.records_processed >= self.limit:
                return

        logger.info("Candidates cycle %d: done.", cycle)

    async def _ingest_committees(self, cycle: int, stats: IngestStats) -> None:
        ck = CHECKPOINT_KEY_TEMPLATE.format(cycle=cycle, endpoint="committees")
        start_page = 1
        if self.resume and self._redis:
            val = await self._redis.get(ck)
            if val:
                start_page = int(val)

        page = start_page
        while True:
            data = await self._fec_get(
                "/committees/",
                {
                    "cycle": cycle,
                    "per_page": PAGE_SIZE,
                    "page": page,
                    "sort": "name",
                },
            )
            results = data.get("results", [])
            if not results:
                break

            for rec in results:
                stats.records_seen += 1
                doc = build_committee_doc(rec)
                if not doc.strip():
                    stats.records_skipped += 1
                    continue
                committee_id = rec.get("committee_id", "")
                entity_tags = [
                    t
                    for t in [
                        rec.get("name", ""),
                        rec.get("party_full", ""),
                        rec.get("state", ""),
                        rec.get("committee_type_full", ""),
                    ]
                    if t
                ]
                await self._buffer_point(
                    f"fec:committee:{committee_id}:{cycle}",
                    doc,
                    {
                        "source": SOURCE_LABEL,
                        "document_type": "fec_committee",
                        "provenance": "fec_api",
                        "ingest_date": TODAY,
                        "cycle": cycle,
                        "committee_id": committee_id,
                        "pub_date": str(cycle),
                        "entity_tags": entity_tags,
                        "ingested_at_unix": int(time.time()),
                    },
                    stats,
                )

            if self._redis and not self.dry_run:
                await self._redis.set(ck, page + 1)

            pagination = data.get("pagination", {})
            if page >= pagination.get("pages", 1):
                break
            page += 1
            await asyncio.sleep(REQUEST_DELAY)

            if self.limit and stats.records_processed >= self.limit:
                return

        logger.info("Committees cycle %d: done.", cycle)

    async def _ingest_contributions(self, cycle: int, stats: IngestStats) -> None:
        ck = CHECKPOINT_KEY_TEMPLATE.format(cycle=cycle, endpoint="contributions")
        start_page = 1
        if self.resume and self._redis:
            val = await self._redis.get(ck)
            if val:
                start_page = int(val)
                logger.info("Contributions cycle %d: resuming from page %d", cycle, start_page)

        page = start_page
        while True:
            data = await self._fec_get(
                "/schedules/schedule_a/",
                {
                    "two_year_transaction_period": cycle,
                    "per_page": PAGE_SIZE,
                    "page": page,
                    "min_amount": int(self.min_amount),
                    "sort": "contribution_receipt_date",
                    "sort_nulls_last": True,
                },
            )
            results = data.get("results", [])
            if not results:
                break

            for rec in results:
                stats.records_seen += 1
                try:
                    amount = float(rec.get("contribution_receipt_amount") or 0)
                except (ValueError, TypeError):
                    amount = 0.0
                if amount < self.min_amount:
                    stats.records_skipped += 1
                    continue

                doc = build_contribution_doc(rec)
                if not doc.strip():
                    stats.records_skipped += 1
                    continue

                transaction_id = rec.get("transaction_id", str(uuid.uuid4()))
                contributor = rec.get("contributor_name", "")
                employer = rec.get("contributor_employer", "")
                committee_name = (rec.get("committee") or {}).get("name", "")
                date_str = (rec.get("contribution_receipt_date") or "")[:10]

                entity_tags = [t for t in [contributor, employer, committee_name] if t]

                # Parse date for ingested_at_unix
                ingest_unix = int(time.time())
                if date_str:
                    try:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                        ingest_unix = int(dt.replace(tzinfo=UTC).timestamp())
                    except ValueError:
                        pass  # malformed date in FEC source data — fall back to ingest time

                await self._buffer_point(
                    f"fec:contrib:{transaction_id}",
                    doc,
                    {
                        "source": SOURCE_LABEL,
                        "document_type": "fec_contribution",
                        "provenance": "fec_api",
                        "ingest_date": TODAY,
                        "cycle": cycle,
                        "amount_usd": amount,
                        "pub_date": date_str,
                        "entity_tags": entity_tags,
                        "ingested_at_unix": ingest_unix,
                    },
                    stats,
                )

                if self.enqueue_notable and amount >= NOTABLE_AMOUNT:
                    await self._maybe_enqueue_contribution(rec, amount, stats)

            if self._redis and not self.dry_run:
                await self._redis.set(ck, page + 1)

            pagination = data.get("pagination", {})
            if page >= pagination.get("pages", 1):
                break
            page += 1
            await asyncio.sleep(REQUEST_DELAY)

            if self.limit and stats.records_processed >= self.limit:
                return

        logger.info("Contributions cycle %d: done.", cycle)

    async def _maybe_enqueue_contribution(self, rec: dict, amount: float, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return
        transaction_id = rec.get("transaction_id", "")
        redis_key = f"osia:fec:enqueued:{transaction_id}"
        if await self._redis.exists(redis_key):
            return
        contributor = rec.get("contributor_name", "Unknown donor")
        employer = rec.get("contributor_employer", "")
        committee_name = (rec.get("committee") or {}).get("name", "")
        date_str = (rec.get("contribution_receipt_date") or "")[:10]
        topic = f"FEC large contribution: {contributor} donated ${amount:,.0f} to {committee_name} ({date_str})"
        if employer:
            topic += f" — employer: {employer}"
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "human-intelligence-and-profiling-desk",
                "priority": "normal",
                "triggered_by": "fec_ingest",
                "metadata": {
                    "transaction_id": transaction_id,
                    "contributor": contributor,
                    "amount_usd": amount,
                    "committee": committee_name,
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 60)
        stats.events_enqueued += 1

    async def _buffer_point(self, key: str, doc: str, payload: dict, stats: IngestStats) -> None:
        chunks = chunk_text(doc)
        if not chunks:
            stats.records_skipped += 1
            return
        stats.records_processed += 1
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.UUID(bytes=hashlib.sha256(f"{key}:{i}".encode()).digest()[:16]))
            p = dict(payload)
            p["text"] = chunk
            if len(chunks) > 1:
                p["chunk_index"] = i
                p["total_chunks"] = len(chunks)
            self._upsert_buffer.append(qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=p))
        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

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
        description="Ingest FEC campaign finance records into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Resume from last Redis checkpoint")
    p.add_argument("--cycle", type=int, default=0, help="Election cycle year (0 = two most recent)")
    p.add_argument("--min-amount", type=float, default=1000.0, help="Minimum contribution amount to ingest")
    p.add_argument("--enqueue-notable", action="store_true", help="Push large donors to research queue")
    p.add_argument("--limit", type=int, default=0, help="Stop after N records (0=no limit)")
    p.add_argument("--embed-batch-size", type=int, default=32, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=64, help="Points per Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = FecIngestor(args)
    asyncio.run(ingestor.run())
