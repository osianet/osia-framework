"""
OSIA USPTO Patent Data Ingestion

Fetches strategic technology patent records from the PatentsView API
(USPTO data) into the 'uspto-patents' Qdrant collection for the
Science, Technology & Commercial desk RAG retrieval.

Patent data reveals who is developing which technologies, maps corporate
R&D strategies, exposes dual-use research, and tracks the transfer of
technology between entities. Essential for assessing adversary
capabilities, identifying emerging threats, and monitoring the
militarisation of civilian technology.

Filtered to strategically relevant CPC sections by default:
  G — Physics (computing, AI, optics, quantum, instruments)
  H — Electricity (electronics, telecom, semiconductors, energy)
  C12 — Biochemistry / genetics / biotech
  B64 — Aerospace / aircraft / spacecraft
  A61 — Medical / pharmaceutical / biodefence

Source: https://api.patentsview.org/
Free public API, no authentication required. USPTO data 1976-present.

Usage:
  uv run python scripts/ingest_uspto.py
  uv run python scripts/ingest_uspto.py --dry-run
  uv run python scripts/ingest_uspto.py --resume
  uv run python scripts/ingest_uspto.py --date-from 2022-01-01
  uv run python scripts/ingest_uspto.py --enqueue-notable
  uv run python scripts/ingest_uspto.py --cpc-sections G H C12

Options:
  --dry-run             Parse and embed but skip Qdrant writes and Redis updates
  --resume              Resume from last Redis checkpoint (last processed patent date)
  --date-from YYYY-MM-DD  Start date (default: 2 years ago)
  --date-to YYYY-MM-DD    End date (default: today)
  --cpc-sections        CPC section codes to include (default: G H C12 B64 A61)
  --enqueue-notable     Push high-impact assignees to S&T research queue
  --limit N             Stop after N patents (0 = no limit)
  --embed-batch-size    Texts per HF embedding call (default: 32)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)

Environment variables (from .env):
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
from datetime import UTC, datetime, timedelta

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
logger = logging.getLogger("osia.uspto_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "uspto-patents"
EMBEDDING_DIM = 384
SOURCE_LABEL = "USPTO Patent Database (PatentsView)"

PATENTSVIEW_URL = "https://api.patentsview.org/patents/query"
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
PAGE_SIZE = 100
REQUEST_DELAY = 1.5

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

CHECKPOINT_KEY = "osia:uspto:last_date"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

DEFAULT_CPC_SECTIONS = ["G", "H", "C12", "B64", "A61"]

# Defence/strategic assignees for notable enqueue
NOTABLE_ASSIGNEE_KEYWORDS = {
    "lockheed", "raytheon", "northrop", "boeing", "bae systems",
    "general dynamics", "l3", "leidos", "saic", "mitre", "darpa",
    "department of defense", "navy", "air force", "army", "defense",
    "huawei", "zte", "hikvision", "dahua", "sensetime", "megvii",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP_WORDS = 60


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


def build_document(patent: dict) -> tuple[str, int | None]:
    """Build a narrative document from a PatentsView patent record."""
    patent_id = patent.get("patent_id", "")
    title = patent.get("patent_title", "")
    date = patent.get("patent_date", "")
    abstract = patent.get("patent_abstract", "")
    num_claims = patent.get("patent_num_claims", "")
    kind = patent.get("patent_kind", "")

    inventors = patent.get("inventors", []) or []
    assignees = patent.get("assignees", []) or []
    cpcs = patent.get("cpcs", []) or []

    patent_unix = _parse_date_unix(date)

    inventor_names = [
        f"{inv.get('inventor_first_name', '')} {inv.get('inventor_last_name', '')}".strip()
        for inv in inventors[:5]
    ]
    assignee_names = [
        a.get("assignee_organization", "") or
        f"{a.get('assignee_first_name', '')} {a.get('assignee_last_name', '')}".strip()
        for a in assignees[:5]
    ]
    assignee_names = [a for a in assignee_names if a]
    assignee_countries = list({a.get("assignee_country", "") for a in assignees if a.get("assignee_country")})

    cpc_groups = list({c.get("cpc_subgroup_id", "") for c in cpcs if c.get("cpc_subgroup_id")})[:8]
    cpc_sections = list({c.get("cpc_section_id", "") for c in cpcs if c.get("cpc_section_id")})

    if not title and not abstract:
        return "", patent_unix

    lines: list[str] = []
    if title:
        lines.append(f"Patent Title: {title}")
    if patent_id:
        lines.append(f"Patent Number: US{patent_id}")
        lines.append(f"USPTO Record: https://patents.google.com/patent/US{patent_id}")
    if date:
        lines.append(f"Grant Date: {date}")
    if kind:
        lines.append(f"Kind: {kind}")
    if num_claims:
        lines.append(f"Claims: {num_claims}")

    if assignee_names:
        lines.append(f"Assignee(s): {'; '.join(assignee_names)}")
    if assignee_countries:
        lines.append(f"Assignee Country: {', '.join(assignee_countries)}")
    if inventor_names:
        lines.append(f"Inventor(s): {'; '.join(inventor_names)}")
    if cpc_sections:
        lines.append(f"CPC Sections: {', '.join(sorted(cpc_sections))}")
    if cpc_groups:
        lines.append(f"CPC Groups: {', '.join(cpc_groups)}")

    if abstract:
        lines.append(f"\nAbstract:\n{abstract}")

    if len(lines) < 4:
        return "", patent_unix

    return "\n".join(lines), patent_unix


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


class UsptoIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.enqueue_notable: bool = args.enqueue_notable
        self.limit: int = args.limit
        self.cpc_sections: list[str] = args.cpc_sections
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.resume: bool = args.resume

        today_dt = datetime.now(UTC)
        self.date_to: str = args.date_to or TODAY
        self.date_from_default: str = args.date_from or (today_dt - timedelta(days=730)).strftime("%Y-%m-%d")

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=None)
        self._redis: aioredis.Redis | None = None
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    async def run(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()

            date_from = await self._resolve_date_from()
            logger.info("Fetching patents from %s to %s (CPC: %s)", date_from, self.date_to,
                        ", ".join(self.cpc_sections))

            stats = IngestStats()
            await self._ingest(stats, date_from)
            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("USPTO patent ingestion complete.")
        finally:
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    async def _resolve_date_from(self) -> str:
        if self.resume and self._redis:
            checkpoint = await self._redis.get(CHECKPOINT_KEY)
            if checkpoint:
                logger.info("Resuming from checkpoint: %s", checkpoint)
                return checkpoint
        return self.date_from_default

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

    async def _ingest(self, stats: IngestStats, date_from: str) -> None:
        # Build CPC filter: OR across all sections
        cpc_criteria = [
            {"cpc_subgroup_id": {"begins_with": section}}
            for section in self.cpc_sections
        ]
        cpc_filter = {"_or": cpc_criteria} if len(cpc_criteria) > 1 else cpc_criteria[0]

        query = {
            "_and": [
                {"patent_date": {"gte": date_from}},
                {"patent_date": {"lte": self.date_to}},
                cpc_filter,
            ]
        }

        fields = [
            "patent_id", "patent_title", "patent_date", "patent_abstract",
            "patent_num_claims", "patent_kind",
            "inventors.inventor_first_name", "inventors.inventor_last_name",
            "inventors.inventor_city", "inventors.inventor_country",
            "assignees.assignee_organization", "assignees.assignee_first_name",
            "assignees.assignee_last_name", "assignees.assignee_country",
            "cpcs.cpc_section_id", "cpcs.cpc_subgroup_id",
        ]

        page = 1
        latest_date = date_from

        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=60.0) as http:
            while True:
                payload = {
                    "q": query,
                    "f": fields,
                    "o": {"per_page": PAGE_SIZE, "page": page, "sort": [{"patent_date": "asc"}]},
                }

                for attempt in range(4):
                    try:
                        resp = await http.post(
                            PATENTSVIEW_URL,
                            json=payload,
                            headers={"Content-Type": "application/json"},
                        )
                        if resp.status_code == 429:
                            wait = 60 * (attempt + 1)
                            logger.warning("PatentsView 429 — waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        break
                    except Exception as exc:
                        logger.warning("PatentsView attempt %d failed (page=%d): %s", attempt + 1, page, exc)
                        await asyncio.sleep(10 * (attempt + 1))
                else:
                    logger.error("Giving up at page %d.", page)
                    break

                patents = data.get("patents") or []
                if not patents:
                    logger.info("No more patents at page %d — done.", page)
                    break

                total_patent_count = data.get("total_patent_count", "?")
                logger.info("Page %d: %d patents (total=%s)", page, len(patents), total_patent_count)

                for patent in patents:
                    stats.records_seen += 1
                    try:
                        patent_date = patent.get("patent_date", "")
                        if patent_date and patent_date > latest_date:
                            latest_date = patent_date
                        await self._process_patent(patent, stats)
                    except Exception as exc:
                        stats.errors += 1
                        logger.debug("Patent error: %s", exc)

                    if self.limit and stats.records_processed >= self.limit:
                        await self._save_checkpoint(latest_date)
                        return

                    if stats.records_processed % 1000 == 0 and stats.records_processed > 0:
                        stats.log_progress()

                await self._save_checkpoint(latest_date)

                if len(patents) < PAGE_SIZE:
                    break
                page += 1
                await asyncio.sleep(REQUEST_DELAY)

    async def _process_patent(self, patent: dict, stats: IngestStats) -> None:
        patent_id = patent.get("patent_id", "")
        doc, patent_unix = build_document(patent)
        if not doc.strip():
            stats.records_skipped += 1
            return

        chunks = chunk_text(doc)
        if not chunks:
            stats.records_skipped += 1
            return

        stats.records_processed += 1

        title = patent.get("patent_title", "")
        date = patent.get("patent_date", "")
        assignees = patent.get("assignees", []) or []
        assignee_names = [
            a.get("assignee_organization", "") or
            f"{a.get('assignee_first_name', '')} {a.get('assignee_last_name', '')}".strip()
            for a in assignees[:3]
        ]
        assignee_names = [a for a in assignee_names if a]
        cpcs = patent.get("cpcs", []) or []
        cpc_sections = list({c.get("cpc_section_id", "") for c in cpcs if c.get("cpc_section_id")})
        entity_tags = [t for t in [title] + assignee_names + cpc_sections if t]
        ingest_unix = patent_unix or int(time.time())

        for i, chunk in enumerate(chunks):
            point_id = str(uuid.UUID(bytes=hashlib.sha256(f"uspto:{patent_id}:{i}".encode()).digest()[:16]))
            payload: dict = {
                "text": chunk,
                "source": SOURCE_LABEL,
                "document_type": "patent",
                "provenance": "patentsview",
                "ingest_date": TODAY,
                "patent_id": patent_id,
                "pub_date": date,
                "entity_tags": entity_tags,
                "ingested_at_unix": ingest_unix,
            }
            if assignee_names:
                payload["assignees"] = assignee_names
            if cpc_sections:
                payload["cpc_sections"] = cpc_sections
            if len(chunks) > 1:
                payload["chunk_index"] = i
                payload["total_chunks"] = len(chunks)

            self._upsert_buffer.append(
                qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
            )
            stats.chunks_produced += 1

        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer(stats)

        if self.enqueue_notable:
            assignee_lower = " ".join(assignee_names).lower()
            if any(kw in assignee_lower for kw in NOTABLE_ASSIGNEE_KEYWORDS):
                await self._maybe_enqueue(patent_id, title, assignee_names, date, stats)

    async def _maybe_enqueue(
        self, patent_id: str, title: str, assignees: list[str], date: str, stats: IngestStats
    ) -> None:
        if not self._redis or self.dry_run:
            return
        redis_key = f"osia:uspto:enqueued:{patent_id}"
        if await self._redis.exists(redis_key):
            return
        assignee_str = "; ".join(assignees[:3])
        topic = f"USPTO strategic patent: {title} — filed by {assignee_str} ({date})"
        job = json.dumps({
            "job_id": str(uuid.uuid4()),
            "topic": topic,
            "desk": "science-technology-and-commercial-desk",
            "priority": "normal",
            "triggered_by": "uspto_ingest",
            "metadata": {"patent_id": patent_id, "assignees": assignees, "date": date},
        })
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 60)
        stats.events_enqueued += 1
        logger.debug("Enqueued strategic patent: %r", title[:80])

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

    async def _save_checkpoint(self, date: str) -> None:
        if self.dry_run or not self._redis:
            return
        await self._redis.set(CHECKPOINT_KEY, date)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest USPTO strategic patents into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Skip Qdrant writes and Redis updates")
    p.add_argument("--resume", action="store_true", help="Resume from last Redis checkpoint date")
    p.add_argument("--date-from", help="Start date YYYY-MM-DD (default: 2 years ago)")
    p.add_argument("--date-to", help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--cpc-sections", nargs="+", default=DEFAULT_CPC_SECTIONS,
                   help="CPC section codes to include")
    p.add_argument("--enqueue-notable", action="store_true",
                   help="Push strategic assignee patents to research queue")
    p.add_argument("--limit", type=int, default=0, help="Stop after N patents (0=no limit)")
    p.add_argument("--embed-batch-size", type=int, default=32, help="Texts per HF embedding call")
    p.add_argument("--embed-concurrency", type=int, default=3, help="Parallel embedding calls")
    p.add_argument("--upsert-batch-size", type=int, default=64, help="Points per Qdrant upsert call")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    ingestor = UsptoIngestor(args)
    asyncio.run(ingestor.run())
