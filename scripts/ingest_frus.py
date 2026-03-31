"""
OSIA FRUS (Foreign Relations of the United States) Ingestion

Downloads and ingests the complete Foreign Relations of the United States (FRUS)
series from the HistoryAtState/frus GitHub repository. FRUS is the US State
Department's official historical record of American foreign policy — thousands of
declassified cables, memos, NSC papers, and presidential correspondence from
1861 through the early 2000s.

Source: https://github.com/HistoryAtState/frus
  - TEI XML format, one XML file per volume
  - 300+ volumes covering 1861–2000
  - Documents: diplomatic cables, NSC memos, State Dept instructions,
    CIA assessments shared with State, presidential correspondence

Usage:
  uv run python scripts/ingest_frus.py
  uv run python scripts/ingest_frus.py --dry-run
  uv run python scripts/ingest_frus.py --limit-volumes 10
  uv run python scripts/ingest_frus.py --resume
  uv run python scripts/ingest_frus.py --year-from 1945 --year-to 1990
  uv run python scripts/ingest_frus.py --enqueue-notable

Options:
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --limit-volumes N     Stop after N volumes (0 = no limit)
  --resume              Skip volumes already in Redis completed set
  --year-from Y         Only ingest volumes whose start year >= Y
  --year-to Y           Only ingest volumes whose start year <= Y
  --enqueue-notable     Push notable documents to Geopolitical desk research queue
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a document body (default: 120)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required for embeddings)
  GITHUB_TOKEN          GitHub PAT (optional — raises API rate limit 60→5000/hr)
  QDRANT_URL            Qdrant URL (default: https://qdrant.osia.dev)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)

Rate limiting / polite scraping:
  - Sends User-Agent: OSIA-Framework/1.0 on all requests to GitHub
  - Waits DOWNLOAD_DELAY (1.0s) between each volume download to avoid
    hammering raw.githubusercontent.com across 300+ files
  - Exponential backoff (15s, 30s, 45s, 60s) on download failures
  - Persistent httpx.AsyncClient reused across all volume downloads
    (one connection pool for the full run, not a new connection per file)
  - GitHub API calls: 4 retries with 10s backoff; set GITHUB_TOKEN to
    raise unauthenticated rate limit from 60 to 5000 req/hr
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
import xml.etree.ElementTree as ET
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
logger = logging.getLogger("osia.frus_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

COLLECTION_NAME = "frus-state-dept"
EMBEDDING_DIM = 384
SOURCE_LABEL = "Foreign Relations of the United States (FRUS)"

GITHUB_API_BASE = "https://api.github.com"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
FRUS_REPO = "HistoryAtState/frus"
FRUS_BRANCH = "master"

TEI_NS = "http://www.tei-c.org/ns/1.0"
TEI = f"{{{TEI_NS}}}"
XML_NS = "http://www.w3.org/XML/1998/namespace"

# Redis keys
COMPLETED_VOLUMES_KEY = "osia:frus:completed_volumes"
RESEARCH_QUEUE_KEY = "osia:research_queue"

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# Chunking
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 225

# Classifications worth enqueueing for Geopolitical desk research
NOTABLE_CLASSIFICATIONS = {"TOP SECRET", "SECRET", "NODIS", "EYES ONLY", "LIMDIS"}

# Polite scraping — identify ourselves and pace downloads
USER_AGENT = "OSIA-Framework/1.0 (open-source intelligence research; +https://osia.dev)"
DOWNLOAD_DELAY = 1.0  # seconds between GitHub raw volume downloads


# ---------------------------------------------------------------------------
# Text utilities
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


# ---------------------------------------------------------------------------
# TEI XML parsing
# ---------------------------------------------------------------------------

_SKIP_TAGS = {f"{TEI}note", f"{TEI}fw", f"{TEI}figDesc"}
_BLOCK_TAGS = {f"{TEI}p", f"{TEI}ab", f"{TEI}lg", f"{TEI}list", f"{TEI}table"}


def _elem_text(elem: ET.Element) -> str:
    """Recursively collect all text from an element, skipping footnotes/figure descriptions."""
    parts: list[str] = []

    def _walk(e: ET.Element) -> None:
        if e.tag in _SKIP_TAGS:
            if e.tail:
                parts.append(e.tail.strip())
            return
        if e.text:
            parts.append(e.text.strip())
        for child in e:
            _walk(child)
        if e.tail:
            parts.append(e.tail.strip())

    _walk(elem)
    return " ".join(p for p in parts if p)


def _parse_volume_title(root: ET.Element) -> str:
    """Extract the volume title from teiHeader."""
    for xpath in [
        f".//{TEI}teiHeader//{TEI}titleStmt/{TEI}title[@type='complete']",
        f".//{TEI}teiHeader//{TEI}titleStmt/{TEI}title",
    ]:
        elem = root.find(xpath)
        if elem is not None:
            title = _elem_text(elem).strip()
            if title:
                return title
    return ""


def _find_document_divs(root: ET.Element) -> list[ET.Element]:
    """Return all <div type='document'> elements from the volume body."""
    body = root.find(f".//{TEI}body")
    if body is None:
        return []
    return [e for e in body.iter(f"{TEI}div") if e.get("type") == "document"]


def _parse_classification(doc_div: ET.Element) -> str:
    """Extract classification from the first short italic paragraph."""
    for p in doc_div.findall(f"{TEI}p"):
        rend = p.get("rend", "")
        text = _elem_text(p).strip()
        if (
            text
            and len(text) <= 120
            and (
                "italic" in rend
                or any(kw in text.upper() for kw in ("SECRET", "CONFIDENTIAL", "UNCLASSIFIED", "NODIS", "LIMDIS"))
            )
        ):
            return text
    return ""


def _build_document_body(doc_div: ET.Element) -> str:
    """
    Build the body text of a FRUS document by collecting all paragraph-level
    content, skipping the head/dateline/opener metadata elements.
    """
    skip_tags = {f"{TEI}head", f"{TEI}dateline"}
    parts: list[str] = []
    for child in doc_div:
        if child.tag in skip_tags:
            continue
        # For opener/closer, extract inline text
        text = _elem_text(child).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def parse_volume(xml_bytes: bytes, volume_filename: str) -> tuple[str, list[dict]]:
    """
    Parse a FRUS TEI XML volume.

    Returns (volume_title, list_of_document_dicts).
    Each document dict has: doc_n, doc_xml_id, doc_title, dateline,
    classification, body_text, volume_title.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("XML parse error in %s: %s", volume_filename, exc)
        return "", []

    volume_title = _parse_volume_title(root)
    documents: list[dict] = []

    for doc_div in _find_document_divs(root):
        doc_n = doc_div.get("n", "")
        doc_xml_id = doc_div.get(f"{{{XML_NS}}}id", "")

        head_elem = doc_div.find(f"{TEI}head")
        doc_title = _elem_text(head_elem).strip() if head_elem is not None else ""

        dateline_elem = doc_div.find(f"{TEI}dateline")
        dateline = _elem_text(dateline_elem).strip() if dateline_elem is not None else ""

        classification = _parse_classification(doc_div)
        body_text = _build_document_body(doc_div)

        documents.append(
            {
                "doc_n": doc_n,
                "doc_xml_id": doc_xml_id,
                "doc_title": doc_title,
                "dateline": dateline,
                "classification": classification,
                "body_text": body_text,
                "volume_title": volume_title,
            }
        )

    return volume_title, documents


# ---------------------------------------------------------------------------
# Year parsing
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"frus(\d{4})")


def volume_start_year(filename: str) -> int | None:
    """Extract the start year from a FRUS volume filename, e.g. frus1969-76v01.xml → 1969."""
    m = _YEAR_RE.search(filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    volumes_seen: int = 0
    volumes_skipped: int = 0
    volumes_processed: int = 0
    documents_seen: int = 0
    documents_skipped: int = 0
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
            "vols=%d processed=%d skipped=%d docs=%d chunks=%d upserted=%d enqueued=%d errors=%d elapsed=%s",
            self.volumes_seen,
            self.volumes_processed,
            self.volumes_skipped,
            self.documents_seen,
            self.chunks_produced,
            self.points_upserted,
            self.docs_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class FrusIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.limit_volumes: int = args.limit_volumes
        self.resume: bool = args.resume
        self.year_from: int | None = args.year_from
        self.year_to: int | None = args.year_to
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

        self._github_headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        }
        if GITHUB_TOKEN:
            self._github_headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self._http = httpx.AsyncClient(
            timeout=120.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        try:
            await self._ensure_collection()

            volume_files = await self._list_volumes()
            if not volume_files:
                logger.error("No volume files found — check GitHub API access.")
                return

            logger.info("Found %d volume files in HistoryAtState/frus.", len(volume_files))

            # Year filtering
            if self.year_from or self.year_to:
                before = len(volume_files)
                volume_files = [vf for vf in volume_files if self._year_in_range(vf["name"])]
                logger.info(
                    "Year filter (%s–%s): %d/%d volumes retained.",
                    self.year_from or "any",
                    self.year_to or "any",
                    len(volume_files),
                    before,
                )

            stats = IngestStats()
            await self._ingest_all(volume_files, stats)

            await self._flush_upsert_buffer(stats)
            stats.log_progress()
            logger.info("FRUS ingestion complete.")
        finally:
            if self._http:
                await self._http.aclose()
            await self._qdrant.close()
            if self._redis:
                await self._redis.aclose()

    def _year_in_range(self, filename: str) -> bool:
        year = volume_start_year(filename)
        if year is None:
            return True  # keep unknown
        if self.year_from and year < self.year_from:
            return False
        if self.year_to and year > self.year_to:
            return False
        return True

    # ------------------------------------------------------------------
    # Collection setup
    # ------------------------------------------------------------------

    async def _ensure_collection(self) -> None:
        if self.dry_run:
            logger.info("[dry-run] Skipping collection setup.")
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

    # ------------------------------------------------------------------
    # Volume listing (GitHub API)
    # ------------------------------------------------------------------

    async def _list_volumes(self) -> list[dict]:
        """
        List all XML files in the HistoryAtState/frus volumes/ directory.
        Returns list of dicts with 'name' and 'download_url'.
        """
        url = f"{GITHUB_API_BASE}/repos/{FRUS_REPO}/contents/volumes"
        assert self._http is not None
        for attempt in range(4):
            try:
                resp = await self._http.get(url, headers=self._github_headers)
                if resp.status_code == 403:
                    logger.warning(
                        "GitHub API 403 — rate limited or access denied. "
                        "Set GITHUB_TOKEN env var to raise limit to 5000/hr."
                    )
                    return []
                resp.raise_for_status()
                items = resp.json()
                return [
                    {"name": item["name"], "raw_url": item.get("download_url", "")}
                    for item in items
                    if isinstance(item, dict) and item.get("name", "").endswith(".xml")
                ]
            except Exception as exc:
                wait = 10 * (attempt + 1)
                logger.warning("GitHub API attempt %d failed: %s — retry in %ds", attempt + 1, exc, wait)
                await asyncio.sleep(wait)
        return []

    # ------------------------------------------------------------------
    # Ingestion loop
    # ------------------------------------------------------------------

    async def _ingest_all(self, volume_files: list[dict], stats: IngestStats) -> None:
        completed: set[str] = set()
        if self.resume and self._redis:
            members = await self._redis.smembers(COMPLETED_VOLUMES_KEY)
            completed = set(members)
            if completed:
                logger.info("Resuming — %d volumes already completed.", len(completed))

        for vf in volume_files:
            filename = vf["name"]
            stats.volumes_seen += 1

            if self.limit_volumes and stats.volumes_processed >= self.limit_volumes:
                logger.info("Reached --limit-volumes %d — stopping.", self.limit_volumes)
                break

            if filename in completed:
                stats.volumes_skipped += 1
                continue

            raw_url = vf.get("raw_url") or (f"{GITHUB_RAW_BASE}/{FRUS_REPO}/{FRUS_BRANCH}/volumes/{filename}")

            await asyncio.sleep(DOWNLOAD_DELAY)
            try:
                await self._ingest_volume(filename, raw_url, stats)
            except Exception as exc:
                stats.errors += 1
                logger.warning("Error ingesting volume %s: %s", filename, exc)
                continue

            if not self.dry_run and self._redis:
                await self._redis.sadd(COMPLETED_VOLUMES_KEY, filename)
            stats.volumes_processed += 1

            if stats.volumes_processed % 10 == 0:
                stats.log_progress()

    async def _ingest_volume(self, filename: str, raw_url: str, stats: IngestStats) -> None:
        logger.debug("Downloading volume %s", filename)
        xml_bytes = await self._download_volume(raw_url)
        if not xml_bytes:
            logger.warning("Empty download for %s — skipping.", filename)
            return

        volume_title, documents = parse_volume(xml_bytes, filename)
        if not documents:
            logger.debug("No documents parsed from %s.", filename)
            return

        logger.info(
            "Volume %s ('%s') — %d documents",
            filename,
            volume_title[:60] if volume_title else "unknown",
            len(documents),
        )

        for doc in documents:
            stats.documents_seen += 1
            body = doc["body_text"]
            if len(body) < self.min_text_len:
                stats.documents_skipped += 1
                continue

            await self._process_document(doc, filename, stats)

        await self._flush_upsert_buffer(stats)

    async def _process_document(self, doc: dict, volume_filename: str, stats: IngestStats) -> None:
        # Build the embeddable header block
        header_parts: list[str] = []
        if doc["volume_title"]:
            header_parts.append(f"Volume: {doc['volume_title']}")
        if doc["doc_n"]:
            header_parts.append(f"Document: {doc['doc_n']}")
        if doc["doc_title"]:
            header_parts.append(f"Title: {doc['doc_title']}")
        if doc["dateline"]:
            header_parts.append(f"Dateline: {doc['dateline']}")
        if doc["classification"]:
            header_parts.append(f"Classification: {doc['classification']}")
        header = "\n".join(header_parts)

        body_chunks = chunk_text(doc["body_text"])
        if not body_chunks:
            stats.documents_skipped += 1
            return

        stats.chunks_produced += len(body_chunks)

        # Stable base ID from volume + document identifiers
        id_key = f"frus:{volume_filename}:{doc['doc_xml_id'] or doc['doc_n']}"
        base_id = str(uuid.UUID(bytes=hashlib.sha256(id_key.encode()).digest()[:16]))

        base_metadata: dict = {
            "source": SOURCE_LABEL,
            "document_type": "diplomatic_document",
            "provenance": "us_state_department_frus",
            "ingest_date": TODAY,
            "volume_filename": volume_filename,
        }
        if doc["volume_title"]:
            base_metadata["volume_title"] = doc["volume_title"]
        if doc["doc_n"]:
            base_metadata["doc_number"] = doc["doc_n"]
        if doc["doc_title"]:
            base_metadata["doc_title"] = doc["doc_title"]
        if doc["dateline"]:
            base_metadata["dateline"] = doc["dateline"]
        if doc["classification"]:
            base_metadata["classification"] = doc["classification"]

        # Parse year from dateline for ingested_at_unix (approximate)
        year_match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", doc["dateline"])
        if year_match:
            approx_ts = int(datetime(int(year_match.group(1)), 1, 1, tzinfo=UTC).timestamp())
            base_metadata["ingested_at_unix"] = approx_ts

        chunk_count = len(body_chunks)
        base_metadata["chunk_count"] = chunk_count

        for idx, chunk in enumerate(body_chunks):
            text = f"{header}\n\n{chunk}" if header else chunk
            if chunk_count == 1:
                point_id = base_id
            else:
                suffix = f"{id_key}:chunk{idx}"
                point_id = str(uuid.UUID(bytes=hashlib.sha256(suffix.encode()).digest()[:16]))

            self._upsert_buffer.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=[0.0] * EMBEDDING_DIM,
                    payload={"text": text, "chunk_index": idx, **base_metadata},
                )
            )

            if len(self._upsert_buffer) >= self.upsert_batch_size:
                await self._flush_upsert_buffer(stats)

        # Optionally enqueue notable (highly classified) documents
        if self.enqueue_notable:
            classification_upper = doc["classification"].upper()
            if any(cls in classification_upper for cls in NOTABLE_CLASSIFICATIONS):
                await self._maybe_enqueue_document(doc, volume_filename, stats)

    # ------------------------------------------------------------------
    # Research queue
    # ------------------------------------------------------------------

    async def _maybe_enqueue_document(self, doc: dict, volume_filename: str, stats: IngestStats) -> None:
        if not self._redis or self.dry_run:
            return

        redis_key = f"osia:frus:enqueued:{volume_filename}:{doc['doc_n']}"
        if await self._redis.exists(redis_key):
            return

        topic = doc["doc_title"] or f"FRUS document {doc['doc_n']} from {volume_filename}"
        job = json.dumps(
            {
                "job_id": str(uuid.uuid4()),
                "topic": topic,
                "desk": "geopolitical-and-security-desk",
                "priority": "low",
                "directives_lens": True,
                "triggered_by": "frus_ingest",
                "metadata": {
                    "volume_filename": volume_filename,
                    "doc_number": doc["doc_n"],
                    "dateline": doc["dateline"],
                    "classification": doc["classification"],
                },
            }
        )
        await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
        await self._redis.set(redis_key, "1", ex=60 * 60 * 24 * 30)
        stats.docs_enqueued += 1
        logger.debug("Research job enqueued: %r → Geopolitical desk", topic[:80])

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
            logger.debug("Upserted %d points.", len(points))

        if stats is not None:
            stats.points_upserted += len(points)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    HF_EMBED_URL = (
        "https://router.huggingface.co/hf-inference/models/"
        "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    )

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
                            self.HF_EMBED_URL,
                            headers={"Authorization": f"Bearer {HF_TOKEN}"},
                            json={"inputs": texts, "options": {"wait_for_model": True}},
                        )
                        if resp.status_code == 429:
                            wait = 30 * (attempt + 1)
                            logger.warning("HF 429 — waiting %ds", wait)
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
    # Volume download
    # ------------------------------------------------------------------

    async def _download_volume(self, raw_url: str) -> bytes | None:
        assert self._http is not None
        for attempt in range(4):
            try:
                resp = await self._http.get(raw_url, headers=self._github_headers)
                if resp.status_code == 404:
                    logger.warning("Volume not found: %s", raw_url)
                    return None
                resp.raise_for_status()
                return resp.content
            except Exception as exc:
                wait = 15 * (attempt + 1)
                logger.warning("Download attempt %d failed (%s): %s", attempt + 1, raw_url, exc)
                await asyncio.sleep(wait)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest HistoryAtState/frus into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Parse and chunk but skip writes")
    p.add_argument(
        "--limit-volumes", type=int, default=0, dest="limit_volumes", help="Stop after N volumes (0=no limit)"
    )
    p.add_argument("--resume", action="store_true", help="Skip volumes already in Redis completed set")
    p.add_argument(
        "--year-from", type=int, default=None, dest="year_from", help="Only ingest volumes with start year >= Y"
    )
    p.add_argument("--year-to", type=int, default=None, dest="year_to", help="Only ingest volumes with start year <= Y")
    p.add_argument(
        "--enqueue-notable",
        action="store_true",
        dest="enqueue_notable",
        help="Enqueue highly classified documents to Geopolitical desk research queue",
    )
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
        "Starting FRUS ingest | limit_volumes=%s year_from=%s year_to=%s enqueue_notable=%s dry_run=%s",
        args.limit_volumes or "none",
        args.year_from or "any",
        args.year_to or "any",
        args.enqueue_notable,
        args.dry_run,
    )
    if args.dry_run:
        logger.warning("DRY RUN — no data will be written.")

    ingestor = FrusIngestor(args)
    asyncio.run(ingestor.run())


if __name__ == "__main__":
    main()
