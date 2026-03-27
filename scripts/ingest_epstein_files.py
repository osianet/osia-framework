"""
OSIA Epstein Files Ingestion

Streams HuggingFace Epstein datasets, chunks documents, embeds via the HF
Inference API, and upserts into a dedicated 'epstein-files' Qdrant collection.
All four public datasets are supported; entity extraction enqueues HUMINT
research jobs for novel Person entities found in each document.

Supported datasets:
  emails      notesbymuneeb/epstein-emails       (5K structured email threads)
  oversight   tensonaut/EPSTEIN_FILES_20K         (25K House Oversight docs)
  index       theelderemo/FULL_EPSTEIN_INDEX      (8.5K indexed docs, OCR)
  nikity      Nikity/Epstein-Files                (4.11M rows, full DOJ library)
  all         Run emails → oversight → index → nikity in order

Usage:
  uv run python scripts/ingest_epstein_files.py --dataset emails
  uv run python scripts/ingest_epstein_files.py --dataset nikity --limit 50000
  uv run python scripts/ingest_epstein_files.py --dataset all --entity-sample-rate 0.05
  uv run python scripts/ingest_epstein_files.py --dataset nikity --resume --dry-run

Options:
  --dataset             Dataset(s) to ingest (default: emails)
  --limit N             Stop after N source records (0 = no limit)
  --entity-sample-rate  Fraction of docs to run full NER on (default: 0.15)
  --skip-entity         Disable entity extraction entirely
  --resume              Resume from last checkpoint stored in Redis
  --dry-run             Parse and chunk but skip Qdrant writes and Redis updates
  --embed-batch-size    Texts per HF embedding call (default: 48)
  --embed-concurrency   Parallel embedding calls (default: 3)
  --upsert-batch-size   Points per Qdrant upsert call (default: 64)
  --min-text-len        Minimum chars for a record to be processed (default: 120)

Environment variables (from .env):
  HF_TOKEN              HuggingFace token (required — for dataset access + embeddings)
  QDRANT_URL            Qdrant URL (default: http://localhost:6333)
  QDRANT_API_KEY        Qdrant API key
  REDIS_URL             Redis URL (default: redis://localhost:6379)
  VENICE_API_KEY        Venice AI API key (for entity extraction via venice-uncensored)
  VENICE_MODEL_UNCENSORED  Venice model for NER (default: venice-uncensored)
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
logger = logging.getLogger("osia.epstein_ingest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_NER_MODEL = os.getenv("VENICE_MODEL_UNCENSORED", "venice-uncensored")

COLLECTION_NAME = "epstein-files"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Entity name validation
# ---------------------------------------------------------------------------

_STRIP_TITLES = re.compile(
    r"^(mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?|sir|lord|lady|rev\.?|gen\.?|col\.?|"
    r"capt\.?|lt\.?|sgt\.?|pres\.?|sen\.?|rep\.?|hon\.?)\s+",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"^(jane doe( \d+)?|john doe( \d+)?|unknown|anonymous|victim|subject|complainant)$",
    re.IGNORECASE,
)


def _word_ok(word: str) -> bool:
    """True if a name token is a valid name component (not a bare abbreviation)."""
    w = word.rstrip(".,")
    if not w:
        return False
    # 2-char all-uppercase = abbreviation (GH, GW, MK) — reject
    if len(w) == 2 and w.isupper():
        return False
    # Single initials, mixed-case short words (Al, de, von), and full words are fine
    return len(w) >= 1


def is_valid_entity_name(name: str) -> bool:
    """
    Return True only for names that are useful research targets:
    - At least two meaningful tokens (first + last name minimum)
    - Surname (last token) must be a real word, not a bare initial like 'Bill C'
    - No single-word entries (first-name-only or surname-only)
    - No all-lowercase strings (noise extraction artefacts)
    - No 2-char all-uppercase abbreviations (MBS, GW, etc.)
    - No placeholder patterns (Jane Doe, John Doe, etc.)
    """
    name = name.strip()
    if not name or len(name) < 5:
        return False
    if _PLACEHOLDER_RE.match(name):
        return False
    core = _STRIP_TITLES.sub("", name).strip()
    words = [w for w in re.split(r"[\s\-]+", core) if w]
    valid_words = [w for w in words if _word_ok(w)]
    if len(valid_words) < 2:
        return False
    # Surname (last token) must be a real word, not a bare initial
    if len(valid_words[-1].rstrip(".,")) < 3:
        return False
    # At least one token must be a full word (>= 3 chars) — rejects 'A. B.' noise
    if not any(len(w.rstrip(".,")) >= 3 for w in valid_words):
        return False
    # Reject fully lowercase strings (noise like 'bob', 'eric', 'noel kalb')
    if name == name.lower():
        return False
    return True


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

# Redis keys
CHECKPOINT_KEY_PREFIX = "osia:epstein:checkpoint:"  # + dataset name
SEEN_ENTITIES_KEY = "osia:research:seen_topics"  # permanent dedup set (shared with entity_extractor)
RESEARCH_QUEUE_KEY = "osia:research_queue"
RESEARCH_COOLDOWN_KEY_PREFIX = "osia:research:seen:"  # + md5(topic)
RESEARCH_COOLDOWN_HOURS = int(os.getenv("RESEARCH_COOLDOWN_HOURS", "72"))

# Dataset registry
DATASETS = {
    "emails": {
        "hf_id": "notesbymuneeb/epstein-emails",
        "source_label": "House Oversight Committee (Epstein Emails)",
        "entity_sample_rate_override": 1.0,  # always do NER — highest intel value
        "doc_type": "email_thread",
    },
    "oversight": {
        "hf_id": "tensonaut/EPSTEIN_FILES_20K",
        "source_label": "House Oversight Committee",
        "entity_sample_rate_override": None,
        "doc_type": "government_document",
    },
    "index": {
        "hf_id": "theelderemo/FULL_EPSTEIN_INDEX",
        "source_label": "Epstein Investigation Index",
        "entity_sample_rate_override": None,
        "doc_type": "indexed_document",
    },
    "nikity": {
        "hf_id": "Nikity/Epstein-Files",
        "source_label": "DOJ Epstein Library",
        "entity_sample_rate_override": None,
        "doc_type": "court_document",
    },
}
DATASET_ORDER = ["emails", "oversight", "index", "nikity"]

TODAY = datetime.now(UTC).strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1500  # characters (~375 tokens)
CHUNK_OVERLAP = 225  # characters (~15%)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Paragraph-aware text chunker with sentence-level fallback and overlap.
    Short texts are returned as a single chunk.
    """
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
        """Emit current chunk and seed the next with the overlap tail."""
        if parts:
            chunks.append("\n\n".join(parts))
            # Keep enough tail text to fill the overlap window
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

        # Paragraph too large on its own — split at sentence boundaries
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
        current_len += para_len + 2  # +2 for "\n\n" separator

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c for c in chunks if len(c.strip()) >= 50]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    dataset: str
    records_seen: int = 0
    records_skipped: int = 0
    records_processed: int = 0
    chunks_created: int = 0
    points_upserted: int = 0
    entities_extracted: int = 0
    research_jobs_enqueued: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def elapsed(self) -> str:
        secs = int(time.time() - self.started_at)
        return f"{secs // 60}m{secs % 60:02d}s"

    def log_progress(self) -> None:
        logger.info(
            "[%s] seen=%d processed=%d skipped=%d chunks=%d upserted=%d "
            "entities=%d research_jobs=%d errors=%d elapsed=%s",
            self.dataset,
            self.records_seen,
            self.records_processed,
            self.records_skipped,
            self.chunks_created,
            self.points_upserted,
            self.entities_extracted,
            self.research_jobs_enqueued,
            self.errors,
            self.elapsed(),
        )


# ---------------------------------------------------------------------------
# Main ingestor
# ---------------------------------------------------------------------------


class EpsteinIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.dry_run: bool = args.dry_run
        self.entity_sample_rate: float = args.entity_sample_rate
        self.skip_entity: bool = args.skip_entity
        self.embed_batch_size: int = args.embed_batch_size
        self.embed_concurrency: int = args.embed_concurrency
        self.upsert_batch_size: int = args.upsert_batch_size
        self.min_text_len: int = args.min_text_len

        self._qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self._redis: aioredis.Redis | None = None  # opened in run()
        self._embed_semaphore = asyncio.Semaphore(self.embed_concurrency)

        # In-memory entity dedup for current run (supplement Redis permanent set)
        self._session_seen_entities: set[str] = set()

        # Pending upsert buffer: list of (collection, PointStruct)
        self._upsert_buffer: list[qdrant_models.PointStruct] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(
        self,
        datasets: list[str],
        limit: int,
        resume: bool,
    ) -> None:
        self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await self._ensure_collection()
            for ds_name in datasets:
                cfg = DATASETS[ds_name]
                stats = IngestStats(dataset=ds_name)
                checkpoint = 0
                if resume:
                    checkpoint = await self._load_checkpoint(ds_name)
                    if checkpoint:
                        logger.info("[%s] Resuming from record %d", ds_name, checkpoint)

                logger.info(
                    "[%s] Starting ingestion of %s (limit=%s, resume_from=%d)",
                    ds_name,
                    cfg["hf_id"],
                    limit or "none",
                    checkpoint,
                )

                if ds_name == "emails":
                    await self._ingest_emails(stats, limit, checkpoint)
                elif ds_name == "oversight":
                    await self._ingest_generic(
                        ds_name,
                        "tensonaut/EPSTEIN_FILES_20K",
                        text_field="text",
                        id_field=None,
                        stats=stats,
                        limit=limit,
                        checkpoint=checkpoint,
                    )
                elif ds_name == "index":
                    await self._ingest_generic(
                        ds_name,
                        "theelderemo/FULL_EPSTEIN_INDEX",
                        text_field="text",
                        id_field="id",
                        stats=stats,
                        limit=limit,
                        checkpoint=checkpoint,
                    )
                elif ds_name == "nikity":
                    await self._ingest_nikity(stats, limit, checkpoint)

                # Flush any remaining buffered points
                await self._flush_upsert_buffer()
                stats.log_progress()
                logger.info("[%s] Ingestion complete.", ds_name)
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
    # Dataset-specific ingestors
    # ------------------------------------------------------------------

    async def _ingest_emails(
        self,
        stats: IngestStats,
        limit: int,
        checkpoint: int,
    ) -> None:
        """
        notesbymuneeb/epstein-emails — structured email threads.
        Each thread is treated as a document; always runs NER since these
        are the highest-intelligence-value records in the corpus.
        """
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("notesbymuneeb/epstein-emails", split="train", streaming=True)
        cfg = DATASETS["emails"]

        async for row in self._stream_rows(ds, stats, limit, checkpoint):
            thread_id = str(row.get("thread_id") or row.get("source_file") or uuid.uuid4())
            subject = str(row.get("subject") or "")
            source_file = str(row.get("source_file") or "")
            raw_messages = row.get("messages") or []

            # Deserialise messages if stored as JSON string
            if isinstance(raw_messages, str):
                try:
                    raw_messages = json.loads(raw_messages)
                except json.JSONDecodeError:
                    raw_messages = []

            # Assemble readable thread text
            parts: list[str] = []
            if subject:
                parts.append(f"Subject: {subject}")
            for msg in raw_messages:
                if not isinstance(msg, dict):
                    continue
                header = " | ".join(f"{k}: {msg[k]}" for k in ("date", "from", "to", "cc") if msg.get(k))
                body = str(msg.get("body") or msg.get("content") or "").strip()
                if header or body:
                    parts.append(f"{header}\n\n{body}" if header else body)

            text = "\n\n---\n\n".join(parts)
            if len(text) < self.min_text_len:
                stats.records_skipped += 1
                continue

            metadata = {
                "source": cfg["source_label"],
                "dataset": cfg["hf_id"],
                "doc_id": thread_id,
                "document_type": "email_thread",
                "subject": subject[:200],
                "source_file": source_file,
                "provenance": "public_record",
                "sensitivity": "high",
                "ingest_date": TODAY,
            }

            # Emails always get NER
            await self._process_document(
                doc_id=thread_id,
                text=text,
                metadata=metadata,
                do_ner=True,
                stats=stats,
            )

        await self._save_checkpoint("emails", stats.records_seen)

    async def _ingest_nikity(
        self,
        stats: IngestStats,
        limit: int,
        checkpoint: int,
    ) -> None:
        """
        Nikity/Epstein-Files — 4.11M rows from the full DOJ library.
        Filters aggressively: require text, no errors, minimum length.
        Streams to avoid OOM; never loads the full dataset into memory.
        """
        from datasets import load_dataset  # type: ignore[import-untyped]

        cfg = DATASETS["nikity"]
        ds = load_dataset(
            cfg["hf_id"],
            split="train",
            streaming=True,
            token=HF_TOKEN or None,
        )

        async for row in self._stream_rows(ds, stats, limit, checkpoint):
            # Hard filters
            if row.get("error"):
                stats.records_skipped += 1
                continue
            text = str(row.get("text_content") or "").strip()
            if len(text) < self.min_text_len:
                stats.records_skipped += 1
                continue

            doc_id = str(row.get("doc_id") or uuid.uuid4())
            file_name = str(row.get("file_name") or "")
            file_type = str(row.get("file_type") or "")
            online_url = str(row.get("online_url") or "")
            dataset_id = row.get("dataset_id")

            # Parse structured metadata if available
            raw_meta = row.get("metadata") or ""
            doc_meta: dict = {}
            if raw_meta:
                try:
                    doc_meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    pass

            metadata: dict = {
                "source": cfg["source_label"],
                "dataset": cfg["hf_id"],
                "doc_id": doc_id,
                "file_name": file_name,
                "file_type": file_type,
                "online_url": online_url,
                "document_type": cfg["doc_type"],
                "provenance": "public_record",
                "sensitivity": "high",
                "ingest_date": TODAY,
            }
            if dataset_id is not None:
                metadata["dataset_id"] = int(dataset_id)
            if doc_meta.get("date"):
                metadata["document_date"] = str(doc_meta["date"])[:20]
            if doc_meta.get("title"):
                metadata["document_title"] = str(doc_meta["title"])[:300]

            do_ner = self._should_extract_entities(cfg)
            await self._process_document(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                do_ner=do_ner,
                stats=stats,
            )

        await self._save_checkpoint("nikity", stats.records_seen)

    async def _ingest_generic(
        self,
        ds_name: str,
        hf_id: str,
        text_field: str,
        id_field: str | None,
        stats: IngestStats,
        limit: int,
        checkpoint: int,
    ) -> None:
        """
        Generic ingestor for oversight and index datasets which share a simple
        {id, text} or {text} structure.
        """
        from datasets import load_dataset  # type: ignore[import-untyped]

        cfg = DATASETS[ds_name]
        ds = load_dataset(hf_id, split="train", streaming=True, token=HF_TOKEN or None)

        async for row in self._stream_rows(ds, stats, limit, checkpoint):
            text = str(row.get(text_field) or "").strip()
            if len(text) < self.min_text_len:
                stats.records_skipped += 1
                continue

            if id_field and row.get(id_field) is not None:
                doc_id = str(row[id_field])
            else:
                # Derive a stable ID from content hash
                doc_id = hashlib.sha256(text[:512].encode()).hexdigest()[:16]

            metadata: dict = {
                "source": cfg["source_label"],
                "dataset": hf_id,
                "doc_id": doc_id,
                "document_type": cfg["doc_type"],
                "provenance": "public_record",
                "sensitivity": "high",
                "ingest_date": TODAY,
            }

            do_ner = self._should_extract_entities(cfg)
            await self._process_document(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
                do_ner=do_ner,
                stats=stats,
            )

        await self._save_checkpoint(ds_name, stats.records_seen)

    # ------------------------------------------------------------------
    # Core document pipeline
    # ------------------------------------------------------------------

    async def _process_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
        do_ner: bool,
        stats: IngestStats,
    ) -> None:
        """Chunk → (optionally NER) → buffer for batch embed+upsert."""
        chunks = chunk_text(text)
        if not chunks:
            stats.records_skipped += 1
            return

        stats.records_processed += 1
        stats.chunks_created += len(chunks)

        # Entity extraction runs on the full document text (before chunking)
        # to preserve context across paragraph boundaries.
        if do_ner and not self.skip_entity:
            names = await self._extract_entities(doc_id, text)
            if names:
                stats.entities_extracted += len(names)
                enqueued = await self._enqueue_research(names)
                stats.research_jobs_enqueued += enqueued

        # Build PointStructs with deterministic IDs so re-runs are idempotent
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.UUID(bytes=hashlib.sha256(chunk.encode()).digest()[:16]))
            payload = {
                "text": chunk,
                "chunk_index": i,
                "chunk_count": len(chunks),
                **metadata,
            }
            self._upsert_buffer.append(
                qdrant_models.PointStruct(id=point_id, vector=[0.0] * EMBEDDING_DIM, payload=payload)
            )

        # Flush when buffer is large enough to embed in one go
        if len(self._upsert_buffer) >= self.upsert_batch_size:
            await self._flush_upsert_buffer()
            stats.points_upserted += self.upsert_batch_size  # approximate

        # Progress log every 500 records
        if stats.records_processed % 500 == 0:
            stats.log_progress()

    async def _flush_upsert_buffer(self) -> None:
        """Embed all buffered chunks concurrently then batch-upsert to Qdrant."""
        if not self._upsert_buffer:
            return

        points = list(self._upsert_buffer)
        self._upsert_buffer.clear()
        texts = [p.payload["text"] for p in points]

        # Embed in sub-batches with limited concurrency
        vectors = await self._embed_all(texts)

        # Patch each point with its real vector
        for point, vector in zip(points, vectors, strict=True):
            point.vector = vector

        if not self.dry_run:
            await self._qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            logger.debug("Upserted %d points to '%s'.", len(points), COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts using the HF Inference API.
        Splits into sub-batches and runs up to `embed_concurrency` calls in parallel.
        """
        batches = [texts[i : i + self.embed_batch_size] for i in range(0, len(texts), self.embed_batch_size)]
        results: list[list[float]] = []
        # Process batches in concurrent groups
        for group_start in range(0, len(batches), self.embed_concurrency):
            group = batches[group_start : group_start + self.embed_concurrency]
            group_results = await asyncio.gather(*[self._embed_batch(b) for b in group])
            for batch_vecs in group_results:
                results.extend(batch_vecs)
        return results

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Single HF Inference API embedding call with retry on 429."""
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
            # Fall back to zero vectors on total failure — point is still indexed
            # and will be reachable by filter, just not by vector similarity.
            logger.error("Embedding failed for batch of %d — using zero vectors.", len(texts))
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _should_extract_entities(self, cfg: dict) -> bool:
        """
        Probabilistic sampling to control NER cost on large datasets.
        Emails always return True via their dataset-level override.
        """
        import random  # noqa: S311

        rate = cfg.get("entity_sample_rate_override") or self.entity_sample_rate
        return random.random() < rate  # noqa: S311

    async def _extract_entities(self, doc_id: str, text: str) -> list[str]:
        """
        Extract person names from a document using Venice (venice-uncensored).
        Uses the first 4000 characters for cost control.
        Returns deduplicated list of names, or [] on failure.

        Venice is used instead of a censored model because this corpus contains
        extremely sensitive content; guardrailed models refuse or sanitise output.
        """
        if not VENICE_API_KEY:
            logger.debug("VENICE_API_KEY not set — skipping entity extraction for %s", doc_id)
            return []

        truncated = text[:4000]
        prompt = (
            "You are a named entity extraction system specialised in legal and government documents.\n"
            "Extract all FULL person names (first name AND last name required) from the text below.\n"
            "Rules:\n"
            "- Include only names where BOTH a first name and a last name are present.\n"
            "- Do NOT include first-name-only entries (e.g. 'Alan', 'Jeffrey', 'Ghislaine').\n"
            "- Do NOT include surname-only entries (e.g. 'Clinton', 'Dershowitz', 'Obama').\n"
            "- Do NOT include abbreviations or initials-only (e.g. 'MBS', 'GW Bush', 'DjT').\n"
            "- Do NOT include placeholder names (e.g. 'Jane Doe', 'John Doe').\n"
            "- Titles (Mr., Dr., Prince, Senator) may be included but the actual name must follow.\n"
            "- If the text refers to a person by title/role only, omit them.\n"
            "Return ONLY a JSON array of strings — no explanation, no markdown.\n"
            'Example: ["Jeffrey Epstein", "Ghislaine Maxwell", "Alan Dershowitz"]\n'
            "If no qualifying names are found, return: []\n\n"
            f"Text:\n{truncated}"
        )

        payload = {
            "model": VENICE_NER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as http:
                    resp = await http.post(
                        f"{VENICE_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {VENICE_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    if resp.status_code == 429:
                        wait = 35 * (attempt + 1)
                        logger.warning("Venice NER 429 — waiting %ds", wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    raw = resp.json()["choices"][0]["message"]["content"].strip()
                    break
            except Exception as exc:
                logger.warning("Venice NER attempt %d failed for %s: %s", attempt + 1, doc_id, exc)
                await asyncio.sleep(5 * (attempt + 1))
        else:
            return []

        # Strip markdown fences if the model wrapped its output
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(line for line in lines if not line.startswith("```")).strip()

        try:
            names = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(names, list):
            return []

        # Secondary filter: catch anything the model returned that still doesn't meet
        # the full-name quality bar (model doesn't always follow instructions perfectly)
        return [str(n).strip() for n in names if n and isinstance(n, str) and is_valid_entity_name(str(n))]

    async def _enqueue_research(self, names: list[str]) -> int:
        """
        Push HUMINT research jobs to osia:research_queue for novel entities.
        Deduplicates against:
          1. In-memory session set (fast)
          2. Redis permanent seen_topics set (cross-session)
        Returns count of jobs actually enqueued.
        """
        if not self._redis or self.dry_run:
            return 0

        enqueued = 0
        for name in names:
            normalised = name.lower().strip()
            if not normalised or len(normalised) < 3:
                continue

            # Fast in-memory check first
            if normalised in self._session_seen_entities:
                continue

            # Redis permanent set check
            already_seen = await self._redis.sismember(SEEN_ENTITIES_KEY, normalised)
            if already_seen:
                self._session_seen_entities.add(normalised)
                continue

            job = json.dumps(
                {
                    "job_id": str(uuid.uuid4()),
                    "topic": name,
                    "desk": "human-intelligence-and-profiling-desk",
                    "priority": "normal",
                    "directives_lens": True,
                    "triggered_by": "epstein_ingest",
                }
            )
            await self._redis.rpush(RESEARCH_QUEUE_KEY, job)
            await self._redis.sadd(SEEN_ENTITIES_KEY, normalised)
            self._session_seen_entities.add(normalised)

            logger.info("Research job enqueued: %r → HUMINT desk", name)
            enqueued += 1

        return enqueued

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    async def _stream_rows(
        self,
        dataset,
        stats: IngestStats,
        limit: int,
        checkpoint: int,
    ):
        """
        Async generator that wraps a synchronous HF streaming dataset.
        Handles checkpoint skipping and limit enforcement.
        Logs a warning every 10K skipped rows during fast-forward.
        """
        loop = asyncio.get_event_loop()
        skipped_ff = 0

        def _iter():
            yield from dataset

        it = _iter()

        while True:
            try:
                row = await loop.run_in_executor(None, next, it)
            except StopIteration:
                break

            stats.records_seen += 1

            # Fast-forward past checkpoint (skip without processing)
            if stats.records_seen <= checkpoint:
                skipped_ff += 1
                if skipped_ff % 10_000 == 0:
                    logger.info("Fast-forwarding checkpoint: %d/%d", skipped_ff, checkpoint)
                continue

            yield row

            if limit and stats.records_seen - checkpoint >= limit:
                logger.info("Reached --limit %d — stopping.", limit)
                break

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    async def _save_checkpoint(self, dataset: str, cursor: int) -> None:
        if self.dry_run or not self._redis:
            return
        key = f"{CHECKPOINT_KEY_PREFIX}{dataset}"
        await self._redis.set(key, cursor)
        logger.info("[%s] Checkpoint saved: %d records", dataset, cursor)

    async def _load_checkpoint(self, dataset: str) -> int:
        if not self._redis:
            return 0
        key = f"{CHECKPOINT_KEY_PREFIX}{dataset}"
        val = await self._redis.get(key)
        return int(val) if val else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest Epstein Files datasets into OSIA Qdrant knowledge base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="emails",
        choices=[*DATASETS.keys(), "all"],
        help="Which dataset to ingest. 'all' runs emails→oversight→index→nikity.",
    )
    p.add_argument("--limit", type=int, default=0, help="Stop after N source records (0=no limit)")
    p.add_argument(
        "--entity-sample-rate",
        type=float,
        default=0.15,
        dest="entity_sample_rate",
        help="Fraction of docs to run Venice NER on (0.0–1.0). Emails always run NER.",
    )
    p.add_argument("--skip-entity", action="store_true", help="Disable all entity extraction")
    p.add_argument("--resume", action="store_true", help="Resume from Redis checkpoint")
    p.add_argument("--dry-run", action="store_true", help="Parse/chunk but skip Qdrant writes")
    p.add_argument("--embed-batch-size", type=int, default=48, dest="embed_batch_size")
    p.add_argument("--embed-concurrency", type=int, default=3, dest="embed_concurrency")
    p.add_argument("--upsert-batch-size", type=int, default=64, dest="upsert_batch_size")
    p.add_argument("--min-text-len", type=int, default=120, dest="min_text_len")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not HF_TOKEN:
        parser.error("HF_TOKEN not set in environment — required for dataset access and embeddings.")

    if args.dataset == "all":
        datasets = DATASET_ORDER
    else:
        datasets = [args.dataset]

    logger.info(
        "Starting Epstein ingest | datasets=%s limit=%s entity_rate=%.0f%% dry_run=%s",
        datasets,
        args.limit or "none",
        (args.entity_sample_rate * 100),
        args.dry_run,
    )

    if args.dry_run:
        logger.warning("DRY RUN — no data will be written to Qdrant or Redis.")

    ingestor = EpsteinIngestor(args)
    asyncio.run(ingestor.run(datasets=datasets, limit=args.limit, resume=args.resume))


if __name__ == "__main__":
    main()
