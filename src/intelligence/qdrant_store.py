"""
OSIA Qdrant Intelligence Store

Direct async Qdrant client replacing AnythingLLM's vector store abstraction.
Provides filtered semantic search, cross-desk retrieval, deterministic upserts,
and collection bootstrapping.

Environment variables:
  QDRANT_URL      — Qdrant server URL
  QDRANT_API_KEY  — Qdrant API key
  HF_TOKEN        — HuggingFace token for embedding API
"""

import asyncio
import hashlib
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

logger = logging.getLogger("osia.qdrant_store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

# Primary desk collections + broad KBs — always searched by cross_desk_search.
# Names match desk YAML slugs and the live Qdrant collections.
DESK_COLLECTIONS: list[str] = [
    # Desk primary collections
    "collection-directorate",
    "geopolitical-and-security-desk",
    "cultural-and-theological-intelligence-desk",
    "science-technology-and-commercial-desk",
    "human-intelligence-and-profiling-desk",
    "finance-and-economics-directorate",
    "cyber-intelligence-and-warfare-desk",
    "information-warfare-desk",
    "environment-and-ecology-desk",
    "the-watch-floor",
    "osia_research_cache",
    # Broadly relevant KB collections included in every cross-desk search
    "epstein-files",  # declassified government documents — DOJ, House Oversight, federal courts
    "wikileaks-cables",  # 124K US diplomatic cables (1966–2010) — classified embassy traffic, geopolitical intel
    "ofac-sanctions",  # OFAC SDN list — 18K+ sanctioned individuals, entities, vessels
    "icij-offshore-leaks",  # 810K+ offshore entities — Panama/Pandora/Paradise Papers, beneficial owners
    "yahoo-finance",  # Yahoo Finance news articles, earnings call transcripts, company profiles
    # Cyber-focused KBs (relevant broadly for threat context)
    "mitre-attack",  # MITRE ATT&CK: ~700 techniques, ~140 APT group profiles, ~600 malware/tools
    "cti-reports",  # 9.7K NER-annotated CTI report texts — malware, threat-actor, IOC entities
    "ttp-mappings",  # 20.7K threat report snippets with expert ATT&CK technique ID labels
    "cve-database",  # 280K NVD CVEs 1999–2025 — descriptions, CVSS scores, CWE classifications
    "cti-bench",  # 5.6K analyst benchmark scenarios
    "cybersecurity-attacks",  # 13K documented global cyber incidents
    "hackerone-reports",  # 12.6K publicly disclosed bug bounty reports
    # Cultural / ideological KBs
    "etymology-database",  # historical origin and evolution of words, terms, and concepts
    "marxists-archive",  # Marxists Internet Archive — political theory, ideology, revolutionary history
]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    text: str
    score: float
    collection: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QdrantStore
# ---------------------------------------------------------------------------


class QdrantStore:
    """
    Async Qdrant client with embedding, upsert, search, and cross-desk retrieval.
    """

    def __init__(self) -> None:
        qdrant_url = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
        self._hf_token = os.getenv("HF_TOKEN", "")
        self._client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key, port=None)

    async def aclose(self) -> None:
        """Close the underlying Qdrant HTTP connection pool."""
        await self._client.close()

    async def __aenter__(self) -> "QdrantStore":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        """Embed a single text via HuggingFace Inference API."""
        return (await self._embed_batch([text]))[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts via HuggingFace Inference API.
        On failure: log error and return zero vectors so callers can continue.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    HF_EMBEDDING_URL,
                    headers={"Authorization": f"Bearer {self._hf_token}"},
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                )
                resp.raise_for_status()
                result = resp.json()
                # HF returns list[list[float]] for batch inputs
                if isinstance(result, list) and result and isinstance(result[0], list):
                    return result
                # Single text may return list[float] — wrap it
                if isinstance(result, list) and result and isinstance(result[0], (int, float)):
                    return [result]
                logger.error("Unexpected HF embedding response shape: %s", type(result))
                return [[0.0] * EMBEDDING_DIM for _ in texts]
        except Exception as exc:
            logger.error("HF embedding API failed: %s — using zero vectors", exc)
            return [[0.0] * EMBEDDING_DIM for _ in texts]

    # ------------------------------------------------------------------
    # Temporal decay
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_decay(results: list["SearchResult"], half_life_days: float) -> list["SearchResult"]:
        """Re-score results with exponential time decay on ingested_at_unix payload field.
        Points without the field are left unmodified (no penalty for legacy data).
        """
        now = time.time()
        for r in results:
            ts = r.metadata.get("ingested_at_unix")
            if ts is not None:
                age_days = (now - float(ts)) / 86400.0
                r.score = r.score * math.exp(-math.log(2) * age_days / half_life_days)
        return sorted(results, key=lambda x: x.score, reverse=True)

    # ------------------------------------------------------------------
    # Point ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _point_id(text: str) -> str:
        """Deterministic UUID from SHA-256 hash of text content."""
        return str(uuid.UUID(bytes=hashlib.sha256(text.encode()).digest()[:16]))

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def ensure_collection(self, name: str) -> None:
        """
        Create collection with 384-dim Cosine vectors if it does not exist.
        Idempotent — no-op if already present.
        Raises on Qdrant API error (caller should abort startup).
        """
        try:
            existing = await self._client.collection_exists(name)
            if existing:
                logger.info("Collection '%s' already present and ready.", name)
                return

            await self._client.create_collection(
                collection_name=name,
                vectors_config=qdrant_models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=1000,
                ),
            )
            logger.info("Collection '%s' created.", name)
        except Exception as exc:
            logger.error("Failed to ensure collection '%s': %s", name, exc)
            raise

    async def collection_stats(self, name: str) -> dict:
        """Return point count and vector count for the named collection."""
        info = await self._client.get_collection(name)
        counts = info.points_count or 0
        vectors = info.vectors_count or 0
        return {"points_count": counts, "vectors_count": vectors}

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert(self, collection: str, text: str, metadata: dict) -> str:
        """
        Embed text, generate deterministic point ID, upsert with metadata.
        Returns the point ID string.
        """
        point_id = self._point_id(text)
        vector = await self._embed(text)

        payload = {"text": text, "ingested_at_unix": int(time.time()), **metadata}

        await self._client.upsert(
            collection_name=collection,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.debug("Upserted point %s into '%s'", point_id, collection)
        return point_id

    # ------------------------------------------------------------------
    # Chunked upsert — for INTSUM write-backs and long documents
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int = 600) -> list[str]:
        """Split markdown text into semantic chunks sized for embedding quality.

        Splits on markdown section headers first, then paragraph breaks.
        Each sub-section header is re-prepended to its child chunks so each
        chunk is self-contained without requiring the full document for context.
        """
        import re

        # Prepend a newline so the first section header is caught by the split
        sections = re.split(r"\n(?=#{1,3} )", "\n" + text.strip())
        sections = [s.strip() for s in sections if s.strip()]

        chunks: list[str] = []
        for section in sections:
            if len(section) <= max_chars:
                chunks.append(section)
                continue

            # Extract section header (first line when it starts with #)
            parts = section.split("\n", 1)
            header = parts[0].strip() if parts[0].startswith("#") else ""
            body = parts[1].strip() if len(parts) > 1 else section

            # Split body on paragraph boundaries
            paragraphs = re.split(r"\n{2,}", body)
            current_parts: list[str] = [header] if header else []
            current_len: int = len(header) if header else 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                add_len = len(para) + (2 if current_parts else 0)
                if current_len + add_len <= max_chars:
                    current_parts.append(para)
                    current_len += add_len
                else:
                    if current_parts:
                        chunks.append("\n\n".join(current_parts))
                    # Re-prepend header so each chunk is self-contained
                    current_parts = [header, para] if header else [para]
                    current_len = (len(header) + 2 + len(para)) if header else len(para)

            if current_parts:
                chunks.append("\n\n".join(current_parts))

        # Drop tiny fragments; fall back to first portion if nothing remains
        return [c for c in chunks if len(c.strip()) >= 40] or [text[: max_chars * 3]]

    async def upsert_chunks(
        self,
        collection: str,
        text: str,
        metadata: dict,
        max_chars: int = 600,
    ) -> list[str]:
        """Split text into semantic chunks and upsert each as a separate Qdrant point.

        Vectors are batched in a single HF API call for efficiency.
        Each chunk inherits all metadata fields plus chunk_index and chunk_total.
        Returns list of point IDs.
        """
        chunks = self._split_into_chunks(text, max_chars)
        if not chunks:
            return []

        total = len(chunks)
        vectors = await self._embed_batch(chunks)
        now_unix = int(time.time())

        points = []
        point_ids = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors, strict=False)):
            point_id = self._point_id(chunk_text)
            payload = {
                "text": chunk_text,
                "ingested_at_unix": now_unix,
                "chunk_index": i,
                "chunk_total": total,
                **metadata,
            }
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
            point_ids.append(point_id)

        await self._client.upsert(collection_name=collection, points=points)
        logger.debug("Upserted %d chunks into '%s'", total, collection)
        return point_ids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int,
        filters: dict | None = None,
        decay_half_life_days: float | None = None,
        payload_boost_field: str | None = None,
    ) -> list[SearchResult]:
        """
        Embed query and return top-K semantically similar points.

        decay_half_life_days: if set, re-scores by exponential time decay on
            ingested_at_unix and re-sorts before returning.

        payload_boost_field: if set, names a float payload field in [0, 1] used
            to boost scores after retrieval.  Fetches top_k * 4 candidates first,
            applies score *= (0.3 + 0.7 * field_value), then returns the top_k.
            Designed for fields like mentions_weight that encode signal strength.
            Points missing the field are treated as weight=0.5 (no strong penalty).
        """
        vector = await self._embed(query)

        qdrant_filter: qdrant_models.Filter | None = None
        if filters:
            must_conditions = [
                qdrant_models.FieldCondition(
                    key=k,
                    match=qdrant_models.MatchValue(value=v),
                )
                for k, v in filters.items()
            ]
            qdrant_filter = qdrant_models.Filter(must=must_conditions)

        fetch_limit = top_k * 4 if payload_boost_field else top_k
        hits_result = await self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=fetch_limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        hits = hits_result.points

        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(
                SearchResult(
                    text=text,
                    score=hit.score,
                    collection=collection,
                    metadata=payload,
                )
            )

        if payload_boost_field:
            for r in results:
                weight = float(r.metadata.get(payload_boost_field, 0.5))
                r.score *= 0.3 + 0.7 * max(0.0, min(1.0, weight))
            results.sort(key=lambda r: r.score, reverse=True)
            results = results[:top_k]

        if decay_half_life_days:
            results = self._apply_decay(results, decay_half_life_days)
        return results

    # ------------------------------------------------------------------
    # Cross-desk search
    # ------------------------------------------------------------------

    async def cross_desk_search(
        self,
        query: str,
        top_k: int,
        entity_tags: list[str] | None = None,
        decay_half_life_days: float | None = None,
        collections: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Fan out across desk collections, rank by score, deduplicate by point ID.

        collections: explicit list of collection names to search. Defaults to
        DESK_COLLECTIONS when None. Pass a desk-specific superset (DESK_COLLECTIONS
        + boost collections) for richer, context-aware retrieval.

        If decay_half_life_days is set, applies exponential recency decay before
        final ranking.
        """
        vector = await self._embed(query)

        # Build optional entity_tags filter
        qdrant_filter: qdrant_models.Filter | None = None
        if entity_tags:
            qdrant_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="entity_tags",
                        match=qdrant_models.MatchAny(any=entity_tags),
                    )
                ]
            )

        async def _search_one(col: str) -> list[tuple[str, SearchResult]]:
            """Search a single collection; return (point_id, result) pairs."""
            try:
                hits_result = await self._client.query_points(
                    collection_name=col,
                    query=vector,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                hits = hits_result.points
                out = []
                for hit in hits:
                    payload = dict(hit.payload or {})
                    text = payload.pop("text", "")
                    out.append(
                        (
                            str(hit.id),
                            SearchResult(
                                text=text,
                                score=hit.score,
                                collection=col,
                                metadata=payload,
                            ),
                        )
                    )
                return out
            except Exception as exc:
                logger.warning("cross_desk_search: collection '%s' unavailable: %s", col, exc)
                return []

        # Fan out concurrently across the requested collection set
        target_collections = collections if collections is not None else DESK_COLLECTIONS
        all_pairs: list[tuple[str, SearchResult]] = []
        tasks = [_search_one(col) for col in target_collections]
        for pairs in await asyncio.gather(*tasks):
            all_pairs.extend(pairs)

        # Deduplicate by point ID, keeping highest score
        seen: dict[str, SearchResult] = {}
        for point_id, result in all_pairs:
            if point_id not in seen or result.score > seen[point_id].score:
                seen[point_id] = result

        # Sort by score descending, apply optional decay, return top_k
        ranked = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        if decay_half_life_days:
            ranked = self._apply_decay(ranked, decay_half_life_days)
        return ranked[:top_k]
