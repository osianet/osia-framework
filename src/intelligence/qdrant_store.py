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

# All desk collections + research cache — used by cross_desk_search.
# Names match desk YAML slugs and the live Qdrant collections.
DESK_COLLECTIONS: list[str] = [
    "collection-directorate",
    "geopolitical-and-security-desk",
    "cultural-and-theological-intelligence-desk",
    "science-technology-and-commercial-desk",
    "human-intelligence-and-profiling-desk",
    "finance-and-economics-directorate",
    "cyber-intelligence-and-warfare-desk",
    "the-watch-floor",
    "osia_research_cache",  # research worker writes here; no colon, SDK-compatible
    "epstein-files",  # declassified government documents — DOJ, House Oversight, federal courts
    "cybersecurity-attacks",  # 13K documented global cyber incidents — actors, TTPs, targets (vinitvek/cybersecurityattacks)
    "hackerone-reports",  # 12.6K publicly disclosed bug bounty reports — CVEs, weaknesses, affected assets (Hacker0x01)
    "wikileaks-cables",  # 124K US diplomatic cables (1966–2010) — classified embassy traffic, geopolitical intel (fn5/wikileaks-cables)
    "mitre-attack",  # MITRE ATT&CK: ~700 techniques, ~140 APT group profiles, ~600 malware/tools, mitigations (enterprise + mobile + ICS)
    "cti-reports",  # 9.7K NER-annotated CTI report texts — malware, threat-actor, IOC entities (mrmoor/cyber-threat-intelligence-splited)
    "ttp-mappings",  # 20.7K threat report snippets with expert ATT&CK technique ID labels (tumeteor/Security-TTP-Mapping)
    "cve-database",  # 280K NVD CVEs 1999–2025 — descriptions, CVSS scores, CWE classifications (stasvinokur/cve-and-cwe-dataset-1999-2025)
    "cti-bench",  # 5.6K analyst benchmark scenarios — malware→ATT&CK, threat actor attribution, CVE→CWE (AI4Sec/cti-bench)
    "etymology-database",  # historical origin and evolution of words, terms, and concepts
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
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int,
        filters: dict | None = None,
        decay_half_life_days: float | None = None,
    ) -> list[SearchResult]:
        """
        Embed query and return top-K semantically similar points.
        Supports optional payload filters (Qdrant filter dict format).
        If decay_half_life_days is set, re-scores results by exponential time decay
        on ingested_at_unix and re-sorts before returning.
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

        hits_result = await self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
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
    ) -> list[SearchResult]:
        """
        Fan out across all registered desk collections + osia:research_cache.
        Rank by score descending, deduplicate by point ID.
        If decay_half_life_days is set, applies exponential recency decay before final ranking.
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

        # Fan out concurrently
        all_pairs: list[tuple[str, SearchResult]] = []
        tasks = [_search_one(col) for col in DESK_COLLECTIONS]
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
