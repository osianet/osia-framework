"""
OSIA Entity Extractor

Named entity extraction pipeline that converts incoming intel text into
structured Entity objects and enqueues background research jobs.

Uses Venice (venice-uncensored) for extraction so sensitive queries about
individuals, organisations, and events are never refused or sanitised.
Falls back to Gemini if VENICE_API_KEY is not set.

Environment variables:
  VENICE_API_KEY   — Venice AI API key (primary — uncensored)
  GEMINI_API_KEY   — Google Gemini API key (fallback only)
  GEMINI_MODEL_ID  — Gemini model to use (default: gemini-2.5-flash)
  REDIS_URL        — Redis connection string
"""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("osia.entity_extractor")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")
VENICE_BASE_URL = "https://api.venice.ai/api/v1"
VENICE_MODEL = os.getenv("VENICE_MODEL_UNCENSORED", "venice-uncensored")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

RESEARCH_QUEUE_KEY = "osia:research_queue"
SEEN_TOPICS_KEY = "osia:research:seen_topics"

# ---------------------------------------------------------------------------
# Entity-to-desk routing table
# ---------------------------------------------------------------------------

ENTITY_DESK_ROUTING: dict[str, str] = {
    "Person": "human-intelligence-and-profiling-desk",
    "Organisation": "geopolitical-and-security-desk",
    "Location": "geopolitical-and-security-desk",
    "Event": "geopolitical-and-security-desk",
    "Technology": "science-technology-and-commercial-desk",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    name: str
    entity_type: str  # "Person" | "Organisation" | "Location" | "Event" | "Technology"
    context: str  # sentence/phrase where entity appeared
    source_desk: str


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a named entity extraction system. Analyse the following text and identify all named entities.

Return ONLY a JSON array (no markdown, no explanation) where each element has these fields:
- "name": the entity name as it appears in the text
- "entity_type": one of "Person", "Organisation", "Location", "Event", "Technology"
- "context": the exact sentence or short phrase from the text where this entity appears

Entity type guidance:
- Person: named individuals (include role/affiliation in context if present)
- Organisation: companies, governments, agencies, NGOs, military units
- Location: countries, cities, regions, geographic features
- Event: named events, operations, conflicts, summits, incidents
- Technology: named technologies, systems, platforms, protocols, weapons systems

If no entities are found, return an empty array: []

Text to analyse:
{text}
"""

# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """
    Extracts named entities from text via Venice (uncensored) and enqueues
    background research jobs. Falls back to Gemini if Venice is unavailable.
    """

    def __init__(self) -> None:
        self._redis_url = REDIS_URL

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    # Venice context window is generous but entity extraction only needs names —
    # cap at 8000 chars to avoid 400s on large transcripts/reports.
    _MAX_EXTRACT_CHARS = 8_000

    async def extract(self, text: str, source_desk: str) -> list[Entity]:
        """
        Identify named entities in text. Tries Venice first, falls back to Gemini.
        Returns a list of Entity objects, or [] on failure.
        """
        if len(text) > self._MAX_EXTRACT_CHARS:
            logger.debug("Entity extractor: truncating %d chars to %d", len(text), self._MAX_EXTRACT_CHARS)
            text = text[: self._MAX_EXTRACT_CHARS]

        raw: str | None = None
        if VENICE_API_KEY:
            raw = await self._extract_via_venice(text)
        if raw is None and GEMINI_API_KEY:
            logger.warning("Venice entity extraction failed — falling back to Gemini")
            raw = await self._extract_via_gemini(text)
        if raw is None:
            logger.warning("No AI API key configured or all providers failed — skipping entity extraction")
            return []

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(line for line in lines if not line.startswith("```")).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Entity extractor: unparseable JSON (%s). Raw: %.200s", exc, raw)
            return []

        if not isinstance(data, list):
            logger.warning("Entity extractor: expected JSON array, got %s. Raw: %.200s", type(data).__name__, raw)
            return []

        entities: list[Entity] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            entity_type = item.get("entity_type", "").strip()
            context = item.get("context", "").strip()

            if not name or entity_type not in ENTITY_DESK_ROUTING:
                continue

            entities.append(
                Entity(
                    name=name,
                    entity_type=entity_type,
                    context=context,
                    source_desk=source_desk,
                )
            )

        logger.info(
            "Extracted %d entities from %d chars of text (desk: %s)",
            len(entities),
            len(text),
            source_desk,
        )
        return entities

    async def _extract_via_venice(self, text: str) -> str | None:
        """Call Venice (venice-uncensored) for entity extraction. Returns raw response text."""
        payload = {
            "model": VENICE_MODEL,
            "messages": [{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
            "temperature": 0.0,
            "max_tokens": 1024,
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
                        logger.warning("Venice entity extractor 429 — waiting %ds", wait)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status_code >= 400 and resp.status_code < 500:
                        # Client error (400, 401, etc.) — retrying won't help
                        logger.warning("Venice entity extraction client error %d — skipping retries", resp.status_code)
                        return None
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.warning("Venice entity extraction attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
        return None

    async def _extract_via_gemini(self, text: str) -> str | None:
        """Gemini fallback for entity extraction when Venice is unavailable."""
        from google import genai as _genai

        client = _genai.Client(api_key=GEMINI_API_KEY)
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL_ID,
                contents=EXTRACTION_PROMPT.format(text=text),
            )
            return response.text.strip()
        except Exception as exc:
            logger.warning("Gemini entity extraction fallback failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Research job enqueuing
    # ------------------------------------------------------------------

    async def enqueue_research_jobs(
        self,
        entities: list[Entity],
        triggered_by: str,
    ) -> None:
        """
        Push a research job to osia:research_queue for each novel entity.
        Skips entities whose normalised name is already in osia:research:seen_topics.
        No-op when entity list is empty.
        """
        if not entities:
            return

        redis = aioredis.from_url(self._redis_url, decode_responses=True)
        try:
            enqueued = 0
            for entity in entities:
                normalised = entity.name.lower().strip()

                already_seen = await redis.sismember(SEEN_TOPICS_KEY, normalised)
                if not already_seen:
                    # Also check the worker's TTL-based cooldown key so that a Redis
                    # restart (which clears seen_topics) doesn't re-enqueue recently
                    # processed topics while their cooldown window is still active.
                    topic_md5 = hashlib.md5(normalised.encode(), usedforsecurity=False).hexdigest()
                    already_seen = bool(await redis.exists(f"osia:research:seen:{topic_md5}"))
                if already_seen:
                    logger.debug("Skipping already-seen entity: %r", entity.name)
                    continue

                desk = ENTITY_DESK_ROUTING[entity.entity_type]
                payload = json.dumps(
                    {
                        "job_id": str(uuid.uuid4()),
                        "topic": entity.name,
                        "desk": desk,
                        "priority": "normal",
                        "directives_lens": True,
                        "triggered_by": triggered_by,
                    }
                )

                await redis.rpush(RESEARCH_QUEUE_KEY, payload)
                await redis.sadd(SEEN_TOPICS_KEY, normalised)

                logger.info(
                    "Enqueued research job for %r → %s (triggered_by: %s)",
                    entity.name,
                    desk,
                    triggered_by,
                )
                enqueued += 1

            logger.info(
                "enqueue_research_jobs: %d/%d entities enqueued",
                enqueued,
                len(entities),
            )
        finally:
            await redis.aclose()
