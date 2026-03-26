"""
OSIA Entity Extractor

Named entity extraction pipeline that converts incoming intel text into
structured Entity objects and enqueues background research jobs.

Environment variables:
  GEMINI_API_KEY   — Google Gemini API key
  GEMINI_MODEL_ID  — Gemini model to use (default: gemini-2.5-flash)
  REDIS_URL        — Redis connection string
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass

import redis.asyncio as aioredis
from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger("osia.entity_extractor")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
# Gemini extraction prompt
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
    Extracts named entities from text using Gemini and enqueues research jobs.
    """

    def __init__(self) -> None:
        self._gemini = genai.Client(api_key=GEMINI_API_KEY)
        self._redis_url = REDIS_URL

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def extract(self, text: str, source_desk: str) -> list[Entity]:
        """
        Call Gemini to identify named entities in text.
        Returns a list of Entity objects, or [] on malformed/unparseable JSON.
        """
        import asyncio

        prompt = EXTRACTION_PROMPT.format(text=text)

        try:
            response = await asyncio.to_thread(
                self._gemini.models.generate_content,
                model=GEMINI_MODEL_ID,
                contents=prompt,
            )
            raw = response.text.strip()
        except Exception as exc:
            logger.warning("Gemini entity extraction call failed: %s", exc)
            return []

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(line for line in lines if not line.startswith("```")).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Entity extractor: unparseable JSON from Gemini (%s). Raw: %.200s",
                exc,
                raw,
            )
            return []

        if not isinstance(data, list):
            logger.warning(
                "Entity extractor: expected JSON array, got %s. Raw: %.200s",
                type(data).__name__,
                raw,
            )
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
