import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import feedparser
import httpx
import redis.asyncio as redis
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai

from src.intelligence.entity_extractor import EntityExtractor
from src.intelligence.qdrant_store import QdrantStore

logger = logging.getLogger("osia.rss")

# ---------------------------------------------------------------------------
# Summarisation model chain
# OpenRouter options first (2), then Gemini direct (non-OR), then raw fallback.
# ---------------------------------------------------------------------------
_OR_SUMMARISE_MODELS = [
    "google/gemma-4-31b-it:free",  # zero-cost first attempt
    "anthropic/claude-haiku-4.5",  # paid, reliable second
]
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Redis keys
SEEN_KEY = "osia:rss:seen_links"
DAILY_DIGEST_KEY = "osia:rss:daily_digest"  # list of summaries collected today

qdrant_store = QdrantStore()
entity_extractor = EntityExtractor()


class RSSIngress:
    def __init__(self):
        load_dotenv()
        self.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
        base_dir = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
        self.feeds_file = base_dir / "config" / "feeds.txt"

    async def _summarise_openrouter(self, prompt: str, model: str) -> str:
        """POST to OpenRouter chat completions and return the assistant text."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _summarise_with_fallback(self, prompt: str, title: str) -> str | None:
        """
        Try OpenRouter models first (2 options), then Gemini direct (non-OR).
        Returns None only if all AI providers fail — caller should fall back to
        raw text truncation.
        """
        if OPENROUTER_API_KEY:
            for model in _OR_SUMMARISE_MODELS:
                try:
                    result = await self._summarise_openrouter(prompt, model)
                    logger.debug("RSS summary via OR model=%s for '%s'", model, title)
                    return result
                except Exception as e:
                    logger.warning("OR summarization failed (model=%s, title='%s'): %s", model, title, e)

        # Non-OR fallback: Gemini direct
        try:
            res = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_id,
                contents=prompt,
            )
            logger.debug("RSS summary via Gemini direct for '%s'", title)
            return res.text
        except Exception as e:
            logger.error("Gemini summarization failed for '%s': %s", title, e)
            return None

    def get_feeds(self) -> list[str]:
        if not self.feeds_file.exists():
            logger.warning("Feeds file not found at %s — no feeds to poll.", self.feeds_file)
            return []
        with open(self.feeds_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    async def process_feeds(self) -> int:
        """
        Poll all RSS feeds, summarize new articles, ingest into the Collection
        Directorate via Qdrant, and stage summaries in Redis for the daily SITREP.

        Returns the number of new articles processed.
        """
        logger.info("RSS Gateway: Scanning intelligence feeds...")
        feeds = self.get_feeds()
        if not feeds:
            logger.info("No feeds configured. Exiting.")
            return 0

        new_count = 0

        for url in feeds:
            try:
                logger.info("Polling: %s", url)
                try:
                    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                        resp = await client.get(url, headers={"User-Agent": "OSIA RSS Ingress/1.0"})
                        resp.raise_for_status()
                        feed = feedparser.parse(resp.text)
                except Exception as e:
                    logger.warning("Feed fetch failed for %s: %s", url, e)
                    continue

                if feed.bozo and not feed.entries:
                    logger.warning("Feed parse error for %s: %s", url, feed.bozo_exception)
                    continue

                for entry in feed.entries[:10]:
                    link = getattr(entry, "link", None)
                    if not link:
                        continue

                    is_seen = await self.redis.sismember(SEEN_KEY, link)
                    if is_seen:
                        continue

                    title = getattr(entry, "title", "Untitled")
                    logger.info("New intelligence: %s", title)

                    # Extract and clean content — try content:encoded first, fall back to summary
                    raw_html = ""
                    if hasattr(entry, "content") and entry.content:
                        raw_html = entry.content[0].get("value", "")
                    if not raw_html:
                        raw_html = getattr(entry, "summary", "")
                    clean_text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)

                    # Get published date if available
                    published = getattr(entry, "published", "")

                    # Summarize via multi-provider fallback chain
                    prompt = (
                        "You are an intelligence analyst. Summarize this news report in 2-3 concise paragraphs, "
                        "focusing on: who is involved, what happened, strategic implications, and any data/figures mentioned.\n\n"
                        f"Title: {title}\n"
                        f"Published: {published}\n"
                        f"Source: {url}\n\n"
                        f"Content:\n{clean_text[:8000]}"  # cap input to avoid token limits
                    )
                    ai_summary = await self._summarise_with_fallback(prompt, title)
                    if ai_summary is None:
                        logger.warning("All summarization providers failed for '%s'; using raw truncation", title)
                        ai_summary = clean_text[:2000]

                    # Build the intelligence record
                    intel_record = (
                        f"TITLE: {title}\n"
                        f"SOURCE: {url}\n"
                        f"LINK: {link}\n"
                        f"PUBLISHED: {published}\n"
                        f"COLLECTED: {datetime.now(UTC).isoformat()}\n\n"
                        f"{ai_summary}"
                    )

                    # 1. Extract entities from the summary
                    try:
                        entities = await entity_extractor.extract(ai_summary, "collection-directorate")
                    except Exception as e:
                        logger.warning("Entity extraction failed for '%s': %s", title, e)
                        entities = []

                    # 2. Enqueue research jobs for extracted entities
                    # Pass the article link (not the feed url) so the research worker
                    # can fetch the source article for context when forming queries.
                    await entity_extractor.enqueue_research_jobs(entities, triggered_by=link)

                    # 3. Upsert into Qdrant collection-directorate (long-term vector store)
                    try:
                        await qdrant_store.upsert(
                            "collection-directorate",
                            intel_record,
                            metadata={
                                "desk": "collection-directorate",
                                "topic": title,
                                "source": url,
                                "reliability_tier": "B",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "entity_tags": [e.name for e in entities],
                                "triggered_by": "rss_ingress",
                            },
                        )
                    except Exception as e:
                        logger.error("Qdrant upsert failed for '%s': %s", title, e)

                    # 4. Stage in Redis daily digest (consumed by SITREP generator)
                    await self.redis.rpush(DAILY_DIGEST_KEY, intel_record)

                    # 5. Mark as seen
                    await self.redis.sadd(SEEN_KEY, link)
                    new_count += 1

            except Exception as e:
                logger.error("Failed to process feed %s: %s", url, e)

        logger.info("RSS scan complete. %d new articles processed.", new_count)
        return new_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ingress = RSSIngress()
    asyncio.run(ingress.process_feeds())
