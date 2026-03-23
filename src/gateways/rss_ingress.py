import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import feedparser
import redis.asyncio as redis
from bs4 import BeautifulSoup
from google import genai
from dotenv import load_dotenv
from src.desks.anythingllm_client import AnythingLLMDesk

logger = logging.getLogger("osia.rss")

# Redis keys
SEEN_KEY = "osia:rss:seen_links"
DAILY_DIGEST_KEY = "osia:rss:daily_digest"  # list of summaries collected today


class RSSIngress:
    def __init__(self):
        load_dotenv()
        self.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
        self.desk_client = AnythingLLMDesk()
        base_dir = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
        self.feeds_file = base_dir / "config" / "feeds.txt"

    def get_feeds(self) -> list[str]:
        if not self.feeds_file.exists():
            logger.warning("Feeds file not found at %s — no feeds to poll.", self.feeds_file)
            return []
        with open(self.feeds_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    async def process_feeds(self) -> int:
        """
        Poll all RSS feeds, summarize new articles, ingest into the Collection
        Directorate, and stage summaries in Redis for the daily SITREP.

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
                feed = feedparser.parse(url)

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

                    # Summarize with Gemini
                    prompt = (
                        "You are an intelligence analyst. Summarize this news report in 2-3 concise paragraphs, "
                        "focusing on: who is involved, what happened, strategic implications, and any data/figures mentioned.\n\n"
                        f"Title: {title}\n"
                        f"Published: {published}\n"
                        f"Source: {url}\n\n"
                        f"Content:\n{clean_text[:8000]}"  # cap input to avoid token limits
                    )
                    try:
                        res = self.client.models.generate_content(model=self.model_id, contents=prompt)
                        ai_summary = res.text
                    except Exception as e:
                        logger.error("Gemini summarization failed for '%s': %s", title, e)
                        # Fall back to raw text truncation
                        ai_summary = clean_text[:2000]

                    # Build the intelligence record
                    intel_record = (
                        f"TITLE: {title}\n"
                        f"SOURCE: {url}\n"
                        f"LINK: {link}\n"
                        f"PUBLISHED: {published}\n"
                        f"COLLECTED: {datetime.now(timezone.utc).isoformat()}\n\n"
                        f"{ai_summary}"
                    )

                    # 1. Ingest into AnythingLLM Collection Directorate (long-term vector store)
                    try:
                        safe_title = (title[:50] + "...") if len(title) > 50 else title
                        await self.desk_client.ingest_raw_data(
                            workspace_slug="collection-directorate",
                            text_content=intel_record,
                            title=f"RSS: {safe_title}",
                        )
                    except Exception as e:
                        logger.error("AnythingLLM ingestion failed for '%s': %s", title, e)

                    # 2. Stage in Redis daily digest (consumed by SITREP generator)
                    await self.redis.rpush(DAILY_DIGEST_KEY, intel_record)

                    # 3. Mark as seen
                    await self.redis.sadd(SEEN_KEY, link)
                    new_count += 1

            except Exception as e:
                logger.error("Failed to process feed %s: %s", url, e)

        logger.info("RSS scan complete. %d new articles processed.", new_count)
        await self.desk_client.close()
        return new_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ingress = RSSIngress()
    asyncio.run(ingress.process_feeds())
