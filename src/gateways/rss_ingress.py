import asyncio
import logging
import os
from pathlib import Path
import feedparser
import redis.asyncio as redis
from bs4 import BeautifulSoup
from google import genai
from dotenv import load_dotenv
from src.desks.anythingllm_client import AnythingLLMDesk

logger = logging.getLogger("osia.rss")


class RSSIngress:
    def __init__(self):
        load_dotenv()
        self.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.desk_client = AnythingLLMDesk()
        base_dir = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
        self.feeds_file = base_dir / "config" / "feeds.txt"
        self.seen_key = "osia:rss:seen_links"

    def get_feeds(self) -> list[str]:
        with open(self.feeds_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    async def process_feeds(self):
        logger.info("RSS Gateway: Scanning intelligence feeds...")
        feeds = self.get_feeds()

        for url in feeds:
            try:
                logger.info("Polling: %s", url)
                feed = feedparser.parse(url)

                for entry in feed.entries[:5]:
                    link = entry.link

                    is_seen = await self.redis.sismember(self.seen_key, link)
                    if not is_seen:
                        logger.info("New intelligence found: %s", entry.title)

                        summary = entry.get("summary", "")
                        clean_text = BeautifulSoup(summary, "html.parser").get_text()

                        prompt = (
                            f"Summarize this news report for intelligence ingestion. "
                            f"Title: {entry.title}\n\nContent: {clean_text}"
                        )
                        res = self.client.models.generate_content(
                            model="gemini-2.5-flash", contents=prompt
                        )
                        ai_summary = res.text

                        await self.desk_client.ingest_raw_data(
                            workspace_slug="collection-directorate",
                            text_content=f"SOURCE: {url}\nLINK: {link}\n\n{ai_summary}",
                            title=f"RSS: {entry.title}",
                        )

                        await self.redis.sadd(self.seen_key, link)

            except Exception as e:
                logger.error("Failed to process feed %s: %s", url, e)

        await self.desk_client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ingress = RSSIngress()
    asyncio.run(ingress.process_feeds())
