import asyncio
import os
import json
import feedparser
import redis.asyncio as redis
from bs4 import BeautifulSoup
from google import genai
from dotenv import load_dotenv
from src.desks.anythingllm_client import AnythingLLMDesk

class RSSIngress:
    def __init__(self):
        load_dotenv()
        self.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.desk_client = AnythingLLMDesk()
        self.feeds_file = "/home/ubuntu/osia-framework/config/feeds.txt"
        self.seen_key = "osia:rss:seen_links"

    def get_feeds(self):
        with open(self.feeds_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    async def process_feeds(self):
        print("[*] RSS Gateway: Scanning intelligence feeds...")
        feeds = self.get_feeds()
        
        for url in feeds:
            try:
                print(f"[*] Polling: {url}")
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:5]: # Last 5 items per feed to avoid overwhelm
                    link = entry.link
                    
                    # Check if we have seen this article before
                    is_seen = await self.redis.sismember(self.seen_key, link)
                    if not is_seen:
                        print(f"[+] New intelligence found: {entry.title}")
                        
                        # Extract and clean content
                        summary = entry.get("summary", "")
                        clean_text = BeautifulSoup(summary, "html.parser").get_text()
                        
                        # Generate a high-speed AI summary for ingestion
                        prompt = f"Summarize this news report for intelligence ingestion. Title: {entry.title}\n\nContent: {clean_text}"
                        res = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                        ai_summary = res.text
                        
                        # Ingest into AnythingLLM Collection Directorate
                        await self.desk_client.ingest_raw_data(
                            workspace_slug="collection-directorate",
                            text_content=f"SOURCE: {url}\nLINK: {link}\n\n{ai_summary}",
                            title=f"RSS: {entry.title}"
                        )
                        
                        # Mark as seen
                        await self.redis.sadd(self.seen_key, link)
                        
            except Exception as e:
                print(f"[-] Failed to process feed {url}: {e}")

if __name__ == "__main__":
    ingress = RSSIngress()
    asyncio.run(ingress.process_feeds())
