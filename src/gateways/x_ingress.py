"""
OSIA X/Twitter Ingress Gateway

Polls monitored X accounts via TwitterAPI.io, summarises new posts with Gemini,
extracts entities, upserts into Qdrant Collection Directorate, and stages
summaries in Redis for the daily SITREP.

Follows the same pattern as rss_ingress.py — direct Collection Directorate
ingestion rather than task queue push.

Environment variables:
  TWITTERAPI_IO_KEY    — API key for TwitterAPI.io
  GEMINI_API_KEY       — Google Gemini API key
  GEMINI_MODEL_ID      — Gemini model (default: gemini-2.5-flash)
  REDIS_URL            — Redis connection string
  OSIA_BASE_DIR        — Project root path
  X_POLL_INTERVAL      — Seconds between poll cycles (default: 900 = 15 min)
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import redis.asyncio as aioredis
import yaml
from dotenv import load_dotenv
from google import genai

from src.intelligence.entity_extractor import EntityExtractor
from src.intelligence.qdrant_store import QdrantStore

logger = logging.getLogger("osia.x_ingress")

# Redis keys
SEEN_KEY = "osia:x:seen_tweets"
DAILY_DIGEST_KEY = "osia:rss:daily_digest"  # shared with RSS — both feed the SITREP

TWITTERAPI_BASE = "https://api.twitterapi.io"

# Rate-limit settings
ACCOUNT_POLL_DELAY = 10.0  # seconds between each account poll — be kind to the API
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 10.0  # seconds — doubles each retry (10, 20, 40)

qdrant_store = QdrantStore()
entity_extractor = EntityExtractor()


class XAccount:
    """A single monitored X account."""

    def __init__(self, username: str, desk: str, label: str = ""):
        self.username = username
        self.desk = desk
        self.label = label or username

    def __repr__(self) -> str:
        return f"XAccount({self.username!r}, desk={self.desk!r})"


class XIngress:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("TWITTERAPI_IO_KEY", "")
        if not self.api_key:
            raise ValueError("TWITTERAPI_IO_KEY is not set — cannot start X ingress")

        self.redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
        self.poll_interval = int(os.getenv("X_POLL_INTERVAL", "900"))

        base_dir = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
        self.config_file = base_dir / "config" / "x_accounts.yaml"
        self.accounts: list[XAccount] = []

    def load_accounts(self) -> list[XAccount]:
        """Load monitored accounts from config/x_accounts.yaml."""
        if not self.config_file.exists():
            logger.warning("X accounts config not found at %s", self.config_file)
            return []

        with open(self.config_file) as f:
            raw = yaml.safe_load(f) or []

        accounts = []
        for entry in raw:
            username = entry.get("username", "").strip()
            desk = entry.get("desk", "").strip()
            if not username or not desk:
                logger.warning("Skipping invalid X account entry: %s", entry)
                continue
            accounts.append(XAccount(username=username, desk=desk, label=entry.get("label", "")))

        self.accounts = accounts
        logger.info("Loaded %d X accounts to monitor.", len(accounts))
        return accounts

    async def _fetch_latest_tweets(self, username: str) -> list[dict]:
        """Fetch the latest tweets for a user via TwitterAPI.io with retry on 429."""
        url = f"{TWITTERAPI_BASE}/twitter/user/last_tweets"
        headers = {"X-API-Key": self.api_key}
        params = {"userName": username, "includeReplies": "false"}

        for attempt in range(RETRY_ATTEMPTS):
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.get(url, headers=headers, params=params)

                if resp.status_code == 429:
                    delay = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Rate limited polling @%s (attempt %d/%d) — backing off %.0fs",
                        username,
                        attempt + 1,
                        RETRY_ATTEMPTS,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()

            if data.get("status") != "success":
                logger.warning("TwitterAPI.io error for @%s: %s", username, data.get("message", "unknown"))
                return []

            return data.get("tweets", [])

        logger.error("Rate limit exhausted for @%s after %d retries — skipping this cycle.", username, RETRY_ATTEMPTS)
        return []

    async def _process_tweet(self, tweet: dict, account: XAccount) -> bool:
        """Process a single tweet — summarise, extract entities, ingest. Returns True if new."""
        tweet_id = tweet.get("id", "")
        if not tweet_id:
            return False

        # Dedup check
        if await self.redis.sismember(SEEN_KEY, tweet_id):
            return False

        text = tweet.get("text", "")
        if not text.strip():
            await self.redis.sadd(SEEN_KEY, tweet_id)
            return False

        created_at = tweet.get("createdAt", "")
        tweet_url = tweet.get("url", f"https://x.com/{account.username}/status/{tweet_id}")

        # Engagement metrics
        likes = tweet.get("likeCount", 0)
        retweets = tweet.get("retweetCount", 0)
        replies = tweet.get("replyCount", 0)
        views = tweet.get("viewCount", 0)

        # Include quoted tweet text if present
        quoted = tweet.get("quoted_tweet")
        quoted_block = ""
        if quoted and quoted.get("text"):
            qt_author = quoted.get("author", {}).get("userName", "unknown")
            quoted_block = f"\n\nQuoted @{qt_author}: {quoted['text']}"

        logger.info("New X post from @%s: %s", account.username, text[:80])

        # Summarise with Gemini
        prompt = (
            "You are an intelligence analyst. A post from an official X/Twitter account has been intercepted.\n"
            "Summarise the intelligence value in 1-2 concise paragraphs. Focus on: what was announced, "
            "who is involved, strategic implications, and any data or figures mentioned.\n"
            "If the post is routine or low-value (e.g. holiday greetings), say so briefly.\n\n"
            f"Account: @{account.username} ({account.label})\n"
            f"Posted: {created_at}\n"
            f"Engagement: {views:,} views, {likes:,} likes, {retweets:,} retweets, {replies:,} replies\n\n"
            f"Post text:\n{text}{quoted_block}"
        )

        try:
            res = self.client.models.generate_content(model=self.model_id, contents=prompt)
            ai_summary = res.text
        except Exception as e:
            logger.error("Gemini summarisation failed for tweet %s: %s", tweet_id, e)
            ai_summary = text[:2000]

        # Build intelligence record
        intel_record = (
            f"TITLE: X post from @{account.username} ({account.label})\n"
            f"SOURCE: x.com/@{account.username}\n"
            f"LINK: {tweet_url}\n"
            f"POSTED: {created_at}\n"
            f"ENGAGEMENT: {views:,} views | {likes:,} likes | {retweets:,} RT | {replies:,} replies\n"
            f"DESK TARGET: {account.desk}\n"
            f"COLLECTED: {datetime.now(UTC).isoformat()}\n\n"
            f"ORIGINAL POST:\n{text}{quoted_block}\n\n"
            f"ANALYSIS:\n{ai_summary}"
        )

        # Entity extraction
        try:
            entities = await entity_extractor.extract(ai_summary, "collection-directorate")
        except Exception as e:
            logger.warning("Entity extraction failed for tweet %s: %s", tweet_id, e)
            entities = []

        # Enqueue research jobs for extracted entities
        await entity_extractor.enqueue_research_jobs(entities, triggered_by=tweet_url)

        # Upsert into Qdrant collection-directorate
        try:
            await qdrant_store.upsert(
                "collection-directorate",
                intel_record,
                metadata={
                    "desk": account.desk,
                    "topic": f"X/@{account.username}: {text[:120]}",
                    "source": f"x.com/@{account.username}",
                    "reliability_tier": "A",  # official accounts = high reliability
                    "timestamp": datetime.now(UTC).isoformat(),
                    "entity_tags": [e.name for e in entities],
                    "triggered_by": "x_ingress",
                    "x_tweet_id": tweet_id,
                    "x_engagement": json.dumps({"views": views, "likes": likes, "retweets": retweets}),
                },
            )
        except Exception as e:
            logger.error("Qdrant upsert failed for tweet %s: %s", tweet_id, e)

        # Stage in Redis daily digest (consumed by SITREP generator)
        await self.redis.rpush(DAILY_DIGEST_KEY, intel_record)

        # Mark as seen
        await self.redis.sadd(SEEN_KEY, tweet_id)
        return True

    async def process_accounts(self) -> int:
        """
        Poll all monitored X accounts, process new tweets.
        Returns the number of new tweets processed.
        """
        logger.info("X Ingress: Scanning monitored accounts...")
        accounts = self.load_accounts()
        if not accounts:
            logger.info("No X accounts configured. Exiting.")
            return 0

        new_count = 0

        for i, account in enumerate(accounts):
            # Stagger requests to avoid rate limiting
            if i > 0:
                await asyncio.sleep(ACCOUNT_POLL_DELAY)

            try:
                logger.info("Polling @%s (%s) → %s", account.username, account.label, account.desk)
                tweets = await self._fetch_latest_tweets(account.username)

                for tweet in tweets:
                    try:
                        if await self._process_tweet(tweet, account):
                            new_count += 1
                    except Exception as e:
                        logger.error("Failed to process tweet from @%s: %s", account.username, e)

            except httpx.HTTPStatusError as e:
                logger.error("TwitterAPI.io HTTP error for @%s: %s %s", account.username, e.response.status_code, e)
            except Exception as e:
                logger.error("Failed to poll @%s: %s", account.username, e)

        logger.info("X scan complete. %d new posts processed.", new_count)
        return new_count

    async def run_loop(self):
        """Run the polling loop continuously."""
        logger.info("X Ingress starting — polling every %d seconds.", self.poll_interval)
        while True:
            try:
                await self.process_accounts()
            except Exception as e:
                logger.exception("X Ingress cycle failed: %s", e)
            await asyncio.sleep(self.poll_interval)


async def main():
    ingress = XIngress()
    # Single run (for cron/manual invocation)
    await ingress.process_accounts()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
