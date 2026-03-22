import asyncio
import json
import os
import redis.asyncio as redis
from datetime import datetime
from dotenv import load_dotenv

async def trigger_sitrep():
    load_dotenv()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    queue_name = os.getenv("OSIA_TASK_QUEUE", "osia:task_queue")
    recipient = os.getenv("SIGNAL_SENDER_NUMBER") # Default to self

    print(f"[*] OSIA: Triggering Daily SITREP for {datetime.now().strftime('%Y-%m-%d')}...")

    task = {
        "source": f"signal:{recipient}",
        "query": (
            "Generate a Daily SITREP (Situational Report). Scour Wikipedia, ArXiv, and news sources "
            "for the most significant developments in the last 24 hours regarding: "
            "1. Global Geopolitics and Security, "
            "2. Global Financial Markets and Sanctions, "
            "3. Emerging Technology and AI Breakthroughs. "
            "Format the output as a formal Intelligence Summary (INTSUM)."
        )
    }

    r = redis.from_url(redis_url)
    await r.rpush(queue_name, json.dumps(task))
    print("[+] SITREP task pushed to queue successfully.")

if __name__ == "__main__":
    asyncio.run(trigger_sitrep())
