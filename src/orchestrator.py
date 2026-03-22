import asyncio
import os
import json
import re
import httpx
import redis.asyncio as redis
from google import genai
from dotenv import load_dotenv
from src.desks.anythingllm_client import AnythingLLMDesk
from src.gateways.adb_device import ADBDevice

class OsiaOrchestrator:
    """The central nervous system of OSIA. Routes tasks from Redis to the appropriate intelligence desks."""
    
    def __init__(self):
        load_dotenv()
        # Redis Queue
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis = redis.from_url(self.redis_url)
        self.queue_name = os.getenv("OSIA_TASK_QUEUE", "osia:task_queue")
        
        # Signal Gateway
        self.signal_api_url = os.getenv("SIGNAL_API_URL", "http://localhost:8081")
        self.signal_number = os.getenv("SIGNAL_SENDER_NUMBER")
        if not self.signal_number:
            raise ValueError("SIGNAL_SENDER_NUMBER environment variable is required")
        
        # Modern Gemini (Chief of Staff Logic)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)
        self.model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
        
        # API Clients
        self.desk_client = AnythingLLMDesk()
        self.adb = ADBDevice()

    async def send_signal_message(self, recipient: str, message: str):
        """Sends a Signal message back to the requester."""
        url = f"{self.signal_api_url}/v2/send"
        payload = {
            "message": message,
            "number": self.signal_number,
            "recipients": [recipient]
        }
        print(f"[*] Sending intelligence report to {recipient} via Signal...")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                print("[+] Report delivered successfully.")
        except Exception as e:
            print(f"[-] Failed to send Signal message: {e}")

    async def run_forever(self):
        print(f"OSIA Orchestrator (Chief of Staff) is online. Listening to Redis queue: {self.queue_name}")
        while True:
            try:
                # Block until a task is pushed to the queue
                result = await self.redis.blpop(self.queue_name, timeout=0)
                if result:
                    _, task_json = result
                    task = json.loads(task_json)
                    await self.process_task(task)
            except Exception as e:
                print(f"Error processing task: {e}")

    async def process_media_link(self, url: str) -> str:
        """Uses ADB to capture media from a URL, then analyzes it with Gemini."""
        print(f"[*] Triggering ADB Capture for URL: {url}")
        
        # 1. Open the URL
        self.adb.open_url(url)
        
        # Give the app a moment to load and start playing
        await asyncio.sleep(5)
        
        # 2. Record the screen
        remote_path = "/sdcard/osia_capture.mp4"
        local_path = "osia_capture.mp4"
        self.adb.record_screen(remote_path=remote_path, time_limit=15)
        
        # 3. Pull the file
        self.adb.pull_file(remote_path, local_path)
        
        # 4. Upload to Gemini
        print("[*] Uploading captured video to Gemini...")
        video_file = self.client.files.upload(file=local_path)
        
        # Wait for processing if needed
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            await asyncio.sleep(2)
            video_file = self.client.files.get(name=video_file.name)
        print()
            
        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed.")
            
        # 5. Analyze with Gemini
        print("[*] Prompting Gemini for Analysis...")
        prompt = "Watch this intercepted short-form video. Transcribe any spoken audio, describe the visual context, identify any text on screen, and summarize the core message or propaganda narrative."
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[video_file, prompt]
        )
        
        # Clean up local file
        if os.path.exists(local_path):
            os.remove(local_path)
            
        return response.text

    async def process_task(self, task: dict):
        """Main routing logic for an incoming OSINT task."""
        source = task.get("source", "unknown")
        query = task.get("query", "")
        
        print(f"\n[+] Received new task from {source}: {query}")
        
        # Check for specific URLs for interception
        url_match = re.search(r'(https?://[^\s]+)', query)
        if url_match:
            url = url_match.group(1)
            if any(domain in url.lower() for domain in ['instagram.com', 'facebook.com', 'tiktok.com', 'youtube.com', 'youtu.be']):
                print(f"[*] Media URL detected: {url}")
                try:
                    media_analysis = await self.process_media_link(url)
                    print(f"[*] Media Analysis Complete. Ingesting into Collection Directorate...")
                    
                    await self.desk_client.ingest_raw_data(
                        workspace_slug="collection-directorate",
                        text_content=media_analysis,
                        title=f"Media Intercept: {url}"
                    )
                    
                    # Augment the query so the specialized desk knows what we found
                    query = f"A media intercept from {url} was just ingested. Context:\n{media_analysis}\n\nOriginal Request: {query}"
                except Exception as e:
                    print(f"[-] Media interception failed: {e}")
        
        # Step 1: Chief of Staff develops a plan
        plan_prompt = f"""
        You are the Chief of Staff for an Open Source Intelligence Agency (OSIA).
        A new Request for Information (RFI) has come in: '{query}'
        
        Which of our specialized desks should analyze this? Choose ONE:
        - geopolitical-and-security-desk
        - cultural-and-theological-intelligence-desk
        - science-technology-and-commercial-desk
        - human-intelligence-and-profiling-desk
        
        Reply with ONLY the slug of the desk, nothing else.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=plan_prompt
            )
            assigned_desk = response.text.strip()
            print(f"[*] Task routed to: {assigned_desk}")
            
            # Step 2: Query the assigned desk
            print(f"[*] Awaiting analysis from {assigned_desk}...")
            analysis = await self.desk_client.send_task(assigned_desk, query)
            print(f"\n[+] Intelligence Synthesis Complete:\n{analysis}")
            
            # Step 3: Log completion
            print("[*] Task processing finalized.")
            if source.startswith("signal:"):
                recipient_number = source.split(":")[1]
                await self.send_signal_message(recipient_number, analysis)
            
        except Exception as e:
            print(f"[-] Orchestration or Desk Analysis Failed: {e}")
