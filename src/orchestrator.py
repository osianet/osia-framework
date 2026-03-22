import os
import json
import re
import asyncio
import time
import httpx
import redis.asyncio as redis
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.desks.anythingllm_client import AnythingLLMDesk
from src.gateways.adb_device import ADBDevice
from src.gateways.mcp_dispatcher import MCPDispatcher

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
        
        # Modern Gemini (Chief of Staff Logic)
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
        
        # API Clients
        self.desk_client = AnythingLLMDesk()
        self.adb = ADBDevice()
        self.mcp = MCPDispatcher()

        # Define Research Tools for the Chief of Staff
        self.tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_wikipedia",
                        description="Search Wikipedia for baseline factual context on a topic.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "query": types.Schema(type="STRING", description="The search term.")
                            },
                            required=["query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_arxiv",
                        description="Search ArXiv for academic papers and technical pre-prints.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "query": types.Schema(type="STRING", description="The academic search query.")
                            },
                            required=["query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_semantic_scholar",
                        description="Search Semantic Scholar for peer-reviewed scientific literature and citations.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "query": types.Schema(type="STRING", description="The scientific search query.")
                            },
                            required=["query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="get_youtube_transcript",
                        description="Retrieve the text transcript of a YouTube video for analysis.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "url": types.Schema(type="STRING", description="The full YouTube video URL.")
                            },
                            required=["url"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="get_current_time",
                        description="Get the current local time in UTC.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "timezone": types.Schema(type="STRING", description="The timezone name, use 'Etc/UTC'.", default="Etc/UTC")
                            },
                            required=["timezone"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_web",
                        description="Search the live web for current events, news, and real-time information using Tavily.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "query": types.Schema(type="STRING", description="The search query.")
                            },
                            required=["query"]
                        )
                    )
                ]
            )
        ]

    async def send_signal_message(self, recipient: str, message: str):
        """Sends a Signal message back to the requester."""
        url = f"{self.signal_api_url}/v2/send"
        
        # Ensure group IDs are properly prefixed for the API
        if not recipient.startswith("+") and not recipient.startswith("group."):
            recipient = f"group.{recipient}"

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
                result = await self.redis.blpop(self.queue_name, timeout=0)
                if result:
                    _, task_json = result
                    task = json.loads(task_json)
                    await self.process_task(task)
            except Exception as e:
                print(f"Error processing task: {e}")

    async def handle_research(self, query: str) -> str:
        """Executes a research loop using MCP tools if Gemini requests them."""
        print(f"[*] Chief of Staff initiating research loop for: {query}")
        
        # Initial request to see if tools are needed
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=query,
            config=types.GenerateContentConfig(tools=self.tools)
        )

        research_results = []
        
        for part in response.candidates[0].content.parts:
            if part.function_call:
                call = part.function_call
                print(f"[*] Gemini requested tool: {call.name}")
                
                mcp_res = None
                if call.name == "search_wikipedia":
                    mcp_res = await self.mcp.call_tool("wikipedia", "search_pages", {"input": {"query": call.args["query"]}})
                elif call.name == "search_arxiv":
                    mcp_res = await self.mcp.call_tool("arxiv", "search_papers", {"query": call.args["query"]})
                elif call.name == "search_semantic_scholar":
                    mcp_res = await self.mcp.call_tool("semantic-scholar", "search_paper", {"query": call.args["query"]})
                elif call.name == "get_youtube_transcript":
                    mcp_res = await self.mcp.call_tool("youtube", "get-transcript", {"url": call.args["url"]})
                elif call.name == "get_current_time":
                    mcp_res = await self.mcp.call_tool("time", "get_current_time", {"timezone": call.args.get("timezone", "Etc/UTC")})
                elif call.name == "search_web":
                    mcp_res = await self.mcp.call_tool("tavily", "tavily_search", {"query": call.args["query"]})
                
                if mcp_res:
                    # Collect result and feed back to Gemini
                    research_results.append(str(mcp_res))
        
        if not research_results:
            return ""

        # Summarize findings
        summary_prompt = f"Summarize these raw intelligence findings into a collection report for our desks:\n\n" + "\n\n".join(research_results)
        summary_res = self.client.models.generate_content(model=self.model_id, contents=summary_prompt)
        return summary_res.text

    async def process_media_link(self, url: str) -> str:
        """Uses ADB to capture media from a URL, then analyzes it with Gemini."""
        print(f"[*] Triggering ADB Capture for URL: {url}")
        self.adb.open_url(url)
        await asyncio.sleep(5)
        remote_path = "/sdcard/osia_capture.mp4"
        local_path = "osia_capture.mp4"
        self.adb.record_screen(remote_path=remote_path, time_limit=15)
        self.adb.pull_file(remote_path, local_path)
        
        print("[*] Uploading captured video to Gemini...")
        video_file = self.client.files.upload(file=local_path)
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            video_file = self.client.files.get(name=video_file.name)
            
        prompt = "Watch this intercepted short-form video. Transcribe any spoken audio, describe the visual context, identify any text on screen, and summarize the core message or propaganda narrative."
        response = self.client.models.generate_content(model=self.model_id, contents=[video_file, prompt])
        
        if os.path.exists(local_path): os.remove(local_path)
        return response.text

    async def process_task(self, task: dict):
        """Main routing logic for an incoming OSINT task."""
        source = task.get("source", "unknown")
        original_query = task.get("query", "")
        query = original_query
        
        print(f"\n[+] Received new task from {source}: {query}")
        
        # 1. Media Link Interception
        url_match = re.search(r'(https?://[^\s]+)', query)
        if url_match:
            url = url_match.group(1)
            if any(domain in url.lower() for domain in ['instagram.com', 'facebook.com', 'tiktok.com', 'youtube.com', 'youtu.be']):
                try:
                    media_analysis = await self.process_media_link(url)
                    # Truncate title to avoid ENAMETOOLONG
                    safe_title = (url[:50] + '...') if len(url) > 50 else url
                    await self.desk_client.ingest_raw_data("collection-directorate", media_analysis, f"Media Intercept: {safe_title}")
                    query = f"A media intercept from {url} was just ingested. Context:\n{media_analysis}\n\nOriginal Request: {query}"
                except Exception as e:
                    print(f"[-] Media interception failed: {e}")

        # 2. Automated Research (Wikipedia/ArXiv)
        try:
            research_summary = await self.handle_research(original_query)
            if research_summary:
                print("[*] Research complete. Injecting into Collection Desk...")
                # Truncate title to avoid ENAMETOOLONG (AnythingLLM limits)
                safe_title = (original_query[:50] + '...') if len(original_query) > 50 else original_query
                await self.desk_client.ingest_raw_data("collection-directorate", research_summary, f"Research: {safe_title}")
                query = f"Baseline research summary for this topic:\n{research_summary}\n\nOriginal Request: {query}"
        except Exception as e:
            print(f"[-] Automated research failed: {e}")

        # Step 1: Chief of Staff develops a plan based on the Core Directives
        with open("DIRECTIVES.md", "r") as f:
            mandate = f.read()

        plan_prompt = f"""
        {mandate}

        ---

        You are the Chief of Staff for OSIA. A new Request for Information (RFI) has come in: '{query}'

        Based on the Socialist Intelligence Mandate above, which of our specialized desks should analyze this? Choose ONE:
        - geopolitical-and-security-desk
        - cultural-and-theological-intelligence-desk
        - science-technology-and-commercial-desk
        - human-intelligence-and-profiling-desk
        - finance-and-economics-directorate

        Reply with ONLY the slug of the desk, nothing else.
        """
        
        try:
            route_res = self.client.models.generate_content(model=self.model_id, contents=plan_prompt)
            assigned_desk = route_res.text.strip()
            print(f"[*] Task routed to: {assigned_desk}")
            
            print(f"[*] Awaiting analysis from {assigned_desk}...")
            analysis = await self.desk_client.send_task(assigned_desk, query)
            print(f"\n[+] Intelligence Synthesis Complete:\n{analysis}")
            
            if source.startswith("signal:"):
                recipient = source.split(":")[1]
                # If we received from a group, source will be signal:group.xyz
                # The send_signal_message already handles prefixing
                await self.send_signal_message(recipient, analysis)
            
        except Exception as e:
            print(f"[-] Orchestration or Desk Analysis Failed: {e}")
