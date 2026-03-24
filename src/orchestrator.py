import os
import json
import re
import asyncio
import logging
from pathlib import Path
import httpx
import redis.asyncio as redis
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.desks.anythingllm_client import AnythingLLMDesk
from src.gateways.adb_device import ADBDevice
from src.gateways.mcp_dispatcher import MCPDispatcher
from src.agents.social_media_agent import SocialMediaAgent

logger = logging.getLogger("osia.orchestrator")

# Domains that trigger the ADB media-capture pipeline
MEDIA_DOMAINS = ("instagram.com", "facebook.com", "tiktok.com", "youtube.com", "youtu.be")

# Valid desk slugs the Chief of Staff can route to
VALID_DESKS = {
    "geopolitical-and-security-desk",
    "cultural-and-theological-intelligence-desk",
    "science-technology-and-commercial-desk",
    "human-intelligence-and-profiling-desk",
    "finance-and-economics-directorate",
    "cyber-intelligence-and-warfare-desk",
}


class OsiaOrchestrator:
    """The central nervous system of OSIA. Routes tasks from Redis to the appropriate intelligence desks."""

    def __init__(self):
        load_dotenv()
        # Paths
        self.base_dir = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent))

        # Redis Queue
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis = redis.from_url(self.redis_url)
        self.queue_name = os.getenv("OSIA_TASK_QUEUE", "osia:task_queue")

        # Signal Gateway
        self.signal_api_url = os.getenv("SIGNAL_API_URL", "http://localhost:8081")
        self.signal_number = os.getenv("SIGNAL_SENDER_NUMBER")
        self._signal_client = httpx.AsyncClient(timeout=30.0)

        # Modern Gemini (Chief of Staff Logic)
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

        # API Clients
        self.desk_client = AnythingLLMDesk()
        self.adb = ADBDevice(device_id=os.getenv("ADB_DEVICE_MEDIA_INTERCEPT"))
        self.mcp = MCPDispatcher()
        self.social_agent = SocialMediaAgent(
            adb=self.adb,
            gemini_client=self.client,
            model_id=self.model_id,
            base_dir=self.base_dir,
        )

        # Define Research Tools for the Chief of Staff
        self.tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_wikipedia",
                        description="Search Wikipedia for baseline factual context on a topic.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={"query": types.Schema(type="STRING", description="The search term.")},
                            required=["query"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="search_arxiv",
                        description="Search ArXiv for academic papers and technical pre-prints.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={"query": types.Schema(type="STRING", description="The academic search query.")},
                            required=["query"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="search_semantic_scholar",
                        description="Search Semantic Scholar for peer-reviewed scientific literature and citations.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={"query": types.Schema(type="STRING", description="The scientific search query.")},
                            required=["query"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="get_youtube_transcript",
                        description="Retrieve the text transcript of a YouTube video for analysis.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={"url": types.Schema(type="STRING", description="The full YouTube video URL.")},
                            required=["url"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="get_current_time",
                        description="Get the current local time in UTC.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "timezone": types.Schema(
                                    type="STRING", description="The timezone name, use 'Etc/UTC'.", default="Etc/UTC"
                                )
                            },
                            required=["timezone"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="search_web",
                        description="Search the live web for current events, news, and real-time information using Tavily.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={"query": types.Schema(type="STRING", description="The search query.")},
                            required=["query"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="read_social_comments",
                        description="Use the physical phone to navigate to a social media post and extract all visible comments.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={"url": types.Schema(type="STRING", description="Direct URL to the social media post.")},
                            required=["url"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="post_social_comment",
                        description="Use the physical phone to post a comment on a social media post.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "url": types.Schema(type="STRING", description="Direct URL to the social media post."),
                                "comment": types.Schema(type="STRING", description="The comment text to post."),
                            },
                            required=["url", "comment"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="reply_social_comment",
                        description="Use the physical phone to reply to a specific comment on a social media post.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "url": types.Schema(type="STRING", description="Direct URL to the social media post."),
                                "target_author": types.Schema(type="STRING", description="Username of the comment author to reply to."),
                                "reply": types.Schema(type="STRING", description="The reply text."),
                            },
                            required=["url", "target_author", "reply"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="read_social_post",
                        description="Use the physical phone to navigate to a social media post and extract its full content, stats, and metadata.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={"url": types.Schema(type="STRING", description="Direct URL to the social media post.")},
                            required=["url"],
                        ),
                    ),
                ]
            )
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def shutdown(self):
        """Gracefully close shared clients."""
        await self._signal_client.aclose()
        await self.desk_client.close()
        await self.mcp.close_all()

    # ------------------------------------------------------------------
    # Signal messaging
    # ------------------------------------------------------------------

    async def send_signal_message(self, recipient: str, message: str):
        """Sends a Signal message back to the requester."""
        url = f"{self.signal_api_url}/v2/send"

        if not recipient.startswith("+") and not recipient.startswith("group."):
            recipient = f"group.{recipient}"

        payload = {
            "message": message,
            "number": self.signal_number,
            "recipients": [recipient],
        }
        logger.info("Sending intelligence report to %s via Signal...", recipient)
        try:
            response = await self._signal_client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Report delivered successfully.")
        except httpx.HTTPStatusError as e:
            logger.error("Signal API returned %s: %s", e.response.status_code, e.response.text)
        except httpx.RequestError as e:
            logger.error("Failed to reach Signal API: %s", e)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_forever(self):
        logger.info("OSIA Orchestrator online. Listening to Redis queue: %s", self.queue_name)
        while True:
            try:
                result = await self.redis.blpop(self.queue_name, timeout=0)
                if result:
                    _, task_json = result
                    task = json.loads(task_json)
                    await self.process_task(task)
            except redis.ConnectionError as e:
                logger.error("Redis connection lost: %s — retrying in 5s", e)
                await asyncio.sleep(5)
            except Exception as e:
                logger.exception("Unexpected error processing task: %s", e)

    # ------------------------------------------------------------------
    # MCP tool dispatch (maps Gemini function names → MCP calls)
    # ------------------------------------------------------------------

    async def _dispatch_tool(self, call) -> str | None:
        """Execute a single Gemini function-call via the appropriate MCP server."""
        name = call.name
        args = dict(call.args) if call.args else {}

        if name == "search_wikipedia":
            return await self.mcp.call_tool("wikipedia", "search_pages", {"input": {"query": args["query"]}})
        elif name == "search_arxiv":
            return await self.mcp.call_tool("arxiv", "search_papers", {"query": args["query"]})
        elif name == "search_semantic_scholar":
            return await self.mcp.call_tool("semantic-scholar", "search_paper", {"query": args["query"]})
        elif name == "get_youtube_transcript":
            return await self._extract_youtube_transcript(args["url"])
        elif name == "get_current_time":
            return await self.mcp.call_tool("time", "get_current_time", {"timezone": args.get("timezone", "Etc/UTC")})
        elif name == "search_web":
            return await self.mcp.call_tool("tavily", "tavily_search", {"query": args["query"]})
        elif name == "read_social_comments":
            result = await self.social_agent.read_comments(args["url"])
            return result.data if result.success else f"FAILED: {result.error}"
        elif name == "post_social_comment":
            result = await self.social_agent.post_comment(args["url"], args["comment"])
            return "Comment posted successfully." if result.success else f"FAILED: {result.error}"
        elif name == "reply_social_comment":
            result = await self.social_agent.reply_to_comment(args["url"], args["target_author"], args["reply"])
            return "Reply posted successfully." if result.success else f"FAILED: {result.error}"
        elif name == "read_social_post":
            result = await self.social_agent.read_post_content(args["url"])
            return result.data if result.success else f"FAILED: {result.error}"
        else:
            logger.warning("Unknown tool requested by Gemini: %s", name)
            return None

    # ------------------------------------------------------------------
    # YouTube transcript extraction (3-tier fallback)
    # ------------------------------------------------------------------

    async def _extract_youtube_transcript(self, video_url: str) -> str | None:
        logger.info("Attempting transcript extraction for: %s", video_url)
        mcp_res = None

        # 1. yt-dlp (software)
        try:
            yt_dlp_bin = self.base_dir / ".venv" / "bin" / "yt-dlp"
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            cmd = [
                str(yt_dlp_bin), "--skip-download",
                "--write-auto-subs", "--sub-lang", "en.*", "--convert-subs", "srt",
                "--output", "yt_intel", "--user-agent", user_agent, "--geo-bypass",
            ]
            cookies_path = self.base_dir / "config" / "youtube_cookies.txt"
            if cookies_path.exists():
                logger.info("Using YouTube Premium cookies.")
                cmd.extend(["--cookies", str(cookies_path)])
            cmd.append(video_url)

            proc = await asyncio.to_thread(
                __import__("subprocess").run, cmd,
                capture_output=True, text=True, cwd=str(self.base_dir),
            )
            srt_path = self.base_dir / "yt_intel.en.srt"
            if srt_path.exists():
                mcp_res = srt_path.read_text()
                srt_path.unlink()
        except Exception as e:
            logger.warning("yt-dlp failed: %s", e)

        # 2. MCP fallback
        if not mcp_res:
            logger.info("Software extraction failed. Falling back to MCP...")
            try:
                mcp_res = await self.mcp.call_tool("youtube", "get-transcript", {"url": video_url})
                if "Error: Request to YouTube was blocked" in str(mcp_res):
                    mcp_res = None
            except Exception as e:
                logger.warning("MCP YouTube fallback failed: %s", e)
                mcp_res = None

        # 3. Physical capture (PHINT)
        if not mcp_res:
            logger.warning("All software extraction blocked. Triggering PHINT capture...")
            try:
                mcp_res = await self.process_media_link(video_url)
                mcp_res = f"PHYSICAL INTERCEPT REPORT:\n{mcp_res}"
            except Exception as e:
                mcp_res = f"ERROR: All extraction methods failed, including physical capture. {e}"

        logger.info("YouTube intelligence length: %d", len(str(mcp_res)))
        return mcp_res

    # ------------------------------------------------------------------
    # Research loop — proper multi-turn tool calling
    # ------------------------------------------------------------------

    async def handle_research(self, query: str) -> str:
        """Executes a multi-turn research loop, feeding tool results back to Gemini."""
        logger.info("Chief of Staff initiating research loop for: %s", query)

        config = types.GenerateContentConfig(tools=self.tools)
        contents = [types.Content(role="user", parts=[types.Part(text=query)])]

        max_rounds = 5  # safety cap to avoid infinite tool loops
        for _round in range(max_rounds):
            response = self.client.models.generate_content(
                model=self.model_id, contents=contents, config=config,
            )

            candidate = response.candidates[0]
            # Append the model's response to the conversation
            contents.append(candidate.content)

            # Collect all function calls from this turn
            function_calls = [p for p in candidate.content.parts if p.function_call]
            if not function_calls:
                # Model is done calling tools — return its final text
                text_parts = [p.text for p in candidate.content.parts if p.text]
                return "\n".join(text_parts)

            # Execute each tool and build FunctionResponse parts
            response_parts = []
            for part in function_calls:
                call = part.function_call
                logger.info("Gemini requested tool: %s", call.name)
                try:
                    result = await self._dispatch_tool(call)
                    result_str = str(result) if result else "No results found."
                except Exception as e:
                    logger.error("Tool %s failed: %s", call.name, e)
                    result_str = f"Tool error: {e}"

                logger.info("Tool '%s' returned data (length: %d)", call.name, len(result_str))
                response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=call.name,
                        response={"result": result_str},
                    ))
                )

            # Feed tool results back to Gemini for the next round
            contents.append(types.Content(role="user", parts=response_parts))

        # If we exhausted rounds, summarize what we have
        logger.warning("Research loop hit max rounds (%d)", max_rounds)
        text_parts = [p.text for p in contents[-1].parts if hasattr(p, "text") and p.text]
        return "\n".join(text_parts) if text_parts else ""

    # ------------------------------------------------------------------
    # ADB media capture
    # ------------------------------------------------------------------

    async def process_media_link(self, url: str) -> str:
        """Uses ADB to capture media from a URL, then analyzes it with Gemini."""
        logger.info("Triggering ADB capture for URL: %s", url)
        await self.adb.open_url(url)
        await asyncio.sleep(5)
        remote_path = "/sdcard/osia_capture.mp4"
        local_path = str(self.base_dir / "osia_capture.mp4")
        await self.adb.record_screen(remote_path=remote_path, time_limit=15)
        await self.adb.pull_file(remote_path, local_path)

        logger.info("Uploading captured video to Gemini...")
        video_file = self.client.files.upload(file=local_path)
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            video_file = self.client.files.get(name=video_file.name)

        prompt = (
            "Watch this intercepted short-form video. Transcribe any spoken audio, "
            "describe the visual context, identify any text on screen, and summarize "
            "the core message or propaganda narrative."
        )
        response = self.client.models.generate_content(
            model=self.model_id, contents=[video_file, prompt]
        )

        capture_path = Path(local_path)
        if capture_path.exists():
            capture_path.unlink()
        return response.text

    # ------------------------------------------------------------------
    # Task processing
    # ------------------------------------------------------------------

    def _safe_title(self, text: str, max_len: int = 50) -> str:
        return (text[:max_len] + "...") if len(text) > max_len else text

    async def process_task(self, task: dict):
        """Main routing logic for an incoming OSINT task."""
        source = task.get("source", "unknown")
        original_query = task.get("query", "")
        query = original_query

        logger.info("Received new task from %s: %s", source, query)

        # 1. Media Link Interception
        url_match = re.search(r"(https?://[^\s]+)", query)
        if url_match:
            url = url_match.group(1)
            if any(domain in url.lower() for domain in MEDIA_DOMAINS):
                try:
                    media_analysis = await self.process_media_link(url)
                    await self.desk_client.ingest_raw_data(
                        "collection-directorate",
                        media_analysis,
                        f"Media Intercept: {self._safe_title(url)}",
                    )
                    query = f"A media intercept from {url} was just ingested. Context:\n{media_analysis}\n\nOriginal Request: {query}"
                except Exception as e:
                    logger.error("Media interception failed: %s", e)

        # 2. Automated Research (multi-turn tool calling)
        try:
            research_summary = await self.handle_research(original_query)
            if research_summary:
                logger.info("Research complete. Injecting into Collection Desk...")
                await self.desk_client.ingest_raw_data(
                    "collection-directorate",
                    research_summary,
                    f"Research: {self._safe_title(original_query)}",
                )
                query = f"Baseline research summary for this topic:\n{research_summary}\n\nOriginal Request: {query}"
        except Exception as e:
            logger.error("Automated research failed: %s", e)

        # 3. Chief of Staff routes to the appropriate desk
        directives_path = self.base_dir / "DIRECTIVES.md"
        mandate = directives_path.read_text()

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
        - cyber-intelligence-and-warfare-desk

        Reply with ONLY the slug of the desk, nothing else.
        """

        try:
            route_res = self.client.models.generate_content(model=self.model_id, contents=plan_prompt)
            assigned_desk = route_res.text.strip()

            # Validate the desk slug to avoid routing to a non-existent workspace
            if assigned_desk not in VALID_DESKS:
                logger.warning("Gemini returned invalid desk '%s', defaulting to geopolitical", assigned_desk)
                assigned_desk = "geopolitical-and-security-desk"

            logger.info("Task routed to: %s", assigned_desk)
            analysis = await self.desk_client.send_task(assigned_desk, query)
            logger.info("Intelligence synthesis complete.")

            if source.startswith("signal:"):
                recipient = source[len("signal:"):]
                await self.send_signal_message(recipient, analysis)

        except Exception as e:
            logger.exception("Orchestration or desk analysis failed: %s", e)
            if source.startswith("signal:"):
                recipient = source[len("signal:"):]
                
                error_type = type(e).__name__
                raw_error = str(e)
                
                # Basic sanitization to prevent leaking local paths or common sensitive patterns
                sanitized_error = re.sub(r'(/home/|/var/|/tmp/|C:\\)[^\s]+', '[REDACTED_PATH]', raw_error)
                sanitized_error = re.sub(r'sk-[A-Za-z0-9_-]{20,}', '[REDACTED_KEY]', sanitized_error)
                sanitized_error = re.sub(r'AIza[0-9A-Za-z-_]{35}', '[REDACTED_KEY]', sanitized_error)
                sanitized_error = re.sub(r'(https?://)[^\s]+', r'\1[REDACTED_URL]', sanitized_error)

                error_msg = (
                    f"⚠️ OSIA System Error ({error_type})\n\n"
                    f"An error occurred while processing your request:\n"
                    f"\"{sanitized_error}\"\n\n"
                    "This could be due to a temporary issue with our intelligence desks or a timeout. "
                    "Please try your query again later."
                )
                await self.send_signal_message(recipient, error_msg)
