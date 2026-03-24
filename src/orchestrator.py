import os
import json
import re
import asyncio
import logging
import subprocess
from pathlib import Path
import httpx
import redis.asyncio as redis
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.desks.anythingllm_client import AnythingLLMDesk
from src.desks.hf_endpoint_manager import HFEndpointManager
from src.gateways.adb_device import ADBDevice
from src.gateways.mcp_dispatcher import MCPDispatcher
from src.agents.social_media_agent import SocialMediaAgent
from src.intelligence.source_tracker import (
    SourceTracker,
    build_citation_protocol,
    audit_report,
)

logger = logging.getLogger("osia.orchestrator")


def _extract_mcp_text(result) -> str:
    """Extract plain text from an MCP CallToolResult."""
    if isinstance(result, str):
        return result
    if hasattr(result, "content") and result.content:
        return "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )
    return str(result)

# Domains that trigger the ADB media-capture pipeline
MEDIA_DOMAINS = ("instagram.com", "facebook.com", "tiktok.com")

# YouTube domains get transcript extraction instead of ADB screen-record
YOUTUBE_DOMAINS = ("youtube.com", "youtu.be")

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
        self.hf_endpoints = HFEndpointManager()
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

    async def send_signal_image(self, recipient: str, image_b64: str, caption: str = ""):
        """Sends a base64-encoded image attachment via Signal."""
        url = f"{self.signal_api_url}/v2/send"

        if not recipient.startswith("+") and not recipient.startswith("group."):
            recipient = f"group.{recipient}"

        payload = {
            "message": caption,
            "number": self.signal_number,
            "recipients": [recipient],
            "base64_attachments": [image_b64],
        }
        logger.info("Sending infographic to %s via Signal...", recipient)
        try:
            response = await self._signal_client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Infographic delivered successfully.")
        except httpx.HTTPStatusError as e:
            logger.error("Signal API returned %s sending image: %s", e.response.status_code, e.response.text)
        except httpx.RequestError as e:
            logger.error("Failed to reach Signal API for image: %s", e)

    async def generate_infographic(self, report_text: str) -> str | None:
        """
        Uses Gemini image generation to create a social-media-ready infographic
        summarising the key points from an intelligence report.

        Returns the image as a base64-encoded PNG string, or None on failure.
        """
        # Ask a text model to distil the report into a tight visual brief first.
        # This keeps the image prompt focused and avoids token-limit issues.
        brief_prompt = (
            "You are a graphic designer briefing an AI image generator. "
            "Read the following intelligence report and extract 4–6 key findings. "
            "Write a concise image-generation prompt (max 300 words) that describes "
            "a bold, dark-themed infographic card suitable for social media (9:16 portrait). "
            "The card should display the key findings as short bullet points with a strong "
            "headline, a dark background, and high-contrast accent colours (red/amber on dark). "
            "Include the label 'OSIA INTELLIGENCE BRIEF' at the top. "
            "Output ONLY the image prompt, no preamble.\n\n"
            f"REPORT:\n{report_text[:6000]}"
        )
        try:
            brief_res = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_id,
                contents=brief_prompt,
            )
            image_prompt = brief_res.text.strip()
            logger.info("Infographic brief generated (%d chars).", len(image_prompt))
        except Exception as e:
            logger.error("Failed to generate infographic brief: %s", e)
            return None

        image_model = os.getenv("GEMINI_IMAGE_MODEL_ID", "gemini-2.5-flash-image")
        try:
            img_res = await asyncio.to_thread(
                self.client.models.generate_content,
                model=image_model,
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="9:16"),
                ),
            )
            for part in img_res.parts:
                if part.thought:
                    continue
                if part.inline_data is not None:
                    import base64
                    return base64.b64encode(part.inline_data.data).decode()
            logger.warning("Gemini image generation returned no image parts.")
        except Exception as e:
            # High-demand / capacity errors — fall back to driving the Gemini Android app
            _capacity_markers = ("high demand", "overloaded", "503", "resource_exhausted", "RESOURCE_EXHAUSTED")
            if any(m.lower() in str(e).lower() for m in _capacity_markers):
                logger.warning("Gemini image API over capacity (%s) — trying phone fallback.", e)
                return await self._generate_infographic_via_phone(image_prompt)
            logger.error("Infographic image generation failed: %s", e)
        return None

    async def _generate_infographic_via_phone(self, image_prompt: str) -> str | None:
        """Drives the Gemini Android app via ADB to generate an infographic image."""
        import base64
        try:
            png_bytes = await self.social_agent.generate_infographic_via_phone(image_prompt)
            if png_bytes:
                return base64.b64encode(png_bytes).decode()
        except Exception as e:
            logger.error("Phone infographic fallback failed: %s", e)
        return None

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
            return _extract_mcp_text(await self.mcp.call_tool("wikipedia", "search_pages", {"input": {"query": args["query"]}}))
        elif name == "search_arxiv":
            return _extract_mcp_text(await self.mcp.call_tool("arxiv", "search_papers", {"query": args["query"]}))
        elif name == "search_semantic_scholar":
            return _extract_mcp_text(await self.mcp.call_tool("semantic-scholar", "search_paper", {"query": args["query"]}))
        elif name == "get_youtube_transcript":
            return await self._extract_youtube_transcript(args["url"])
        elif name == "get_current_time":
            return _extract_mcp_text(await self.mcp.call_tool("time", "get_current_time", {"timezone": args.get("timezone", "Etc/UTC")}))
        elif name == "search_web":
            return _extract_mcp_text(await self.mcp.call_tool("tavily", "tavily_search", {"query": args["query"]}))
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
        transcript = None

        # 1. yt-dlp (local software, fastest)
        try:
            tmp_dir = self.base_dir / "tmp"
            tmp_dir.mkdir(exist_ok=True)
            yt_dlp_bin = self.base_dir / ".venv" / "bin" / "yt-dlp"
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            cmd = [
                str(yt_dlp_bin), "--skip-download",
                "--write-auto-subs", "--sub-lang", "en.*", "--convert-subs", "srt",
                "--output", str(tmp_dir / "yt_intel"), "--user-agent", user_agent, "--geo-bypass",
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
            srt_path = tmp_dir / "yt_intel.en.srt"
            if srt_path.exists():
                transcript = srt_path.read_text()
                srt_path.unlink()
            else:
                logger.warning(
                    "yt-dlp exited %d but no SRT produced.\nstdout: %s\nstderr: %s",
                    proc.returncode,
                    proc.stdout[-500:] if proc.stdout else "(empty)",
                    proc.stderr[-500:] if proc.stderr else "(empty)",
                )
            # Clean up any leftover subtitle files (vtt, srt, etc.)
            for leftover in tmp_dir.glob("yt_intel*"):
                leftover.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("yt-dlp failed: %s", e)

        # 2. youtube-transcript-api (pure Python, no MCP overhead)
        if not transcript:
            logger.info("yt-dlp failed. Trying youtube-transcript-api...")
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                # Extract video ID from URL
                video_id = None
                if "youtu.be/" in video_url:
                    video_id = video_url.split("youtu.be/")[1].split("?")[0]
                elif "v=" in video_url:
                    video_id = video_url.split("v=")[1].split("&")[0]
                if video_id:
                    ytt_api = YouTubeTranscriptApi()
                    fetched = await asyncio.to_thread(
                        ytt_api.fetch, video_id, languages=["en"]
                    )
                    transcript = "\n".join(
                        f"[{snippet.start:.1f}s] {snippet.text}" for snippet in fetched
                    )
            except Exception as e:
                logger.warning("youtube-transcript-api failed: %s", e)

        # 3. MCP YouTube tool (Node.js server)
        if not transcript:
            logger.info("Python extraction failed. Falling back to MCP...")
            try:
                result = await self.mcp.call_tool("youtube", "get-transcript", {"url": video_url})
                transcript = _extract_mcp_text(result)
                if transcript and "Error: Request to YouTube was blocked" in transcript:
                    transcript = None
            except Exception as e:
                logger.warning("MCP YouTube fallback failed: %s", e)

        # 4. Gemini native YouTube URL (direct Google access, uses tokens)
        if not transcript:
            logger.info("All scraping methods failed. Using Gemini native YouTube analysis...")
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=types.Content(
                        parts=[
                            types.Part(
                                file_data=types.FileData(file_uri=video_url)
                            ),
                            types.Part(text=(
                                "Produce a detailed transcript of this video. Include timestamps "
                                "in [MM:SS] format. Capture all spoken words verbatim, describe "
                                "key visual elements, and note any on-screen text."
                            )),
                        ]
                    ),
                )
                transcript = response.text
            except Exception as e:
                logger.warning("Gemini native YouTube analysis failed: %s", e)

        # 5. Physical ADB capture (last resort)
        if not transcript:
            logger.warning("All extraction methods blocked. Triggering PHINT capture...")
            try:
                transcript = await self.process_media_link(video_url)
                transcript = f"PHYSICAL INTERCEPT REPORT:\n{transcript}"
            except Exception as e:
                transcript = f"ERROR: All extraction methods failed, including physical capture. {e}"

        logger.info("YouTube intelligence length: %d", len(str(transcript)))
        return transcript

    # ------------------------------------------------------------------
    # Research loop — proper multi-turn tool calling
    # ------------------------------------------------------------------

    async def handle_research(self, query: str) -> tuple[str, SourceTracker]:
        """Executes a multi-turn research loop, feeding tool results back to Gemini.

        Returns the final research text and a SourceTracker containing
        provenance metadata for every tool call made during the loop.
        """
        logger.info("Chief of Staff initiating research loop for: %s", query)
        tracker = SourceTracker()

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
                return "\n".join(text_parts), tracker

            # Execute each tool and build FunctionResponse parts
            response_parts = []
            for part in function_calls:
                call = part.function_call
                logger.info("Gemini requested tool: %s", call.name)
                tool_query = ""
                if call.args:
                    tool_query = call.args.get("query", call.args.get("url", ""))
                try:
                    result = await self._dispatch_tool(call)
                    result_str = str(result) if result else "No results found."
                except Exception as e:
                    logger.error("Tool %s failed: %s", call.name, e)
                    result_str = f"Tool error: {e}"

                # Record source provenance
                tracker.record(call.name, tool_query, result_str)

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
        return ("\n".join(text_parts) if text_parts else ""), tracker

    # ------------------------------------------------------------------
    # ADB media capture
    # ------------------------------------------------------------------

    async def _detect_video_duration(self, url: str) -> int | None:
        """
        Use yt-dlp --dump-json to fetch video metadata without downloading.
        Returns duration in seconds, or None if unavailable.
        Much more reliable than trying to read a timestamp from a screenshot.
        """
        yt_dlp_bin = self.base_dir / ".venv" / "bin" / "yt-dlp"
        if not yt_dlp_bin.exists():
            yt_dlp_bin = Path("yt-dlp")  # fall back to system PATH

        cmd = [str(yt_dlp_bin), "--dump-json", "--no-playlist", url]
        try:
            proc = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                duration = data.get("duration")
                if duration:
                    return int(duration)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.warning("yt-dlp duration probe failed: %s", e)
        return None

    async def process_media_link(self, url: str) -> str:
        """Uses ADB to capture media from a URL, then analyzes it with Gemini."""
        logger.info("Triggering ADB capture for URL: %s", url)

        _FALLBACK_DURATION = 60  # sane fallback for short-form content
        _MAX_DURATION = 180      # hard cap — no point recording more than 3 min
        _BUFFER_SECS = 3

        # Probe duration via yt-dlp metadata before touching the phone
        detected = await self._detect_video_duration(url)
        if detected and 3 <= detected <= _MAX_DURATION:
            record_duration = min(detected + _BUFFER_SECS, _MAX_DURATION)
            logger.info("Detected video duration: %ds — recording for %ds", detected, record_duration)
        else:
            record_duration = _FALLBACK_DURATION
            logger.info("Could not detect duration (got %r) — using fallback %ds", detected, record_duration)

        await self.adb.open_url(url)
        await asyncio.sleep(5)  # wait for player to render

        # TTL covers the full record window plus upload headroom.
        lock_ttl = record_duration + 120
        await self.redis.set("osia:adb:lock", "orchestrator", ex=lock_ttl)
        try:
            remote_path = "/sdcard/osia_capture.mp4"
            local_path = str(self.base_dir / "osia_capture.mp4")
            await self.adb.record_screen(remote_path=remote_path, time_limit=record_duration)
            await self.adb.pull_file(remote_path, local_path)
        finally:
            await self.redis.delete("osia:adb:lock")

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
        media_analysis = None
        research_summary = None
        source_tracker: SourceTracker | None = None

        logger.info("Received new task from %s: %s", source, query)

        # 1. Media Link Interception
        url_match = re.search(r"(https?://[^\s]+)", query)
        if url_match:
            url = url_match.group(1)
            url_lower = url.lower()

            # YouTube → transcript extraction (full-length support)
            if any(domain in url_lower for domain in YOUTUBE_DOMAINS):
                try:
                    media_analysis = await self._extract_youtube_transcript(url)
                    if media_analysis:
                        await self.desk_client.ingest_raw_data(
                            "collection-directorate",
                            media_analysis,
                            f"YouTube Transcript: {self._safe_title(url)}",
                        )
                        query = f"A YouTube transcript from {url} was just ingested. Context:\n{media_analysis}\n\nOriginal Request: {query}"
                except Exception as e:
                    logger.error("YouTube transcript extraction failed: %s", e)

            # Other social media → ADB screen-record capture
            elif any(domain in url_lower for domain in MEDIA_DOMAINS):
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
            research_summary, source_tracker = await self.handle_research(original_query)
            if research_summary:
                logger.info("Research complete. Injecting into Collection Desk...")
                await self.desk_client.ingest_raw_data(
                    "collection-directorate",
                    research_summary,
                    f"Research: {self._safe_title(original_query)}",
                )
                # Build source manifest for desk consumption
                manifest = source_tracker.format_manifest() if source_tracker else ""
                citation_block = build_citation_protocol()
                query = (
                    f"Baseline research summary for this topic:\n{research_summary}\n\n"
                    f"{manifest}\n\n"
                    f"{citation_block}\n\n"
                    f"Original Request: {query}"
                )
        except Exception as e:
            logger.error("Automated research failed: %s", e)

        # 3. Chief of Staff routes to the appropriate desk
        directives_path = self.base_dir / "DIRECTIVES.md"
        mandate = directives_path.read_text()

        plan_prompt = f"""
        {mandate}

        ---

        You are the Chief of Staff for OSIA. A new Request for Information (RFI) has come in: '{query}'

        Which of our specialized desks should analyze this? Choose ONE:
        - geopolitical-and-security-desk: Geopolitical forecasting, conflict analysis, and national sovereignty. (Has Country Intel tools)
        - cultural-and-theological-intelligence-desk: Sociological drivers, religious movements, and cultural drivers. (Has Cultural Observatory)
        - science-technology-and-commercial-desk: Technical breakthroughs, software analysis, and ecological tech. (Has GitHub Intel)
        - human-intelligence-and-profiling-desk: Behavioral profiling, digital personas, and tracking individuals. (Has Username Recon)
        - finance-and-economics-directorate: Capital flows, labor rights, and market dynamics. (Has Stock Market Intel)
        - cyber-intelligence-and-warfare-desk: Digital infrastructure, state-sponsored cyber-warfare, and technical network reconnaissance. (Has Kali Linux Sandbox for Nmap/Whois/Dig)

        Reply with ONLY the slug of the desk, nothing else.
        """

        try:
            route_res = self.client.models.generate_content(model=self.model_id, contents=plan_prompt)
            assigned_desk = (route_res.text or "").strip()

            # Validate the desk slug to avoid routing to a non-existent workspace
            if assigned_desk not in VALID_DESKS:
                logger.warning("Gemini returned invalid desk '%s', defaulting to geopolitical", assigned_desk)
                assigned_desk = "geopolitical-and-security-desk"

            logger.info("Task routed to: %s", assigned_desk)

            # Ingest media analysis into the assigned desk so it's available in its vector DB
            if media_analysis:
                try:
                    await self.desk_client.ingest_raw_data(
                        assigned_desk,
                        media_analysis,
                        f"Media Intercept: {self._safe_title(url)}",
                    )
                except Exception as e:
                    logger.warning("Failed to ingest media into desk '%s': %s", assigned_desk, e)

            # Ingest research summary into the assigned desk
            if research_summary:
                try:
                    await self.desk_client.ingest_raw_data(
                        assigned_desk,
                        research_summary,
                        f"Research: {self._safe_title(original_query)}",
                    )
                except Exception as e:
                    logger.warning("Failed to ingest research into desk '%s': %s", assigned_desk, e)

            # Wake up HF endpoint if this desk uses one (no-op for cloud-API desks)
            hf_ready = await self.hf_endpoints.ensure_ready(assigned_desk)
            if not hf_ready:
                logger.warning("HF endpoint for '%s' not ready — proceeding with existing desk config", assigned_desk)

            try:
                analysis = await self.desk_client.send_task(assigned_desk, query)
            except Exception as desk_err:
                logger.warning(
                    "Desk '%s' failed (%s), falling back to Gemini direct analysis",
                    assigned_desk, desk_err,
                )
                # Load the desk-specific prompt template if available
                desk_prompt_path = self.base_dir / "templates" / "prompts" / f"{assigned_desk}.txt"
                desk_persona = ""
                if desk_prompt_path.exists():
                    desk_persona = desk_prompt_path.read_text()

                fallback_prompt = (
                    f"{mandate}\n\n"
                    f"--- DESK PERSONA ---\n{desk_persona}\n\n" if desk_persona else f"{mandate}\n\n"
                ) + (
                    f"You are acting as the {assigned_desk} for OSIA. "
                    f"Analyze the following intelligence request and provide a thorough report.\n\n"
                    f"{build_citation_protocol()}\n\n"
                    f"{query}"
                )
                fallback_res = self.client.models.generate_content(
                    model=self.model_id, contents=fallback_prompt
                )
                analysis = fallback_res.text
                logger.info("Gemini fallback analysis complete (desk '%s' was bypassed).", assigned_desk)

            logger.info("Intelligence synthesis complete.")

            # Audit citations and append source quality summary
            analysis += audit_report(analysis, source_tracker)

            if source.startswith("signal:"):
                recipient = source[len("signal:"):]
                await self.send_signal_message(recipient, analysis)
                # Generate and send a social-media infographic alongside the text report
                try:
                    infographic_b64 = await self.generate_infographic(analysis)
                    if infographic_b64:
                        await self.send_signal_image(recipient, infographic_b64, caption="📊 OSIA Intelligence Brief")
                except Exception as img_err:
                    logger.warning("Infographic delivery failed (non-fatal): %s", img_err)

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
