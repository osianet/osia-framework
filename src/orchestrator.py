import asyncio
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx
import redis.asyncio as redis
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.agents.social_media_agent import SocialMediaAgent
from src.desks.desk_registry import DeskRegistry
from src.desks.hf_endpoint_manager import HFEndpointManager
from src.gateways.adb_device import ADBDevice
from src.gateways.mcp_dispatcher import MCPDispatcher
from src.intelligence.entity_extractor import EntityExtractor
from src.intelligence.qdrant_store import QdrantStore
from src.intelligence.report_generator import generate_intsum_pdf
from src.intelligence.source_tracker import (
    SourceTracker,
    audit_report,
    build_citation_protocol,
)

logger = logging.getLogger("osia.orchestrator")


def _extract_mcp_text(result) -> str:
    """Extract plain text from an MCP CallToolResult."""
    if isinstance(result, str):
        return result
    if hasattr(result, "content") and result.content:
        return "\n".join(block.text for block in result.content if hasattr(block, "text"))
    return str(result)


# Domains that trigger the ADB media-capture pipeline
MEDIA_DOMAINS = ("instagram.com", "facebook.com", "tiktok.com")

# YouTube domains get transcript extraction instead of ADB screen-record
YOUTUBE_DOMAINS = ("youtube.com", "youtu.be")

# Qdrant collections bootstrapped at startup — names match desk YAML slugs
BOOTSTRAP_COLLECTIONS = [
    "collection-directorate",
    "geopolitical-and-security-desk",
    "cultural-and-theological-intelligence-desk",
    "science-technology-and-commercial-desk",
    "human-intelligence-and-profiling-desk",
    "finance-and-economics-directorate",
    "cyber-intelligence-and-warfare-desk",
    "the-watch-floor",
    "osia_research_cache",
    "epstein-files",
    "cybersecurity-attacks",
    "hackerone-reports",
    "iran-israel-war-2026",
]


def _load_default_desk(base_dir: Path) -> str:
    """Load default_desk from config/osia.yaml, falling back to the-watch-floor."""
    config_path = base_dir / "config" / "osia.yaml"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("default_desk", "the-watch-floor")
    except FileNotFoundError:
        return "the-watch-floor"
    except Exception as e:
        logger.warning("Failed to load config/osia.yaml: %s — using default desk", e)
        return "the-watch-floor"


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

        # Modern Gemini (media analysis, research loop, image generation)
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

        # Venice (desk routing) — uncensored so sensitive queries are never refused or misrouted
        self._venice_base_url = "https://api.venice.ai/api/v1"
        self._venice_api_key = os.getenv("VENICE_API_KEY", "")
        self._routing_model = os.getenv("VENICE_ROUTING_MODEL", "venice-uncensored")

        # New architecture: DeskRegistry, QdrantStore, EntityExtractor
        self.desk_registry = DeskRegistry()
        self.qdrant = QdrantStore()
        self.entity_extractor = EntityExtractor()

        # Populate valid desks dynamically from registry
        self.valid_desks: set[str] = set(self.desk_registry.list_slugs())

        # Load default desk from config/osia.yaml
        self.default_desk = _load_default_desk(self.base_dir)

        # Other infrastructure
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
                            properties={
                                "query": types.Schema(type="STRING", description="The scientific search query.")
                            },
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
                            properties={
                                "url": types.Schema(type="STRING", description="Direct URL to the social media post.")
                            },
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
                                "target_author": types.Schema(
                                    type="STRING", description="Username of the comment author to reply to."
                                ),
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
                            properties={
                                "url": types.Schema(type="STRING", description="Direct URL to the social media post.")
                            },
                            required=["url"],
                        ),
                    ),
                ]
            )
        ]

    # ------------------------------------------------------------------
    # Startup bootstrap
    # ------------------------------------------------------------------

    async def bootstrap(self) -> None:
        """Bootstrap Qdrant collections and log startup state."""
        logger.info("Bootstrapping Qdrant collections...")
        for collection in BOOTSTRAP_COLLECTIONS:
            await self.qdrant.ensure_collection(collection)
        logger.info(
            "Startup complete. Valid desks: %s. Default desk: %s",
            sorted(self.valid_desks),
            self.default_desk,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def shutdown(self):
        """Gracefully close shared clients."""
        await self._signal_client.aclose()
        await self.desk_registry.close()
        await self.mcp.close_all()

    # ------------------------------------------------------------------
    # Signal messaging
    # ------------------------------------------------------------------

    async def _signal_post(
        self, payload: dict, label: str = "message", retries: int = 5, retry_delay: int = 30
    ) -> bool:
        """POST to the Signal API with retry logic. Returns True on success."""
        url = f"{self.signal_api_url}/v2/send"
        for attempt in range(1, retries + 1):
            try:
                response = await self._signal_client.post(url, json=payload)
                response.raise_for_status()
                logger.info("Signal %s delivered successfully.", label)
                return True
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Signal API returned %s for %s (attempt %d/%d): %s",
                    e.response.status_code,
                    label,
                    attempt,
                    retries,
                    e.response.text,
                )
            except httpx.RequestError as e:
                logger.error(
                    "Failed to reach Signal API for %s (attempt %d/%d): %s",
                    label,
                    attempt,
                    retries,
                    e,
                )
            if attempt < retries:
                logger.info("Retrying Signal %s in %ds...", label, retry_delay)
                await asyncio.sleep(retry_delay)
        logger.error("Signal %s failed after %d attempts — giving up.", label, retries)
        return False

    async def send_signal_message(self, recipient: str, message: str):
        """Sends a Signal message back to the requester."""
        if not recipient.startswith("+") and not recipient.startswith("group."):
            recipient = f"group.{recipient}"
        payload = {
            "message": message,
            "number": self.signal_number,
            "recipients": [recipient],
        }
        logger.info("Sending intelligence report to %s via Signal...", recipient)
        await self._signal_post(payload, label="message")

    async def send_signal_image(self, recipient: str, image_b64: str, caption: str = ""):
        """Sends a base64-encoded image attachment via Signal."""
        if not recipient.startswith("+") and not recipient.startswith("group."):
            recipient = f"group.{recipient}"
        payload = {
            "message": caption,
            "number": self.signal_number,
            "recipients": [recipient],
            "base64_attachments": [image_b64],
        }
        logger.info("Sending infographic to %s via Signal...", recipient)
        await self._signal_post(payload, label="image")

    async def generate_infographic(self, report_text: str) -> str | None:
        """
        Uses Gemini image generation to create a social-media-ready infographic
        summarising the key points from an intelligence report.

        Returns the image as a base64-encoded PNG string, or None on failure.
        """
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
            image_prompt = (brief_res.text or "").strip()
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
        await self.bootstrap()
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
            return _extract_mcp_text(
                await self.mcp.call_tool("wikipedia", "search_pages", {"input": {"query": args["query"]}})
            )
        elif name == "search_arxiv":
            return _extract_mcp_text(await self.mcp.call_tool("arxiv", "search_papers", {"query": args["query"]}))
        elif name == "search_semantic_scholar":
            return _extract_mcp_text(
                await self.mcp.call_tool("semantic-scholar", "search_paper", {"query": args["query"]})
            )
        elif name == "get_youtube_transcript":
            return await self._extract_youtube_transcript(args["url"])
        elif name == "get_current_time":
            return _extract_mcp_text(
                await self.mcp.call_tool("time", "get_current_time", {"timezone": args.get("timezone", "Etc/UTC")})
            )
        elif name == "search_web":
            return _extract_mcp_text(await self.mcp.call_tool("tavily", "tavily_search", {"query": args["query"]}))
        elif name == "read_social_comments":
            # Try yt-dlp first — much faster and doesn't use the phone
            metadata = await self._fetch_social_metadata(args["url"])
            if metadata:
                logger.info("read_social_comments: served from yt-dlp metadata")
                return metadata
            logger.info("read_social_comments: yt-dlp unavailable, falling back to ADB")
            result = await self.social_agent.read_comments(args["url"])
            return result.data if result.success else f"FAILED: {result.error}"
        elif name == "post_social_comment":
            result = await self.social_agent.post_comment(args["url"], args["comment"])
            return "Comment posted successfully." if result.success else f"FAILED: {result.error}"
        elif name == "reply_social_comment":
            result = await self.social_agent.reply_to_comment(args["url"], args["target_author"], args["reply"])
            return "Reply posted successfully." if result.success else f"FAILED: {result.error}"
        elif name == "read_social_post":
            # Try yt-dlp first — returns caption, author, likes and comments without ADB
            metadata = await self._fetch_social_metadata(args["url"])
            if metadata:
                logger.info("read_social_post: served from yt-dlp metadata")
                return metadata
            logger.info("read_social_post: yt-dlp unavailable, falling back to ADB")
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
                str(yt_dlp_bin),
                "--skip-download",
                "--write-auto-subs",
                "--sub-lang",
                "en.*",
                "--convert-subs",
                "srt",
                "--output",
                str(tmp_dir / "yt_intel"),
                "--user-agent",
                user_agent,
                "--geo-bypass",
            ]
            cookies_path = self.base_dir / "config" / "youtube_cookies.txt"
            if cookies_path.exists():
                logger.info("Using YouTube Premium cookies.")
                cmd.extend(["--cookies", str(cookies_path)])
            cmd.append(video_url)

            proc = await asyncio.to_thread(
                __import__("subprocess").run,
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
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
            for leftover in tmp_dir.glob("yt_intel*"):
                leftover.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("yt-dlp failed: %s", e)

        # 2. youtube-transcript-api (pure Python, no MCP overhead)
        if not transcript:
            logger.info("yt-dlp failed. Trying youtube-transcript-api...")
            try:
                from youtube_transcript_api import YouTubeTranscriptApi

                video_id = None
                if "youtu.be/" in video_url:
                    video_id = video_url.split("youtu.be/")[1].split("?")[0]
                elif "v=" in video_url:
                    video_id = video_url.split("v=")[1].split("&")[0]
                if video_id:
                    ytt_api = YouTubeTranscriptApi()
                    fetched = await asyncio.to_thread(ytt_api.fetch, video_id, languages=["en"])
                    transcript = "\n".join(f"[{snippet.start:.1f}s] {snippet.text}" for snippet in fetched)
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
                            types.Part(file_data=types.FileData(file_uri=video_url)),
                            types.Part(
                                text=(
                                    "Produce a detailed transcript of this video. Include timestamps "
                                    "in [MM:SS] format. Capture all spoken words verbatim, describe "
                                    "key visual elements, and note any on-screen text."
                                )
                            ),
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
                transcript, _ = await self.process_media_link(video_url)
                transcript = f"PHYSICAL INTERCEPT REPORT:\n{transcript}"
            except Exception as e:
                transcript = f"ERROR: All extraction methods failed, including physical capture. {e}"

        logger.info("YouTube intelligence length: %d", len(str(transcript)))
        return transcript

    # ------------------------------------------------------------------
    # Research loop — proper multi-turn tool calling
    # ------------------------------------------------------------------

    async def handle_research(self, query: str) -> tuple[str, SourceTracker]:
        """Executes a multi-turn research loop, feeding tool results back to Gemini."""
        logger.info("Chief of Staff initiating research loop for: %s", query)
        tracker = SourceTracker()

        config = types.GenerateContentConfig(tools=self.tools)
        contents = [types.Content(role="user", parts=[types.Part(text=query)])]

        max_rounds = 5
        for _round in range(max_rounds):
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )

            candidate = response.candidates[0]
            contents.append(candidate.content)

            function_calls = [p for p in candidate.content.parts if p.function_call]
            if not function_calls:
                text_parts = [p.text for p in candidate.content.parts if p.text]
                return "\n".join(text_parts), tracker

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

                tracker.record(call.name, tool_query, result_str)
                logger.info("Tool '%s' returned data (length: %d)", call.name, len(result_str))
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=call.name,
                            response={"result": result_str},
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=response_parts))

        logger.warning("Research loop hit max rounds (%d)", max_rounds)
        text_parts = [p.text for p in contents[-1].parts if hasattr(p, "text") and p.text]
        return ("\n".join(text_parts) if text_parts else ""), tracker

    # ------------------------------------------------------------------
    # ADB media capture
    # ------------------------------------------------------------------

    async def _fetch_yt_dlp_metadata(self, url: str) -> dict | None:
        """Run yt-dlp --dump-json and return the parsed metadata dict, or None on failure."""
        yt_dlp_bin = self.base_dir / ".venv" / "bin" / "yt-dlp"
        if not yt_dlp_bin.exists():
            yt_dlp_bin = Path("yt-dlp")

        cmd = [
            str(yt_dlp_bin),
            "--dump-json",
            "--no-playlist",
            "--write-comments",
            "--extractor-args",
            "youtube:max_comments=50",
        ]
        # Inject platform-specific cookie files when available
        _COOKIE_FILES = {
            "instagram.com": self.base_dir / "config" / "instagram_cookies.txt",
            "youtube.com": self.base_dir / "config" / "youtube_cookies.txt",
            "youtu.be": self.base_dir / "config" / "youtube_cookies.txt",
        }
        for domain, cookie_path in _COOKIE_FILES.items():
            if domain in url and cookie_path.exists():
                cmd.extend(["--cookies", str(cookie_path)])
                break
        cmd.append(url)
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return json.loads(proc.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.warning("yt-dlp metadata fetch failed: %s", e)
        return None

    async def _detect_video_duration(self, url: str) -> int | None:
        """Use yt-dlp --dump-json to fetch video duration without downloading."""
        data = await self._fetch_yt_dlp_metadata(url)
        if data:
            duration = data.get("duration")
            if duration:
                return int(duration)
        return None

    async def _fetch_social_metadata(self, url: str) -> str | None:
        """
        Use yt-dlp to extract post metadata and comments without touching the phone.
        Returns a formatted string suitable for the research loop, or None if unavailable.
        """
        data = await self._fetch_yt_dlp_metadata(url)
        if not data:
            return None

        # Instagram does not expose comment_count (or like_count) via yt-dlp.
        # Fall back to a single ADB screen-capture read to get the real numbers.
        if "instagram.com" in url and not data.get("comment_count"):
            logger.info("_fetch_social_metadata: Instagram comment_count missing — trying screen capture fallback")
            try:
                screen_counts = await self.social_agent.read_engagement_counts(url)
                if screen_counts.get("comments") is not None:
                    data["comment_count"] = screen_counts["comments"]
                if screen_counts.get("likes") is not None and not data.get("like_count"):
                    data["like_count"] = screen_counts["likes"]
            except Exception as _sc_err:
                logger.warning("Screen engagement fallback failed: %s", _sc_err)

        lines = []
        if data.get("uploader") or data.get("channel"):
            lines.append(f"Author: {data.get('uploader') or data.get('channel')}")
        if data.get("description"):
            lines.append(f"Caption: {data['description']}")
        if data.get("like_count") is not None:
            lines.append(f"Likes: {data['like_count']}")
        if data.get("comment_count") is not None:
            lines.append(f"Total comments: {data['comment_count']}")
        if data.get("upload_date"):
            lines.append(f"Posted: {data['upload_date']}")

        comments = data.get("comments") or []
        if comments:
            lines.append(f"\nTop {len(comments)} comments:")
            for c in comments:
                author = c.get("author", "unknown")
                text = c.get("text", "")
                lines.append(f"  @{author}: {text}")
        else:
            lines.append("\nNo comments available via metadata.")

        return "\n".join(lines)

    async def _analyse_comment_sentiment(self, comments: list[dict]) -> str | None:
        """
        Run LLM-based sentiment analysis on video/post comments.
        Returns a formatted COMMS SENTIMENT block, or None on failure.
        """
        if not comments:
            return None

        comment_text = "\n".join(f"@{c.get('author', 'unknown')}: {c.get('text', '')}" for c in comments[:50])

        prompt = (
            "You are an intelligence analyst. Analyse the following social media comments for "
            "sentiment and key themes. Respond with a concise structured block in this exact format:\n\n"
            "## COMMS SENTIMENT\n"
            "Overall: [POSITIVE/NEGATIVE/MIXED/NEUTRAL] — one-line characterisation\n"
            "Dominant themes: [comma-separated key themes, max 5]\n"
            "Notable signals:\n"
            '  - "[quote]" [@author] [LABEL: HOSTILE/SUPPORTIVE/SUSPICIOUS/CONSPIRATORIAL/SCEPTICAL/OTHER]\n'
            "  (2-4 entries)\n"
            "Intelligence value: [HIGH/MODERATE/LOW] — reason\n\n"
            f"Comments:\n{comment_text}"
        )

        try:
            response = self.client.models.generate_content(model=self.model_id, contents=prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning("Comment sentiment analysis failed: %s", e)
            return None

    async def process_media_link(self, url: str) -> tuple[str, dict]:
        """
        Uses ADB to capture media from a URL, then analyzes it with Gemini.

        Returns:
            (analysis_text, engagement_counts) where engagement_counts is a dict
            with keys ``likes``, ``comments``, ``shares`` (int or None).
            engagement_counts is populated via a single screenshot taken while the
            reel is loaded, before recording begins — no extra URL open needed.
        """
        logger.info("Triggering ADB capture for URL: %s", url)

        _FALLBACK_DURATION = 60
        _MAX_DURATION = 180
        _BUFFER_SECS = 3

        detected = await self._detect_video_duration(url)
        if detected and 3 <= detected <= _MAX_DURATION:
            record_duration = min(detected + _BUFFER_SECS, _MAX_DURATION)
            logger.info("Detected video duration: %ds — recording for %ds", detected, record_duration)
        else:
            record_duration = _FALLBACK_DURATION
            logger.info("Could not detect duration (got %r) — using fallback %ds", detected, record_duration)

        # Acquire ADB lock BEFORE touching the phone so the persona daemon yields
        lock_ttl = record_duration + 120
        await self.redis.set("osia:adb:lock", "orchestrator", ex=lock_ttl)
        engagement_counts: dict = {}
        try:
            await self.adb.open_url(url)
            await asyncio.sleep(3)

            # Reel is now loaded — grab engagement counts from the live screen
            # before recording starts (no second URL open required).
            try:
                engagement_counts = await self.social_agent._read_engagement_from_current_screen()
            except Exception as _ec_err:
                logger.warning("Engagement count screenshot failed: %s", _ec_err)

            remote_path = "/sdcard/osia_capture.mp4"
            local_path = str(self.base_dir / "osia_capture.mp4")

            async def _press_back_after(delay: float):
                await asyncio.sleep(delay)
                logger.info("Pressing back to stop video playback.")
                await self.adb.press_back()

            await asyncio.gather(
                self.adb.record_screen(remote_path=remote_path, time_limit=record_duration),
                _press_back_after(max(record_duration - 1, 1)),
            )
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
        response = self.client.models.generate_content(model=self.model_id, contents=[video_file, prompt])

        capture_path = Path(local_path)
        if capture_path.exists():
            capture_path.unlink()
        return response.text, engagement_counts

    # ------------------------------------------------------------------
    # Desk routing via Venice (uncensored — no query is refused or misrouted)
    # ------------------------------------------------------------------

    async def _route_to_desk(self, prompt: str) -> str:
        """Call Venice uncensored to select a desk slug. Falls back to Gemini on error."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    f"{self._venice_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._venice_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._routing_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 64,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("Venice routing failed (%s) — falling back to Gemini", e)
            result = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_id,
                contents=prompt,
            )
            return (result.text or "").strip()

    # ------------------------------------------------------------------
    # Qdrant context injection
    # ------------------------------------------------------------------

    async def _build_context_block(
        self,
        assigned_desk: str,
        query: str,
        entity_names: list[str],
    ) -> str | None:
        """
        Fetch desk-specific, cross-desk, and boost-collection Qdrant results.
        Applies 70-day half-life temporal decay so recent intel ranks higher.
        Deduplicates by text fingerprint and formats an ## INTELLIGENCE CONTEXT block.
        Returns None if no results.
        """
        desk_cfg = None
        try:
            desk_cfg = self.desk_registry.get(assigned_desk)
            collection = desk_cfg.qdrant.collection
            top_k = desk_cfg.qdrant.context_top_k
        except KeyError:
            collection = "collection-directorate"
            top_k = 5

        # Primary desk collection search with temporal decay
        results = []
        try:
            results = await self.qdrant.search(collection, query, top_k, decay_half_life_days=70.0)
        except Exception as e:
            logger.warning("Qdrant desk search unavailable (%s) — proceeding without context", e)

        # Cross-desk search — uses entity names when available, falls back to raw query.
        # Always runs so collections like epstein-files surface even without entity extraction.
        cross_search_query = " ".join(entity_names) if entity_names else query
        cross_results = []
        try:
            cross_results = await self.qdrant.cross_desk_search(cross_search_query, top_k=3, decay_half_life_days=70.0)
        except Exception as e:
            logger.warning("Qdrant cross-desk search unavailable (%s)", e)

        # Boost collections — guaranteed hits from desk-specific KB collections
        # regardless of their global ranking vs. other collections.
        boost_results = []
        if desk_cfg and desk_cfg.qdrant.boost_collections:
            boost_top_k = desk_cfg.qdrant.boost_top_k

            async def _search_boost(col: str) -> list:
                try:
                    return await self.qdrant.search(col, cross_search_query, boost_top_k, decay_half_life_days=70.0)
                except Exception as e:
                    logger.warning("Boost collection '%s' unavailable: %s", col, e)
                    return []

            boost_batches = await asyncio.gather(*[_search_boost(col) for col in desk_cfg.qdrant.boost_collections])
            for batch in boost_batches:
                boost_results.extend(batch)

        # Deduplicate by (collection, text fingerprint), keeping highest-scoring copy
        seen: dict[tuple[str, str], float] = {}
        combined = []
        for r in results + cross_results + boost_results:
            key = (r.collection, r.text[:120])
            if key not in seen or r.score > seen[key]:
                seen[key] = r.score
                combined.append(r)

        if not combined:
            return None

        # Sort combined results by score descending for presentation
        combined.sort(key=lambda r: r.score, reverse=True)

        lines = ["## INTELLIGENCE CONTEXT\n"]
        for r in combined:
            reliability = r.metadata.get("reliability_tier", "?")
            timestamp = r.metadata.get("timestamp", r.metadata.get("collected_at", ""))
            source_label = r.metadata.get("source", r.collection)
            lines.append(
                f"[{r.collection} | {source_label}] (Reliability: {reliability}, Score: {r.score:.2f}) {timestamp[:10] if timestamp else ''}\n{r.text}\n"
            )
        return "\n".join(lines)

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
        url: str | None = None

        logger.info("Received new task from %s: %s", source, query)

        # 1a. Signal attachment analysis (images / videos sent directly to the chat)
        signal_attachments = task.get("attachments") or []
        if signal_attachments:
            attach_parts: list[str] = []
            for att in signal_attachments:
                att_path = att.get("path", "")
                att_type = att.get("content_type", "")
                if not att_path or not Path(att_path).exists():
                    continue
                try:
                    logger.info("Uploading Signal attachment to Gemini: %s (%s)", att_path, att_type)
                    att_file = self.client.files.upload(file=att_path)
                    while att_file.state.name == "PROCESSING":
                        await asyncio.sleep(2)
                        att_file = self.client.files.get(name=att_file.name)

                    if att_type.startswith("image/"):
                        att_prompt = (
                            "Analyse this image sent as a Signal intelligence attachment. "
                            "Describe all visible content in detail: text, faces, locations, objects, "
                            "maps, documents, or any other identifiable elements. "
                            "Summarise the intelligence value."
                        )
                    else:
                        att_prompt = (
                            "Analyse this video sent as a Signal intelligence attachment. "
                            "Transcribe any spoken audio, describe the visual content, identify text on screen, "
                            "and summarise the core message or intelligence value."
                        )

                    att_response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=[att_file, att_prompt],
                    )
                    attach_parts.append(f"[{att_type}]\n{att_response.text.strip()}")
                except Exception as _att_err:
                    logger.error("Signal attachment analysis failed (%s): %s", att_path, _att_err)
                finally:
                    # Clean up temp file
                    try:
                        Path(att_path).unlink(missing_ok=True)
                    except Exception as cleanup_err:
                        logger.warning(
                            "Failed to delete temporary Signal attachment file (%s): %s",
                            att_path,
                            cleanup_err,
                        )

            if attach_parts:
                attach_block = "\n\n".join(attach_parts)
                if media_analysis:
                    media_analysis += f"\n\n## SIGNAL ATTACHMENTS\n{attach_block}"
                else:
                    media_analysis = f"## SIGNAL ATTACHMENTS\n{attach_block}"
                # Inject attachment analysis into the query so the desk and router
                # see the actual content — mirrors how the YouTube/ADB paths work.
                if original_query:
                    query = f"{original_query}\n\n{media_analysis}"
                else:
                    query = f"A Signal media attachment was intercepted and analysed:\n\n{media_analysis}"

        # 1b. Media Link Interception
        url_match = re.search(r"(https?://[^\s]+)", query)
        if url_match:
            url = url_match.group(1)
            url_lower = url.lower()

            if any(domain in url_lower for domain in YOUTUBE_DOMAINS):
                try:
                    transcript, yt_meta = await asyncio.gather(
                        self._extract_youtube_transcript(url),
                        self._fetch_yt_dlp_metadata(url),
                        return_exceptions=True,
                    )
                    if isinstance(transcript, BaseException):
                        logger.error("YouTube transcript extraction failed: %s", transcript)
                        transcript = None
                    if isinstance(yt_meta, BaseException):
                        logger.warning("YouTube metadata fetch failed: %s", yt_meta)
                        yt_meta = None

                    context_parts: list[str] = []
                    if transcript:
                        context_parts.append(f"## TRANSCRIPT\n{transcript}")

                    if yt_meta:
                        if yt_meta.get("description"):
                            context_parts.append(f"## VIDEO DESCRIPTION\n{yt_meta['description']}")
                        comments = yt_meta.get("comments") or []
                        if comments:
                            sentiment = await self._analyse_comment_sentiment(comments)
                            if sentiment:
                                context_parts.append(sentiment)

                    if context_parts:
                        media_analysis = "\n\n".join(context_parts)
                        query = f"A YouTube video from {url} was just ingested. Context:\n{media_analysis}\n\nOriginal Request: {query}"
                except Exception as e:
                    logger.error("YouTube ingress failed: %s", e)

            elif any(domain in url_lower for domain in MEDIA_DOMAINS):
                try:
                    media_result, raw_meta = await asyncio.gather(
                        self.process_media_link(url),
                        self._fetch_yt_dlp_metadata(url),
                        return_exceptions=True,
                    )
                    if isinstance(media_result, BaseException):
                        logger.error("Media interception failed: %s", media_result)
                        adb_analysis = None
                        screen_counts: dict = {}
                    else:
                        adb_analysis, screen_counts = media_result
                    if isinstance(raw_meta, BaseException):
                        logger.warning("Social metadata fetch failed: %s", raw_meta)
                        raw_meta = None

                    # Merge screen-captured engagement counts into yt-dlp metadata.
                    # Instagram never exposes comment_count via yt-dlp; the screenshot
                    # taken while the reel was loaded gives us the real numbers.
                    if raw_meta and screen_counts:
                        if screen_counts.get("comments") is not None and not raw_meta.get("comment_count"):
                            raw_meta["comment_count"] = screen_counts["comments"]
                        if screen_counts.get("likes") is not None and not raw_meta.get("like_count"):
                            raw_meta["like_count"] = screen_counts["likes"]
                        if screen_counts.get("shares") is not None:
                            raw_meta["shares_count"] = screen_counts["shares"]

                    context_parts = []
                    if adb_analysis:
                        context_parts.append(f"## VISUAL INTERCEPT\n{adb_analysis}")

                    if raw_meta:
                        # Format description/caption block from raw metadata
                        meta_lines = []
                        if raw_meta.get("uploader") or raw_meta.get("channel"):
                            meta_lines.append(f"Author: {raw_meta.get('uploader') or raw_meta.get('channel')}")
                        if raw_meta.get("description"):
                            meta_lines.append(f"Caption: {raw_meta['description']}")
                        if raw_meta.get("like_count") is not None:
                            meta_lines.append(f"Likes: {raw_meta['like_count']}")
                        if raw_meta.get("comment_count") is not None:
                            meta_lines.append(f"Total comments: {raw_meta['comment_count']}")
                        if raw_meta.get("shares_count") is not None:
                            meta_lines.append(f"Shares: {raw_meta['shares_count']}")
                        if raw_meta.get("upload_date"):
                            meta_lines.append(f"Posted: {raw_meta['upload_date']}")
                        if meta_lines:
                            context_parts.append("## POST METADATA\n" + "\n".join(meta_lines))

                        comments = raw_meta.get("comments") or []
                        if comments:
                            sentiment = await self._analyse_comment_sentiment(comments)
                            if sentiment:
                                context_parts.append(sentiment)

                    if context_parts:
                        media_analysis = "\n\n".join(context_parts)
                        query = f"A media intercept from {url} was just ingested. Context:\n{media_analysis}\n\nOriginal Request: {query}"
                except Exception as e:
                    logger.error("Media interception failed: %s", e)

        # 2. Automated Research (multi-turn tool calling)
        try:
            research_summary, source_tracker = await self.handle_research(original_query or media_analysis or query)
            if research_summary:
                logger.info("Research complete.")
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

        # 3. Entity extraction + research job enqueuing
        entities = []
        entity_names: list[str] = []
        text_for_extraction = research_summary or media_analysis or original_query
        triggered_by = url or (source[len("signal:") :] if source.startswith("signal:") else source)
        try:
            entities = await self.entity_extractor.extract(text_for_extraction, "collection-directorate")
            entity_names = [e.name for e in entities]
            await self.entity_extractor.enqueue_research_jobs(entities, triggered_by=triggered_by)
        except Exception as e:
            logger.error("Entity extraction/enqueuing failed: %s", e)

        # 4. Chief of Staff routes to the appropriate desk
        directives_path = self.base_dir / "DIRECTIVES.md"
        mandate = directives_path.read_text()

        valid_desks_list = "\n".join(f"- {s}" for s in sorted(self.valid_desks))
        plan_prompt = f"""
        {mandate}

        ---

        You are the Chief of Staff for OSIA. A new Request for Information (RFI) has come in: '{query if original_query else (media_analysis or query)}'

        Which of our specialized desks should analyze this? Choose ONE from the following list:
{valid_desks_list}

        Reply with ONLY the slug of the desk, nothing else.
        """

        try:
            pinned_desk = task.get("desk")
            if pinned_desk and pinned_desk in self.valid_desks:
                assigned_desk = pinned_desk
                logger.info("Task desk pre-assigned to '%s' (bypassing AI routing)", assigned_desk)
            else:
                if pinned_desk:
                    logger.warning("Pre-assigned desk '%s' is not valid — falling back to AI routing", pinned_desk)
                assigned_desk = await self._route_to_desk(plan_prompt)

                if assigned_desk not in self.valid_desks:
                    logger.warning(
                        "Gemini returned invalid desk '%s', routing to default '%s'",
                        assigned_desk,
                        self.default_desk,
                    )
                    assigned_desk = self.default_desk

            logger.info("Task routed to: %s", assigned_desk)

            # 5. Qdrant context injection
            context_block = await self._build_context_block(assigned_desk, original_query, entity_names)

            # 6. Invoke desk via DeskRegistry
            try:
                response_data = await self.desk_registry.invoke(assigned_desk, query, context_block)
                # invoke returns (text, metadata) tuple per design
                if isinstance(response_data, tuple):
                    analysis, invoke_meta = response_data
                else:
                    analysis = response_data
                    invoke_meta = {}

                model_used = invoke_meta.get("model_used", "primary")
                model_id_used = invoke_meta.get("model_id", "unknown")
                logger.info(
                    "Desk '%s' responded via %s model (%s).",
                    assigned_desk,
                    model_used,
                    model_id_used,
                )
            except Exception as desk_err:
                logger.error("Desk '%s' invocation failed: %s", assigned_desk, desk_err)
                raise

            logger.info("Intelligence synthesis complete.")

            # Audit citations and append source quality summary
            analysis += audit_report(analysis, source_tracker)

            # 7. Store desk analysis in Qdrant for future context retrieval
            try:
                desk_cfg = self.desk_registry.get(assigned_desk)
                desk_collection = desk_cfg.qdrant.collection
                await self.qdrant.upsert(
                    desk_collection,
                    analysis,
                    metadata={
                        "desk": assigned_desk,
                        "topic": (original_query or "")[:200],
                        "source": source,
                        "reliability_tier": "A",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "entity_tags": entity_names,
                        "triggered_by": triggered_by,
                        "model_used": model_id_used,
                    },
                )
                logger.debug("Stored analysis in Qdrant collection '%s'", desk_collection)
            except Exception as e:
                logger.warning("Failed to store analysis in Qdrant for desk '%s': %s", assigned_desk, e)

            if source.startswith("signal:"):
                recipient = source[len("signal:") :]
                await self.send_signal_message(recipient, analysis)
                try:
                    infographic_b64 = await self.generate_infographic(analysis)
                    if infographic_b64:
                        await self.send_signal_image(recipient, infographic_b64, caption="📊 OSIA Intelligence Brief")
                except Exception as img_err:
                    logger.warning("Infographic delivery failed (non-fatal): %s", img_err)

            # Archive a PDF copy of every completed analysis
            if os.getenv("OSIA_DEBUG_PDF", "false").lower() == "true":
                logger.info("Attempting PDF archival for desk %s, source %s", assigned_desk, source)
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: generate_intsum_pdf(analysis, assigned_desk, source),
                )
            except Exception as pdf_err:
                logger.warning("PDF archival failed (non-fatal): %s", pdf_err)

        except Exception as e:
            logger.exception("Orchestration or desk analysis failed: %s", e)
            if source.startswith("signal:"):
                recipient = source[len("signal:") :]

                error_type = type(e).__name__
                raw_error = str(e)

                sanitized_error = re.sub(r"(/home/|/var/|/tmp/|C:\\)[^\s]+", "[REDACTED_PATH]", raw_error)
                sanitized_error = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED_KEY]", sanitized_error)
                sanitized_error = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_KEY]", sanitized_error)
                sanitized_error = re.sub(r"(https?://)[^\s]+", r"\1[REDACTED_URL]", sanitized_error)

                error_msg = (
                    f"⚠️ OSIA System Error ({error_type})\n\n"
                    f"An error occurred while processing your request:\n"
                    f'"{sanitized_error}"\n\n'
                    "This could be due to a temporary issue with our intelligence desks or a timeout. "
                    "Please try your query again later."
                )
                await self.send_signal_message(recipient, error_msg)
