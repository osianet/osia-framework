import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

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
from src.gateways.kali_dispatcher import KaliDispatcher
from src.gateways.mcp_dispatcher import MCPDispatcher
from src.intelligence.entity_extractor import EntityExtractor
from src.intelligence.infographic_renderer import generate_infographic
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


def _format_kali_result(result: dict) -> str:
    """Format a Kali API response envelope into a readable string for the research loop."""
    tool = result.get("tool", "unknown")
    target = result.get("target") or result.get("url", "")
    duration = result.get("duration_seconds", 0)
    timed_out = result.get("timed_out", False)
    rc = result.get("return_code", 0)
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()

    header = f"[kali:{tool}] target={target} rc={rc} duration={duration:.1f}s"
    if timed_out:
        header += " TIMED_OUT"
    parts = [header]
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    return "\n".join(parts)


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
    "information-warfare-desk",
    "environment-and-ecology-desk",
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
        # Timeout of 300s: video generate_content calls can take 60-120s for longer clips;
        # the default httpx timeout (~60s) causes "Server disconnected" on busy Gemini nodes.
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=300),
        )
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
        self.kali = KaliDispatcher()
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
                    # ----------------------------------------------------------
                    # Kali Linux recon & assessment tools (local API, loopback)
                    # ----------------------------------------------------------
                    types.FunctionDeclaration(
                        name="kali_nmap",
                        description="Run an nmap port scan / service fingerprint against a host or CIDR range. Use for infrastructure mapping, open port discovery, service/version detection, and OS fingerprinting.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(
                                    type="STRING", description="IP address, hostname, or CIDR range to scan."
                                ),
                                "ports": types.Schema(
                                    type="STRING", description="Port spec, e.g. '22', '22,80,443', '1-1024'."
                                ),
                                "top_ports": types.Schema(type="INTEGER", description="Scan the N most common ports."),
                                "scan_type": types.Schema(
                                    type="STRING", description="Scan type: syn, connect, udp, ack, fin, null, xmas."
                                ),
                                "version_detection": types.Schema(
                                    type="BOOLEAN", description="Enable service/version detection (-sV)."
                                ),
                                "os_detection": types.Schema(
                                    type="BOOLEAN", description="Enable OS fingerprinting (-O)."
                                ),
                                "aggressive": types.Schema(
                                    type="BOOLEAN",
                                    description="Enable aggressive mode (-A): version + scripts + OS + traceroute.",
                                ),
                                "scripts": types.Schema(
                                    type="ARRAY",
                                    items=types.Schema(type="STRING"),
                                    description="NSE scripts to run, e.g. ['ssl-cert', 'http-title'].",
                                ),
                                "skip_host_discovery": types.Schema(
                                    type="BOOLEAN", description="Treat host as online (-Pn), skip ping probe."
                                ),
                                "timing": types.Schema(
                                    type="INTEGER", description="Timing template 0-5 (0=paranoid, 3=normal, 5=insane)."
                                ),
                                "open_only": types.Schema(type="BOOLEAN", description="Show only open ports."),
                                "no_dns": types.Schema(type="BOOLEAN", description="Skip reverse DNS resolution (-n)."),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_whois",
                        description="WHOIS lookup for a domain name or IP address. Returns registrar, registration dates, nameservers, and registrant details.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(
                                    type="STRING", description="Domain name or IP address to look up."
                                ),
                                "server": types.Schema(
                                    type="STRING",
                                    description="Query this WHOIS server directly (e.g. 'whois.arin.net').",
                                ),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_dig",
                        description="DNS lookup for any record type. Use for resolving A/AAAA/MX/NS/TXT/CNAME/SOA records, PTR (reverse DNS), DNSSEC validation, or tracing delegation chains.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(
                                    type="STRING", description="Domain name or IP address (for PTR lookups)."
                                ),
                                "record_type": types.Schema(
                                    type="STRING",
                                    description="Record type: A, AAAA, MX, NS, TXT, CNAME, SOA, PTR, SRV, CAA, ANY.",
                                ),
                                "nameserver": types.Schema(
                                    type="STRING", description="Query this resolver directly (e.g. '8.8.8.8')."
                                ),
                                "short": types.Schema(
                                    type="BOOLEAN", description="Return answer values only (+short)."
                                ),
                                "trace": types.Schema(
                                    type="BOOLEAN", description="Trace delegation chain from root (+trace)."
                                ),
                                "dnssec": types.Schema(type="BOOLEAN", description="Request DNSSEC records."),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_sslscan",
                        description="Scan TLS/SSL configuration of a host: supported protocol versions, cipher suites, certificate details, OCSP stapling. Use to assess TLS posture or extract certificate metadata.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(type="STRING", description="Hostname or hostname:port."),
                                "port": types.Schema(type="INTEGER", description="Target port (default 443)."),
                                "show_certificate": types.Schema(
                                    type="BOOLEAN", description="Include full certificate details."
                                ),
                                "starttls": types.Schema(
                                    type="STRING",
                                    description="STARTTLS protocol: smtp, ftp, imap, pop3, xmpp, ldap, rdp.",
                                ),
                                "ocsp": types.Schema(type="BOOLEAN", description="Check OCSP stapling."),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_whatweb",
                        description="Fingerprint web application technologies: CMS, frameworks, server software, JavaScript libraries, analytics, and WAF detection.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(type="STRING", description="URL or hostname to fingerprint."),
                                "aggression": types.Schema(
                                    type="INTEGER",
                                    description="Aggression level: 1 (stealthy, single request) or 3 (aggressive, multiple requests).",
                                ),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_harvester",
                        description="Passive OSINT subdomain and infrastructure enumeration. Queries certificate transparency logs, DNS aggregators, and threat intel sources to map a target domain's attack surface.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(
                                    type="STRING", description="Root domain to enumerate (e.g. 'example.com')."
                                ),
                                "sources": types.Schema(
                                    type="ARRAY",
                                    items=types.Schema(type="STRING"),
                                    description="Data sources: crtsh, dnsdumpster, hackertarget, otx, urlscan, anubis, certspotter, etc.",
                                ),
                                "limit": types.Schema(type="INTEGER", description="Max results per source (10-500)."),
                                "dns_resolve": types.Schema(
                                    type="BOOLEAN", description="Resolve discovered hostnames to IPs."
                                ),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_amass",
                        description="Active domain enumeration via Amass — zone transfers, certificate grabs, and public data sources. More thorough than harvester for large targets.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(type="STRING", description="Domain to enumerate."),
                                "mode": types.Schema(
                                    type="STRING",
                                    description="'passive' (public data only) or 'active' (zone transfers, cert grabs).",
                                ),
                                "timeout": types.Schema(
                                    type="INTEGER", description="Minutes before stopping (1-30, default 5)."
                                ),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_curl_probe",
                        description="HTTP/HTTPS probe: fetch headers, response bodies, test endpoints. Use to investigate web services, check server responses, or retrieve RDAP/API data.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "url": types.Schema(
                                    type="STRING", description="URL to probe (must start with http:// or https://)."
                                ),
                                "method": types.Schema(
                                    type="STRING", description="HTTP method: GET, HEAD, POST, OPTIONS."
                                ),
                                "headers": types.Schema(
                                    type="OBJECT", description="Extra request headers as key-value pairs."
                                ),
                                "body": types.Schema(type="STRING", description="Request body for POST."),
                                "follow_redirects": types.Schema(
                                    type="BOOLEAN", description="Follow Location redirects."
                                ),
                                "insecure": types.Schema(
                                    type="BOOLEAN", description="Skip TLS certificate verification."
                                ),
                                "max_time": types.Schema(
                                    type="INTEGER", description="Total timeout in seconds (1-60)."
                                ),
                            },
                            required=["url"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_ping",
                        description="ICMP echo probe to check host reachability and measure round-trip latency.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(type="STRING", description="IP address or hostname to ping."),
                                "count": types.Schema(type="INTEGER", description="Number of packets to send (1-20)."),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_traceroute",
                        description="Hop-by-hop network path trace to a destination. Reveals routing infrastructure, ISP hops, and geographic path.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(type="STRING", description="IP address or hostname to trace."),
                                "max_hops": types.Schema(
                                    type="INTEGER", description="Maximum hop count (1-64, default 30)."
                                ),
                                "protocol": types.Schema(
                                    type="STRING", description="Probe protocol: icmp, udp, or tcp."
                                ),
                                "port": types.Schema(
                                    type="INTEGER", description="Destination port for TCP/UDP probes."
                                ),
                            },
                            required=["target"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="kali_nikto",
                        description="Web server vulnerability scanner: misconfigured headers, default files, known CVEs, injection points, and outdated software. Use for web server security assessment.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "target": types.Schema(type="STRING", description="URL or hostname to scan."),
                                "port": types.Schema(type="INTEGER", description="Override target port."),
                                "ssl": types.Schema(type="BOOLEAN", description="Force SSL/TLS."),
                                "tuning": types.Schema(
                                    type="ARRAY",
                                    items=types.Schema(type="STRING"),
                                    description="Test categories: 0=upload, 1=interesting-files, 2=misconfig, 3=info-disclosure, 4=injection, 9=sqli, a=auth-bypass, b=software-id.",
                                ),
                                "max_time": types.Schema(type="INTEGER", description="Timeout in seconds (10-600)."),
                            },
                            required=["target"],
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

        # Recover any tasks that were in-flight when the service last crashed.
        # They were moved to osia:task_processing but never completed, so we
        # move them back to the front of osia:task_queue for reprocessing.
        recovered = 0
        while True:
            item = await self.redis.lmove("osia:task_processing", self.queue_name, "LEFT", "LEFT")
            if item is None:
                break
            recovered += 1
        if recovered:
            logger.warning(
                "Recovered %d in-progress task(s) from previous crash — re-queued for processing.",
                recovered,
            )

        # Post a brief startup status message to the Signal group
        signal_group = os.getenv("SIGNAL_GROUP_ID")
        if signal_group:
            try:
                queued = await self.redis.llen(self.queue_name)
                research_queued = await self.redis.llen("osia:research_queue")
                lines = [
                    "⬛ OSIA ONLINE ⬛",
                    f"🕐 {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
                    f"🗂 Desks active: {len(self.valid_desks)}",
                    f"📥 Task queue: {queued} pending",
                    f"🔬 Research queue: {research_queued} pending",
                ]
                if recovered:
                    lines.append(f"♻️ {recovered} task(s) recovered from crash")
                await self.send_signal_message(signal_group, "\n".join(lines))
            except Exception as e:
                logger.warning("Failed to send startup status to Signal: %s", e)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def shutdown(self):
        """Gracefully close shared clients."""
        await self._signal_client.aclose()
        await self.desk_registry.close()
        await self.mcp.close_all()
        await self.kali.close()

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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_forever(self):
        await self.bootstrap()
        logger.info("OSIA Orchestrator online. Listening to Redis queue: %s", self.queue_name)
        while True:
            try:
                # Atomically move the task from the input queue into the
                # processing list before touching it.  If the service crashes
                # mid-task, bootstrap() will move it back to the input queue
                # on the next start so it isn't silently dropped.
                task_json = await self.redis.blmove(
                    self.queue_name, "osia:task_processing", 0, src="LEFT", dest="RIGHT"
                )
                if task_json:
                    task = json.loads(task_json)
                    try:
                        await self.process_task(task)
                    finally:
                        # Remove from processing list regardless of outcome.
                        # A task that raises an unhandled exception has already
                        # been logged; re-queuing it would just loop forever.
                        await self.redis.lrem("osia:task_processing", 1, task_json)
            except redis.ConnectionError as e:
                logger.error("Redis connection lost: %s — retrying in 5s", e)
                await asyncio.sleep(5)
            except Exception as e:
                logger.exception("Unexpected error processing task: %s", e)

    # ------------------------------------------------------------------
    # Signal command handling
    # ------------------------------------------------------------------

    # Short aliases → full desk slugs, for convenience in /desk and /investigate
    _DESK_ALIASES: dict[str, str] = {
        "geo": "geopolitical-and-security-desk",
        "geopolitical": "geopolitical-and-security-desk",
        "humint": "human-intelligence-and-profiling-desk",
        "cultural": "cultural-and-theological-intelligence-desk",
        "cyber": "cyber-intelligence-and-warfare-desk",
        "finance": "finance-and-economics-directorate",
        "sci": "science-technology-and-commercial-desk",
        "science": "science-technology-and-commercial-desk",
        "infowar": "information-warfare-desk",
        "env": "environment-and-ecology-desk",
        "environment": "environment-and-ecology-desk",
        "watch": "the-watch-floor",
        "watchfloor": "the-watch-floor",
    }

    def _resolve_desk_slug(self, raw: str) -> str | None:
        """Resolve a full slug or alias to a valid desk slug, or return None."""
        if raw in self.valid_desks:
            return raw
        return self._DESK_ALIASES.get(raw.lower())

    async def _handle_signal_command(self, task: dict, source: str) -> bool:
        """
        Intercept slash commands from any source.

        Returns True  → command was fully handled; caller should return.
        Returns False → not a handled standalone command; caller continues
                        normal processing (task dict may have been mutated
                        in-place, e.g. by /desk).
        """
        query = task.get("query", "").strip()
        recipient = source[len("signal:") :] if source.startswith("signal:") else source

        parts = query.split(None, 2)
        command = parts[0].lower()
        arg1 = parts[1] if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""

        if command == "/help":
            await self._cmd_help(recipient)
            return True

        if command == "/investigate":
            topic = (arg1 + (" " + rest if rest else "")).strip()
            if not topic:
                await self.send_signal_message(recipient, "Usage: /investigate <topic>")
                return True
            await self._cmd_investigate(topic, recipient)
            return True

        if command == "/desk":
            # /desk <slug|alias> <query…>
            if not arg1 or not rest:
                await self.send_signal_message(
                    recipient,
                    "Usage: /desk <slug> <query>\nSend /desks for available slugs.",
                )
                return True
            slug = self._resolve_desk_slug(arg1)
            if slug is None:
                await self.send_signal_message(
                    recipient,
                    f"Unknown desk: {arg1}\nSend /desks for available slugs and aliases.",
                )
                return True
            # Mutate the task dict so normal process_task flow uses the pinned desk
            task["desk"] = slug
            task["query"] = rest
            return False  # continue normal processing with the pinned desk

        if command == "/search":
            query_text = (arg1 + (" " + rest if rest else "")).strip()
            if not query_text:
                await self.send_signal_message(recipient, "Usage: /search <query>")
                return True
            await self._cmd_search(query_text, recipient)
            return True

        if command == "/status":
            await self._cmd_status(recipient, arg1)
            return True

        if command == "/desks":
            await self._cmd_desks(recipient)
            return True

        # Unknown command
        await self.send_signal_message(
            recipient,
            f"Unknown command: {command}\nSend /help for available commands.",
        )
        return True

    async def _route_to_desks(self, topic: str, count: int = 3) -> list[str]:
        """
        Ask the Chief of Staff to select the `count` most relevant desks for a
        multi-desk investigation. Returns validated slug list; falls back to
        a sensible default set on failure.
        """
        valid_desks_list = "\n".join(f"- {s}" for s in sorted(self.valid_desks))
        prompt = (
            f"You are the Chief of Staff for OSIA. An investigation directive has arrived:\n\n"
            f'"{topic}"\n\n'
            f"Select the {count} most relevant intelligence desks to investigate this topic. "
            f"Choose from:\n{valid_desks_list}\n\n"
            f"Reply with ONLY the {count} desk slugs, one per line, no punctuation, nothing else."
        )
        try:
            raw = await self._route_to_desk(prompt)
            slugs = [line.strip().lstrip("- •").strip() for line in raw.splitlines() if line.strip()]
            valid = [s for s in slugs if s in self.valid_desks][:count]
            if valid:
                return valid
        except Exception as e:
            logger.warning("Multi-desk routing failed: %s — using defaults", e)

        return [
            "geopolitical-and-security-desk",
            "cultural-and-theological-intelligence-desk",
            "information-warfare-desk",
        ]

    async def _cmd_help(self, recipient: str) -> None:
        help_text = (
            "◈ OSIA OPERATIVE GUIDE ◈\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "── NATURAL LANGUAGE ─────────────────\n"
            "Just send a message. The Chief of Staff reads it, selects the\n"
            "most relevant desk, assembles context from the knowledge base,\n"
            "and returns a full intelligence assessment. No command needed.\n\n"
            "  Examples:\n"
            "  · What is the current threat posture in the South China Sea?\n"
            "  · Profile Klaus Schwab — background, influence networks, funding.\n"
            "  · Explain the strategic implications of the CHIPS Act.\n\n"
            "── VIDEO & SOCIAL MEDIA LINKS ───────\n"
            "Paste a URL and OSIA processes it automatically:\n\n"
            "  YouTube      — transcript extracted and analysed by the desk.\n"
            "  TikTok       — video screen-recorded via ADB and analysed by Gemini.\n"
            "  Instagram    — post and comments captured via ADB.\n"
            "  Facebook     — post content captured via ADB.\n\n"
            "  Include context after the URL to guide the analysis:\n"
            "  · https://youtu.be/xyz  — assess for disinfo markers\n\n"
            "── RESEARCH QUERIES ─────────────────\n"
            "Anything phrased as an open question or investigation directive\n"
            "triggers a live multi-tool research loop: web search (Tavily),\n"
            "academic papers (Semantic Scholar / ArXiv), Wikipedia, and the\n"
            "full OSIA knowledge base. Results are embedded into the desk KB\n"
            "for future retrieval.\n\n"
            "── SLASH COMMANDS ───────────────────\n"
            "/investigate <topic>\n"
            "  Fan-out deep research across the 3 most relevant desks in\n"
            "  parallel. Each desk produces a full INTSUM. Use for complex\n"
            "  multi-angle topics where one desk is not enough.\n\n"
            "/desk <slug> <query>\n"
            "  Bypass AI routing and send directly to a named desk.\n"
            "  Useful when you know exactly which desk should handle it.\n"
            "  Aliases: geo, humint, cyber, infowar, env, cultural, finance,\n"
            "           sci, watch.\n\n"
            "/search <query>\n"
            "  Semantic search across all desk knowledge bases. Returns the\n"
            "  top 5 stored intelligence items with relevance scores. Good\n"
            "  for retrieving prior research without triggering a new analysis.\n\n"
            "/status [section]\n"
            "  System health dashboard. Sections:\n"
            "    system   — CPU, memory, disk, temperature\n"
            "    services — systemd service states\n"
            "    docker   — container states\n"
            "    api      — HTTP + Redis health\n"
            "    qdrant   — KB collections and vector counts\n"
            "    queue    — task / research / RSS queue depths\n"
            "    worker   — research worker timer and last run stats\n\n"
            "/desks\n"
            "  List all available desk slugs and their aliases.\n\n"
            "/help\n"
            "  Show this guide.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "All analysis is backed by OSIA's persistent vector knowledge\n"
            "base (Qdrant) — 17 collections spanning geopolitics, HUMINT,\n"
            "cyber, finance, InfoWar, ecology, and declassified archives\n"
            "(Church Committee, FRUS, Pentagon Papers, CIA CREST, FBI Vault,\n"
            "WikiLeaks cables, Epstein files). Research auto-embeds into KB."
        )
        await self.send_signal_message(recipient, help_text)

    async def _cmd_investigate(self, topic: str, recipient: str) -> None:
        """Fan out an investigation across the top 3 relevant desks via research queue."""
        await self.send_signal_message(
            recipient,
            f"Routing investigation to relevant desks: {topic[:120]}{'...' if len(topic) > 120 else ''}",
        )
        desks = await self._route_to_desks(topic, count=3)
        for desk in desks:
            payload = json.dumps(
                {
                    "job_id": str(uuid.uuid4()),
                    "topic": topic,
                    "desk": desk,
                    "priority": "high",
                    "directives_lens": True,
                    "triggered_by": f"signal-investigate:{recipient}",
                }
            )
            await self.redis.rpush("osia:research_queue", payload)
            logger.info("Enqueued investigation job → %s", desk)

        desk_list = "\n".join(f"  • {d}" for d in desks)
        queue_depth = await self.redis.llen("osia:research_queue")
        await self.send_signal_message(
            recipient,
            f"Investigation queued across {len(desks)} desk(s):\n{desk_list}\n\n"
            f"Research queue depth: {queue_depth}\n"
            f"Results surface on the next worker run (every 2h).\n"
            f"Trigger now: sudo systemctl start osia-research-worker.service",
        )

    async def _cmd_search(self, query: str, recipient: str) -> None:
        """Cross-desk Qdrant search; returns top 5 results to Signal."""
        await self.send_signal_message(
            recipient,
            f"Searching intelligence database: {query[:80]}{'...' if len(query) > 80 else ''}",
        )
        try:
            results = await self.qdrant.cross_desk_search(query, top_k=5, decay_half_life_days=70.0)
        except Exception as e:
            await self.send_signal_message(recipient, f"Search failed: {e}")
            return

        if not results:
            await self.send_signal_message(recipient, "No matching intelligence found.")
            return

        lines = [f"Top {len(results)} result(s) for: {query}\n"]
        for i, r in enumerate(results, 1):
            preview = r.text[:220].replace("\n", " ").strip()
            ts = r.metadata.get("timestamp", r.metadata.get("collected_at", ""))[:10]
            lines.append(f"[{i}] {r.collection} (score: {r.score:.3f}{', ' + ts if ts else ''})\n{preview}…")
        await self.send_signal_message(recipient, "\n\n".join(lines))

    # ------------------------------------------------------------------
    # Status dashboard helpers
    # ------------------------------------------------------------------

    _STATUS_SERVICES = [
        "osia-orchestrator.service",
        "osia-signal-ingress.service",
        "osia-rss-ingress.service",
        "osia-mcp-arxiv-bridge.service",
        "osia-mcp-phone-bridge.service",
        "osia-mcp-semantic-scholar-bridge.service",
        "osia-mcp-tavily-bridge.service",
        "osia-mcp-time-bridge.service",
        "osia-mcp-wikipedia-bridge.service",
        "osia-cyber-bridge.service",
        "osia-status-api.service",
        "osia-queue-api.service",
    ]
    _STATUS_TIMERS = [
        "osia-daily-sitrep.timer",
        "osia-rss-ingress.timer",
        "osia-research-worker.timer",
    ]
    _STATUS_CONTAINERS = [
        "osia-anythingllm",
        "osia-qdrant",
        "osia-redis",
        "osia-signal",
        "mailserver",
        "osia-kali",
    ]

    async def _run_cmd(self, *args: str, timeout: float = 5.0) -> str:
        """Run a subprocess and return stripped stdout, or empty string on any failure."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode().strip()
        except Exception:
            return ""

    async def _cmd_status(self, recipient: str, subcommand: str = "") -> None:
        """Full system status dashboard, or a specific section."""
        sub = subcommand.lower().strip()

        if sub in ("system", "sys"):
            msg = await self._status_system()
        elif sub in ("services", "svc", "service"):
            msg = await self._status_services()
        elif sub in ("docker", "containers"):
            msg = await self._status_docker()
        elif sub in ("api", "health"):
            msg = await self._status_api()
        elif sub in ("qdrant", "kb", "collections"):
            msg = await self._status_qdrant()
        elif sub in ("queue", "queues"):
            msg = await self._status_queue()
        elif sub in ("worker", "research"):
            msg = await self._status_worker()
        elif sub in ("", "all", "full"):
            # Compact summary across all sections — gathered concurrently
            now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
            sections = await asyncio.gather(
                self._status_system(),
                self._status_services(),
                self._status_docker(),
                self._status_api(),
                self._status_qdrant_compact(),
                self._status_queue(),
                self._status_worker_compact(),
                return_exceptions=True,
            )
            parts = [f"OSIA STATUS — {now}"]
            for s in sections:
                if isinstance(s, Exception):
                    parts.append(f"[section error: {s}]")
                elif s:
                    parts.append(s)
            parts.append("—\n/status <section> for detail:\nsystem|services|docker|api|qdrant|queue|worker")
            msg = "\n\n".join(parts)
        else:
            msg = f"Unknown section: {sub}\nUsage: /status [system|services|docker|api|qdrant|queue|worker]"

        await self.send_signal_message(recipient, msg)

    async def _status_system(self) -> str:
        """System stats: hostname, uptime, load, CPU temp, memory, disk."""
        lines = ["SYSTEM"]

        # Hostname + uptime
        hostname = await self._run_cmd("hostname")
        uptime_raw = await self._run_cmd("uptime", "-p")
        lines.append(f"  Host: {hostname or 'unknown'}  |  {uptime_raw or 'uptime unavailable'}")

        # Load average
        try:
            load_avg = Path("/proc/loadavg").read_text().split()[:3]
            lines.append(f"  Load: {' '.join(load_avg)}")
        except Exception:
            logger.debug("Could not read /proc/loadavg", exc_info=True)

        # CPU temperature (ARM SBCs)
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            try:
                cpu_temp = int(temp_path.read_text().strip()) // 1000
                warn = " 🔴" if cpu_temp >= 85 else (" ⚠️" if cpu_temp >= 70 else "")
                lines.append(f"  CPU Temp: {cpu_temp}°C{warn}")
            except Exception:
                logger.debug("Could not read CPU temperature", exc_info=True)

        # Memory
        try:
            meminfo = Path("/proc/meminfo").read_text()
            mem_total_kb = int(next(ln for ln in meminfo.splitlines() if ln.startswith("MemTotal:")).split()[1])
            mem_avail_kb = int(next(ln for ln in meminfo.splitlines() if ln.startswith("MemAvailable:")).split()[1])
            mem_used_mb = (mem_total_kb - mem_avail_kb) // 1024
            mem_total_mb = mem_total_kb // 1024
            mem_pct = mem_used_mb * 100 // mem_total_mb if mem_total_mb else 0
            warn = " 🔴" if mem_pct >= 90 else (" ⚠️" if mem_pct >= 75 else "")
            lines.append(f"  Mem: {mem_used_mb}MB / {mem_total_mb}MB ({mem_pct}%){warn}")
        except Exception:
            logger.debug("Could not read /proc/meminfo", exc_info=True)

        # Disk
        project_dir = str(Path(__file__).resolve().parents[2])
        disk_raw = await self._run_cmd("df", "-h", project_dir)
        if disk_raw:
            parts = disk_raw.splitlines()[-1].split()
            if len(parts) >= 5:
                disk_pct = int(parts[4].rstrip("%")) if parts[4].rstrip("%").isdigit() else 0
                warn = " 🔴" if disk_pct >= 95 else (" ⚠️" if disk_pct >= 80 else "")
                lines.append(f"  Disk: {parts[2]} used, {parts[3]} free ({parts[4]}){warn}")

        return "\n".join(lines)

    async def _status_services(self) -> str:
        """Systemd service + timer health."""
        # Check all services concurrently
        states = await asyncio.gather(
            *[self._run_cmd("systemctl", "is-active", svc) for svc in self._STATUS_SERVICES],
            return_exceptions=True,
        )

        ok = []
        fail = []
        for svc, state in zip(self._STATUS_SERVICES, states, strict=True):
            short = svc.replace(".service", "").replace("osia-", "")
            if isinstance(state, Exception) or state != "active":
                fail.append(f"  ❌ {short} ({state if isinstance(state, str) else 'error'})")
            else:
                ok.append(short)

        lines = [f"SERVICES [{len(ok)}/{len(self._STATUS_SERVICES)} active]"]
        if fail:
            lines.extend(fail)
        else:
            lines.append("  All services active")

        # Timers with next trigger
        lines.append("")
        lines.append("TIMERS")
        for timer in self._STATUS_TIMERS:
            active = await self._run_cmd("systemctl", "is-active", timer)
            short = timer.replace(".timer", "").replace("osia-", "")
            if active != "active":
                lines.append(f"  ❌ {short} ({active or 'inactive'})")
            else:
                next_run = await self._run_cmd("systemctl", "show", "-p", "NextElapseUSecRealtime", "--value", timer)
                # Trim to "YYYY-MM-DD HH:MM" if it looks like a date string
                next_short = next_run[:16] if len(next_run) > 10 else next_run
                lines.append(f"  ✅ {short}  →  {next_short or 'n/a'}")

        return "\n".join(lines)

    async def _status_docker(self) -> str:
        """Docker container states."""
        # Single docker inspect call for all containers
        fmt = "{{.Name}}: {{.State.Status}}"
        raw = await self._run_cmd("docker", "inspect", "--format", fmt, *self._STATUS_CONTAINERS, timeout=8.0)

        lines = [f"DOCKER [{len(self._STATUS_CONTAINERS)} containers]"]
        if not raw:
            lines.append("  docker unavailable or no containers found")
            return "\n".join(lines)

        ok_count = 0
        for line in raw.splitlines():
            if ": " not in line:
                continue
            name, state = line.split(": ", 1)
            name = name.lstrip("/")
            if state == "running":
                ok_count += 1
                lines.append(f"  ✅ {name}")
            else:
                lines.append(f"  ❌ {name} ({state})")

        # Rewrite header with count
        lines[0] = f"DOCKER [{ok_count}/{len(self._STATUS_CONTAINERS)} running]"
        return "\n".join(lines)

    async def _status_api(self) -> str:
        """HTTP and Redis health checks."""
        lines = ["API HEALTH"]

        qdrant_url = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        queue_api_ua = os.getenv("QUEUE_API_UA_SENTINEL", "osia-worker/1")

        async def _http_check(name: str, url: str, headers: dict | None = None) -> str:
            try:
                async with httpx.AsyncClient(timeout=4.0) as c:
                    resp = await c.get(url, headers=headers or {})
                ok = 200 <= resp.status_code < 400
                return f"  {'✅' if ok else '❌'} {name} (HTTP {resp.status_code})"
            except Exception as exc:
                return f"  ❌ {name} (timeout/error: {type(exc).__name__})"

        qdrant_headers = {"api-key": qdrant_key} if qdrant_key else {}
        checks = await asyncio.gather(
            _http_check("Qdrant", f"{qdrant_url}/collections", qdrant_headers),
            _http_check("Signal API", f"{self.signal_api_url}/v1/about"),
            _http_check("Queue API", "http://localhost:8098/health", {"User-Agent": queue_api_ua}),
            _http_check("Status API", "http://localhost:8099/health"),
        )
        lines.extend(checks)

        # Redis ping
        try:
            pong = await self.redis.ping()
            lines.append(f"  {'✅' if pong else '❌'} Redis")
        except Exception as exc:
            lines.append(f"  ❌ Redis ({exc})")

        # Phone bridge
        try:
            async with httpx.AsyncClient(timeout=4.0) as c:
                resp = await c.get("http://localhost:8006/health")
            data = resp.json()
            connected = data.get("phone_connected", False)
            device = data.get("device_id", "none")
            icon = "✅" if connected else "⚠️"
            lines.append(f"  {icon} Phone Bridge (device: {device}, connected: {connected})")
        except Exception:
            lines.append("  ❌ Phone Bridge (unreachable)")

        return "\n".join(lines)

    async def _status_qdrant(self) -> str:
        """Full Qdrant KB collection listing with vector counts."""
        qdrant_url = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        headers = {"api-key": qdrant_key} if qdrant_key else {}

        lines = ["QDRANT KB"]
        try:
            async with httpx.AsyncClient(timeout=8.0) as c:
                resp = await c.get(f"{qdrant_url}/collections", headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            lines.append(f"  ❌ Qdrant unavailable: {exc}")
            return "\n".join(lines)

        collections = sorted(
            data.get("result", {}).get("collections", []),
            key=lambda x: x.get("name", ""),
        )
        if not collections:
            lines.append("  No collections found")
            return "\n".join(lines)

        total_points = 0
        coll_details = await asyncio.gather(
            *[self._qdrant_collection_info(qdrant_url, headers, coll["name"]) for coll in collections],
            return_exceptions=True,
        )
        for coll, info in zip(collections, coll_details, strict=True):
            name = coll["name"]
            if isinstance(info, Exception) or info is None:
                lines.append(f"  • {name}: (error)")
            else:
                pts = info.get("points_count", 0) or 0
                total_points += pts
                lines.append(f"  • {name}: {pts:,} pts")

        lines.append("  ─────────────────────────")
        lines.append(f"  Total: {len(collections)} collections, {total_points:,} points")
        return "\n".join(lines)

    async def _qdrant_collection_info(self, base_url: str, headers: dict, name: str) -> dict | None:
        """Fetch point/vector counts for a single Qdrant collection."""
        try:
            async with httpx.AsyncClient(timeout=6.0) as c:
                resp = await c.get(f"{base_url}/collections/{name}", headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data.get("result", {})
        except Exception:
            return None

    async def _status_qdrant_compact(self) -> str:
        """Single-line Qdrant summary for the full /status view."""
        qdrant_url = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        headers = {"api-key": qdrant_key} if qdrant_key else {}
        try:
            async with httpx.AsyncClient(timeout=6.0) as c:
                resp = await c.get(f"{qdrant_url}/collections", headers=headers)
                resp.raise_for_status()
                data = resp.json()
            colls = data.get("result", {}).get("collections", [])
            return f"QDRANT KB\n  {len(colls)} collections (use /status qdrant for counts)"
        except Exception:
            return "QDRANT KB\n  ❌ Qdrant unavailable"

    async def _status_queue(self) -> str:
        """Redis queue depths with optional next-task preview."""
        lines = ["QUEUES"]
        try:
            task_depth, research_depth, rss_depth = await asyncio.gather(
                self.redis.llen("osia:task_queue"),
                self.redis.llen("osia:research_queue"),
                self.redis.llen("osia:rss:daily_digest"),
            )
        except Exception as exc:
            lines.append(f"  ❌ Redis unavailable: {exc}")
            return "\n".join(lines)

        lines.append(f"  task_queue:       {task_depth} pending")
        lines.append(f"  research_queue:   {research_depth} pending")
        lines.append(f"  rss:daily_digest: {rss_depth} staged")

        if task_depth > 0:
            try:
                raw = await self.redis.lindex("osia:task_queue", 0)
                if raw:
                    preview = json.loads(raw).get("query", str(raw))[:120]
                    lines.append(f"  Next task: {preview}")
            except Exception:
                logger.debug("Could not fetch next task preview from Redis", exc_info=True)

        return "\n".join(lines)

    async def _status_worker(self) -> str:
        """Research worker timer, last run, and queue stats."""
        lines = ["RESEARCH WORKER"]

        # Timer next run
        active = await self._run_cmd("systemctl", "is-active", "osia-research-worker.timer")
        if active == "active":
            next_run = await self._run_cmd(
                "systemctl",
                "show",
                "-p",
                "NextElapseUSecRealtime",
                "--value",
                "osia-research-worker.timer",
            )
            next_short = next_run[:16] if len(next_run) > 10 else next_run
            lines.append(f"  Timer: ✅ active  →  next {next_short or 'n/a'}")
        else:
            lines.append(f"  Timer: ❌ {active or 'inactive'}")

        # Last journal entry from the worker service
        last_log = await self._run_cmd(
            "journalctl",
            "-u",
            "osia-research-worker.service",
            "-n",
            "3",
            "--no-pager",
            "-q",
            "--output=short",
            timeout=6.0,
        )
        if last_log:
            # Trim to last two lines for Signal readability
            log_lines = [ln for ln in last_log.splitlines() if ln.strip()][-2:]
            lines.append("  Last runs:")
            for ln in log_lines:
                lines.append(f"    {ln[:100]}")

        # Queue and dedup stats
        try:
            research_depth, seen_count = await asyncio.gather(
                self.redis.llen("osia:research_queue"),
                self.redis.scard("osia:research:seen_topics"),
            )
            lines.append(f"  Pending:  {research_depth} queued, {seen_count} already seen")
        except Exception:
            lines.append("  Redis unavailable — cannot check queue")

        return "\n".join(lines)

    async def _status_worker_compact(self) -> str:
        """Single-line worker summary for the full /status view."""
        active = await self._run_cmd("systemctl", "is-active", "osia-research-worker.timer")
        try:
            depth = await self.redis.llen("osia:research_queue")
        except Exception:
            depth = "?"
        icon = "✅" if active == "active" else "❌"
        return f"RESEARCH WORKER\n  Timer: {icon}  |  Research queue: {depth} pending"

    async def _cmd_desks(self, recipient: str) -> None:
        """List all valid desk slugs and their aliases."""
        desk_list = "\n".join(f"  • {s}" for s in sorted(self.valid_desks))
        alias_list = "\n".join(f"  {alias} → {slug}" for alias, slug in sorted(self._DESK_ALIASES.items()))
        await self.send_signal_message(
            recipient,
            f"Available desks:\n{desk_list}\n\nAliases (for /desk and /investigate):\n{alias_list}",
        )

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
        elif name == "kali_nmap":
            result = await self.kali.call_tool("nmap", args)
            return _format_kali_result(result)
        elif name == "kali_whois":
            result = await self.kali.call_tool("whois", args)
            return _format_kali_result(result)
        elif name == "kali_dig":
            result = await self.kali.call_tool("dig", args)
            return _format_kali_result(result)
        elif name == "kali_sslscan":
            result = await self.kali.call_tool("sslscan", args)
            return _format_kali_result(result)
        elif name == "kali_whatweb":
            result = await self.kali.call_tool("whatweb", args)
            return _format_kali_result(result)
        elif name == "kali_harvester":
            result = await self.kali.call_tool("harvester", args)
            return _format_kali_result(result)
        elif name == "kali_amass":
            result = await self.kali.call_tool("amass", args)
            return _format_kali_result(result)
        elif name == "kali_curl_probe":
            result = await self.kali.call_tool("curl", args)
            return _format_kali_result(result)
        elif name == "kali_ping":
            result = await self.kali.call_tool("ping", args)
            return _format_kali_result(result)
        elif name == "kali_traceroute":
            result = await self.kali.call_tool("traceroute", args)
            return _format_kali_result(result)
        elif name == "kali_nikto":
            result = await self.kali.call_tool("nikto", args)
            return _format_kali_result(result)
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
                "--js-runtimes",
                "node",
                "--remote-components",
                "ejs:github",
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
            for _attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=config,
                    )
                    break
                except Exception as e:
                    msg = str(e)
                    msg_lower = msg.lower()
                    _is_network = any(
                        tok in msg_lower for tok in ("ssl", "handshake", "timed out", "connection", "reset", "eof")
                    )
                    _is_capacity = any(tok in msg for tok in ("503", "429", "UNAVAILABLE", "502", "504", "OVERLOADED"))
                    if _is_network:
                        # Network-level failure — provider is unreachable right now.
                        # Retrying won't help; fall through to _research_fallback immediately.
                        logger.warning(
                            "Research loop: network error (%s) — skipping retries, falling back.",
                            msg[:80],
                        )
                        raise
                    elif _attempt < 2 and _is_capacity:
                        # Capacity error (429/503) — short backoff is worth trying.
                        wait = 10 * (_attempt + 1)
                        logger.warning(
                            "Research loop Gemini attempt %d failed (%s), retrying in %ds",
                            _attempt + 1,
                            msg[:80],
                            wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise

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

    async def _download_video_yt_dlp(self, url: str) -> str | None:
        """Try to download the actual video via yt-dlp for direct analysis.

        For public Instagram reels (and TikTok/Facebook) this gives the original
        encoded video with full-quality audio — far superior to ADB screenrecord
        which routes audio through Android's screencast pipeline.

        Returns the local file path on success, or None if the content is
        login-gated, geo-blocked, or yt-dlp otherwise fails.
        """
        yt_dlp_bin = self.base_dir / ".venv" / "bin" / "yt-dlp"
        if not yt_dlp_bin.exists():
            yt_dlp_bin = Path("yt-dlp")

        output_path = str(self.base_dir / "osia_ytdlp_video.mp4")

        cmd = [
            str(yt_dlp_bin),
            "--no-playlist",
            # Best quality up to 720p with audio; fall back progressively
            "-f",
            "bv[height<=720]+ba/b[height<=720]/best[height<=720]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            output_path,
            "--socket-timeout",
            "30",
        ]
        # Re-use the same cookie files as the metadata fetcher
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
            proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=120)
            out = Path(output_path)
            if proc.returncode == 0 and out.exists() and out.stat().st_size > 10_000:
                logger.info(
                    "yt-dlp video download succeeded (%d bytes): %s",
                    out.stat().st_size,
                    output_path,
                )
                return output_path
            logger.info(
                "yt-dlp video download unavailable (rc=%d) — content may require login.\nstderr: %s",
                proc.returncode,
                proc.stderr[-300:] if proc.stderr else "(empty)",
            )
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp video download timed out after 120s")
        except Exception as e:
            logger.warning("yt-dlp video download error: %s", e)
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
        if urlparse(url).hostname in ("instagram.com", "www.instagram.com") and not data.get("comment_count"):
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

        _OR_HEADERS = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://osia.dev",
            "X-Title": "OSIA Intelligence Framework",
        }
        _TRANSIENT = ("503", "502", "429", "UNAVAILABLE", "overloaded")

        for tier, model_id in enumerate(
            ("meta-llama/llama-3.1-8b-instruct", "mistralai/mistral-small-2603"),
            start=1,
        ):
            for attempt in range(2):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as http:
                        resp = await http.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=_OR_HEADERS,
                            json={
                                "model": model_id,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 300,
                                "temperature": 0.2,
                            },
                        )
                        resp.raise_for_status()
                        return resp.json()["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    msg = str(e)
                    if attempt == 0 and any(t in msg for t in _TRANSIENT):
                        logger.warning(
                            "Sentiment tier %d (%s) attempt 1 failed (%s), retrying", tier, model_id, msg[:80]
                        )
                        await asyncio.sleep(10)
                    else:
                        logger.warning(
                            "Sentiment tier %d (%s) failed (%s) — trying next tier", tier, model_id, msg[:80]
                        )
                        break

        # Tier 3: Gemini direct (last resort)
        try:
            response = self.client.models.generate_content(model=self.model_id, contents=prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning("Comment sentiment analysis all tiers failed: %s", e)
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
        _MAX_DURATION = 900
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

        try:
            return await self._analyse_video_file(local_path, engagement_counts)
        finally:
            try:
                Path(local_path).unlink(missing_ok=True)
            except Exception as exc:
                logger.debug("Could not remove ADB capture file %s: %s", local_path, exc)

    async def _analyse_video_file(
        self,
        video_path: str,
        engagement_counts: dict,
    ) -> tuple[str | None, dict]:
        """Transcode a local video and analyse it via Gemini (falling back to Reka).

        Shared by both the ADB screenrecord path (called from process_media_link)
        and the yt-dlp direct download path.  Cleans up the transcoded temp file
        but leaves ``video_path`` untouched — callers are responsible for their own
        input file cleanup.

        Audio is encoded at 128 kbps (up from the old 64 kbps) so that speech is
        preserved at sufficient fidelity for verbatim transcription.
        """
        _GEMINI_MAX_SECS = 180
        upload_path = video_path
        transcoded_path = video_path.replace(".mp4", "_tc.mp4")

        ff = await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-t",
                str(_GEMINI_MAX_SECS),  # hard cap at 3 minutes
                "-vf",
                "scale='min(720,iw)':-2",  # 720p max — sufficient for content analysis
                "-b:v",
                "500k",  # 500 kbps video — readable text/faces at 720p
                "-b:a",
                "128k",  # 128 kbps audio — sufficient for verbatim transcription
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                transcoded_path,
            ],
            capture_output=True,
            timeout=_GEMINI_MAX_SECS + 60,
        )
        if ff.returncode == 0:
            upload_path = transcoded_path
            logger.info("Transcoded to 720p/500k/128k for Gemini upload (%s).", transcoded_path)
        else:
            logger.warning(
                "ffmpeg transcode failed (rc=%d) — uploading original.\n%s",
                ff.returncode,
                ff.stderr.decode(errors="replace")[:500],
            )

        try:
            _MAX_UPLOAD_ATTEMPTS = 3
            video_file = None
            for attempt in range(1, _MAX_UPLOAD_ATTEMPTS + 1):
                try:
                    logger.info(
                        "Uploading video to Gemini (%s) — attempt %d/%d...",
                        upload_path,
                        attempt,
                        _MAX_UPLOAD_ATTEMPTS,
                    )
                    video_file = await asyncio.to_thread(self.client.files.upload, file=upload_path)
                    break
                except Exception as upload_err:
                    logger.warning(
                        "Gemini upload attempt %d/%d failed: %s",
                        attempt,
                        _MAX_UPLOAD_ATTEMPTS,
                        upload_err,
                    )
                    if attempt < _MAX_UPLOAD_ATTEMPTS:
                        msg = str(upload_err).lower()
                        is_network_err = any(
                            tok in msg for tok in ("ssl", "handshake", "timed out", "connection", "reset")
                        )
                        if is_network_err and attempt >= 2:
                            logger.warning(
                                "Gemini upload: persistent network error — skipping final retry, falling back to Reka."
                            )
                            return await self._analyse_video_reka(upload_path, engagement_counts)
                        wait = 5 if is_network_err else 5 * attempt
                        logger.info("Retrying upload in %ds...", wait)
                        await asyncio.sleep(wait)
                    else:
                        logger.error(
                            "Gemini upload failed after %d attempts — trying Reka fallback.",
                            _MAX_UPLOAD_ATTEMPTS,
                        )
                        return await self._analyse_video_reka(upload_path, engagement_counts)

            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                video_file = await asyncio.to_thread(self.client.files.get, name=video_file.name)

            prompt = (
                "Watch this intercepted video. Transcribe ALL spoken audio verbatim "
                "and as completely as possible — do not paraphrase or summarise speech, "
                "reproduce the exact words spoken. Then describe the visual context, "
                "identify any text on screen, and summarize the core message or narrative."
            )
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_id,
                    contents=[video_file, prompt],
                )
                return response.text, engagement_counts
            except Exception as gc_err:
                logger.warning("Gemini generate_content failed: %s — trying Reka fallback.", gc_err)
                return await self._analyse_video_reka(upload_path, engagement_counts)
        finally:
            try:
                Path(transcoded_path).unlink(missing_ok=True)
            except Exception as exc:
                logger.debug("Could not remove transcoded file %s: %s", transcoded_path, exc)

    async def _research_fallback(self, query: str) -> tuple[str, SourceTracker]:
        """Direct-tool research when the Gemini multi-turn loop is unavailable.

        Runs Tavily web search and Wikipedia in parallel without a reasoning
        loop — no LLM involved, so works even when every Google endpoint is down.
        Returns concatenated raw results; the desk model provides the synthesis.
        """
        logger.info("Direct-tool research fallback for: %s", query[:120])
        tracker = SourceTracker()
        parts: list[str] = []
        q = query[:500]  # keep tool queries short

        web, wiki = await asyncio.gather(
            self.mcp.call_tool("tavily", "tavily_search", {"query": q}),
            self.mcp.call_tool("wikipedia", "search_pages", {"input": {"query": q[:200]}}),
            return_exceptions=True,
        )

        if not isinstance(web, BaseException):
            text = _extract_mcp_text(web)
            if text:
                parts.append(f"## Web Search (Tavily)\n{text}")
                tracker.record("search_web", q, text)
            else:
                logger.debug("Fallback: Tavily returned empty result")
        else:
            logger.warning("Fallback: Tavily failed: %s", web)

        if not isinstance(wiki, BaseException):
            text = _extract_mcp_text(wiki)
            if text:
                parts.append(f"## Wikipedia\n{text}")
                tracker.record("search_wikipedia", q[:200], text)
            else:
                logger.debug("Fallback: Wikipedia returned empty result")
        else:
            logger.warning("Fallback: Wikipedia failed: %s", wiki)

        return "\n\n".join(parts), tracker

    # ------------------------------------------------------------------
    # Direct article URL fetcher
    # ------------------------------------------------------------------

    async def _fetch_article_url(self, url: str) -> str | None:
        """Fetch and extract article text from a URL using browser impersonation.

        Tier 1 — curl_cffi with Chrome TLS fingerprint impersonation.
            Spoofs the TLS ClientHello signature of a real Chrome browser,
            defeating fingerprint-based bot detection used by most news sites.

        Tier 2 — httpx with browser-like headers.
            Covers sites that check headers/UA but not TLS fingerprint.

        Both tiers use trafilatura for article text extraction, with a
        BeautifulSoup fallback for pages that don't parse cleanly.
        Returns None if the page is unreachable or yields no extractable text.
        """
        import trafilatura
        from bs4 import BeautifulSoup
        from curl_cffi.requests import AsyncSession

        _BROWSER_HEADERS = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

        html: str | None = None

        # Tier 1: curl_cffi — Chrome TLS fingerprint impersonation
        try:
            async with AsyncSession(impersonate="chrome") as session:
                resp = await session.get(url, headers=_BROWSER_HEADERS, timeout=20, allow_redirects=True)
                if resp.status_code == 200:
                    html = resp.text
                else:
                    logger.warning("curl_cffi fetch got HTTP %d for %s", resp.status_code, url)
        except Exception as e:
            logger.warning("curl_cffi article fetch failed for %s: %s", url, e)

        # Tier 2: httpx with browser-like headers
        if not html:
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(20.0),
                    follow_redirects=True,
                    headers=_BROWSER_HEADERS,
                ) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        html = resp.text
                    else:
                        logger.warning("httpx article fetch got HTTP %d for %s", resp.status_code, url)
            except Exception as e:
                logger.warning("httpx article fetch failed for %s: %s", url, e)

        if not html:
            return None

        # Primary: trafilatura — best-in-class article extraction
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )

        # Fallback: BeautifulSoup plain-text extraction
        if not text:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                tag.decompose()
            raw = soup.get_text(separator="\n", strip=True)
            # Keep only lines with substance (> 40 chars) to strip nav/boilerplate
            lines = [ln for ln in raw.splitlines() if len(ln.strip()) > 40]
            text = "\n".join(lines)[:20000] or None

        if text:
            logger.info("Article fetch extracted %d chars from %s", len(text), url)
        return text or None

    # ------------------------------------------------------------------
    # Reka video analysis fallback
    # ------------------------------------------------------------------

    async def _analyse_video_reka(
        self,
        upload_path: str,
        engagement_counts: dict,
    ) -> tuple[str | None, dict]:
        """Analyse a captured video via Reka — fallback when Gemini is unavailable.

        Two-tier fallback covering any video duration:

        Tier 1 — OpenRouter rekaai/reka-edge
            60s-trimmed base64 clip. Fast — no upload round-trip. Uses the
            existing OPENROUTER_API_KEY. Good for typical short reels.

        Tier 2 — Reka Vision API (vision-agent.api.reka.ai)
            Full multipart file upload + async indexing + Q&A. Handles any
            duration up to the 180s transcode cap (and beyond if needed).
            Requires REKA_API_KEY. Remote video is deleted after analysis.

        Returns (analysis_text, engagement_counts), or (None, engagement_counts)
        if both tiers fail so the caller can degrade to metadata-only analysis.
        """
        prompt = (
            "Watch this intercepted video. Transcribe ALL spoken audio verbatim "
            "and as completely as possible — do not paraphrase or summarise speech, "
            "reproduce the exact words spoken. Then describe the visual context, "
            "identify any text on screen, and summarize the core message or narrative."
        )
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        reka_key = os.getenv("REKA_API_KEY")

        # Probe duration so we can decide which tier to use.
        # Tier 1 only covers the first 60s — skip it entirely for longer videos
        # so we don't silently return a partial analysis.
        _OR_MAX_SECS = 60
        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    upload_path,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            video_duration = float(probe.stdout.strip() or "0")
        except Exception:
            video_duration = 0.0
        logger.info("Reka fallback: video duration=%.0fs (Tier 1 threshold=%ds)", video_duration, _OR_MAX_SECS)

        # ── Tier 1: OpenRouter reka-edge — base64, 60s trim ───────────────────
        # Skipped for videos longer than 60s — Tier 2 handles those in full.
        if openrouter_key and video_duration <= _OR_MAX_SECS:
            _OR_MAX_SECS = 60
            trimmed_path = str(Path(upload_path).with_name(Path(upload_path).stem + "_reka60.mp4"))
            try:
                ff = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        upload_path,
                        "-t",
                        str(_OR_MAX_SECS),
                        "-c",
                        "copy",
                        trimmed_path,
                    ],
                    capture_output=True,
                    timeout=60,
                )
                use_path = trimmed_path if ff.returncode == 0 else upload_path
                video_bytes = Path(use_path).read_bytes()
                data_uri = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode()

                async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http:
                    resp = await http.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "rekaai/reka-edge",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "video_url", "video_url": {"url": data_uri}},
                                    ],
                                }
                            ],
                        },
                    )
                    resp.raise_for_status()
                    analysis = resp.json()["choices"][0]["message"]["content"]
                    logger.info("Reka video analysis succeeded via OpenRouter (60s clip).")
                    return analysis, engagement_counts
            except Exception as or_err:
                logger.warning("Reka via OpenRouter failed: %s — trying Reka Vision API.", or_err)
            finally:
                try:
                    Path(trimmed_path).unlink(missing_ok=True)
                except Exception:
                    pass  # best-effort cleanup — ignore if already deleted

        # ── Tier 2: Reka Vision API — full video, any duration ────────────────
        if not reka_key:
            logger.error("REKA_API_KEY not set — cannot use Vision API. All video analysis providers exhausted.")
            return None, engagement_counts

        _VISION_BASE = "https://vision-agent.api.reka.ai"
        _VISION_HEADERS = {"X-Api-Key": reka_key}
        video_id: str | None = None

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http:
                # Upload
                logger.info("Reka Vision API: uploading %s...", upload_path)
                with open(upload_path, "rb") as fh:
                    up_resp = await http.post(
                        f"{_VISION_BASE}/v1/videos/upload",
                        headers=_VISION_HEADERS,
                        files={"file": ("video.mp4", fh, "video/mp4")},
                        data={
                            "video_name": f"osia_{uuid.uuid4().hex[:8]}",
                            "index": "true",
                        },
                    )
                up_resp.raise_for_status()
                video_id = up_resp.json()["video_id"]
                logger.info(
                    "Reka Vision: upload complete (video_id=%s) — polling for indexing...",
                    video_id,
                )

                # Poll until indexed — up to 5 minutes
                for _ in range(60):
                    await asyncio.sleep(5)
                    poll = await http.get(
                        f"{_VISION_BASE}/v1/videos/{video_id}",
                        headers=_VISION_HEADERS,
                    )
                    status = poll.json().get("indexing_status", "")
                    logger.debug("Reka Vision indexing status: %s", status)
                    if status == "indexed":
                        break
                    if status == "failed":
                        raise RuntimeError(f"Reka Vision indexing failed (video_id={video_id})")
                else:
                    raise TimeoutError("Reka Vision indexing did not complete within 5 minutes")

                # Q&A — API uses messages array (chat format), not a bare "question" field
                qa_resp = await http.post(
                    f"{_VISION_BASE}/v1/qa/chat",
                    headers={**_VISION_HEADERS, "Content-Type": "application/json"},
                    json={
                        "video_id": video_id,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if not qa_resp.is_success:
                    logger.error(
                        "Reka Vision Q&A returned %d: %s",
                        qa_resp.status_code,
                        qa_resp.text[:300],
                    )
                qa_resp.raise_for_status()
                qa_data = qa_resp.json()
                # chat_response is a JSON string containing a sections array;
                # concatenate all markdown sections to get the full analysis.
                analysis = None
                chat_response_raw = qa_data.get("chat_response")
                if chat_response_raw:
                    try:
                        cr = json.loads(chat_response_raw)
                        sections = cr.get("sections", [])
                        analysis = "\n\n".join(s["markdown"] for s in sections if s.get("markdown")).strip() or None
                    except (json.JSONDecodeError, TypeError, KeyError):
                        # Fallback: use the raw string directly if it isn't JSON
                        analysis = chat_response_raw.strip() or None
                if not analysis:
                    logger.error(
                        "Reka Vision Q&A: could not extract text. Keys: %s. Raw: %s",
                        list(qa_data.keys()),
                        str(qa_data)[:300],
                    )
                    raise ValueError(f"Unrecognised Reka Q&A response shape: {list(qa_data.keys())}")
                logger.info("Reka Vision API analysis succeeded (video_id=%s).", video_id)
                return analysis, engagement_counts

        except Exception as vision_err:
            logger.error("Reka Vision API failed: %s", vision_err)
            return None, engagement_counts
        finally:
            # Always delete the remote video to avoid accumulating storage
            if video_id:
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as http:
                        await http.delete(
                            f"{_VISION_BASE}/v1/videos/{video_id}",
                            headers=_VISION_HEADERS,
                        )
                    logger.debug("Reka Vision: deleted remote video_id=%s", video_id)
                except Exception as del_err:
                    logger.debug("Reka Vision cleanup failed (non-fatal): %s", del_err)

    # ------------------------------------------------------------------
    # Desk routing via Venice (uncensored — no query is refused or misrouted)
    # ------------------------------------------------------------------

    async def _route_to_desk(self, prompt: str) -> str:
        """Call Venice uncensored to select a desk slug. Falls back to OpenRouter then Gemini."""
        # Tier 1: Venice (uncensored — never refuses routing queries)
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
            logger.warning("Venice routing failed (%s) — falling back to OpenRouter", e)

        # Tier 2: OpenRouter
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://osia.dev",
                        "X-Title": "OSIA Intelligence Framework",
                    },
                    json={
                        "model": "meta-llama/llama-3.1-8b-instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 64,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("OpenRouter routing failed (%s) — falling back to Gemini", e)

        # Tier 3: Gemini direct
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
                    # Collections with pre-computed signal-strength weights get a score
                    # multiplier so high-signal records surface above low-signal noise.
                    _BOOST_FIELDS = {
                        "gdelt-events": "mentions_weight",
                        "acled-conflict-events": "fatality_weight",
                        "epa-toxic-releases": "release_weight",
                        "noaa-storm-events": "damage_weight",
                    }
                    boost_field = _BOOST_FIELDS.get(col)
                    return await self.qdrant.search(
                        col,
                        cross_search_query,
                        boost_top_k,
                        decay_half_life_days=70.0,
                        payload_boost_field=boost_field,
                    )
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

        # Intercept slash commands before any media/desk processing.
        # /desk mutates task in-place and returns False so normal flow continues.
        if original_query.strip().startswith("/"):
            if await self._handle_signal_command(task, source):
                return
            # Re-extract in case /desk mutated the query and desk fields
            original_query = task.get("query", original_query)

        query = original_query
        media_analysis = None
        research_summary = None
        source_tracker: SourceTracker | None = None
        url: str | None = None
        _article_fetched = False

        logger.info("Received new task from %s: %s", source, query)

        # Immediate acknowledgement — let the operative know OSIA has the task
        # before any slow processing (ADB capture, research loop etc.) begins.
        if source.startswith("signal:"):
            _recipient = source[len("signal:") :]
            _ack_url_match = re.search(r"(https?://[^\s]+)", original_query)
            _ack_url = _ack_url_match.group(1).lower() if _ack_url_match else ""
            if any(d in _ack_url for d in MEDIA_DOMAINS):
                _platform = next(d.split(".")[0].capitalize() for d in MEDIA_DOMAINS if d in _ack_url)
                _ack_msg = (
                    f"⬛ OSIA TASKED ⬛\n"
                    f"{_platform} reel intercepted — media extraction initiated.\n"
                    f"Stand by for full INTSUM (~2–3 min)."
                )
            elif any(d in _ack_url for d in YOUTUBE_DOMAINS):
                _ack_msg = (
                    "⬛ OSIA TASKED ⬛\nYouTube video detected — transcript extraction initiated.\nINTSUM incoming."
                )
            elif _ack_url:
                _ack_msg = "⬛ OSIA TASKED ⬛\nURL received — research and analysis initiated.\nINTSUM incoming."
            else:
                _ack_msg = (
                    "⬛ OSIA TASKED ⬛\nQuery received — research loop and desk analysis initiated.\nINTSUM incoming."
                )
            await self.send_signal_message(_recipient, _ack_msg)

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
                    # Try yt-dlp direct download first — runs in parallel with metadata
                    # fetch so we don't pay extra time if both succeed.
                    # For public reels this gives original audio quality; login-gated
                    # content returns None and we fall back to ADB screenrecord.
                    yt_video_result, raw_meta = await asyncio.gather(
                        self._download_video_yt_dlp(url),
                        self._fetch_yt_dlp_metadata(url),
                        return_exceptions=True,
                    )
                    yt_video_path = None if isinstance(yt_video_result, BaseException) else yt_video_result
                    if isinstance(yt_video_result, BaseException):
                        logger.warning("yt-dlp video download raised: %s", yt_video_result)
                    if isinstance(raw_meta, BaseException):
                        logger.warning("Social metadata fetch failed: %s", raw_meta)
                        raw_meta = None

                    adb_analysis = None
                    screen_counts: dict = {}

                    if yt_video_path:
                        # Public content: analyse the clean yt-dlp download directly.
                        # No ADB recording needed — engagement counts come from metadata.
                        logger.info("Using yt-dlp video for analysis (skipping ADB screenrecord)")
                        try:
                            adb_analysis, _ = await self._analyse_video_file(yt_video_path, {})
                        except Exception as ytdlp_err:
                            logger.error("yt-dlp video analysis failed: %s", ytdlp_err)
                        finally:
                            try:
                                Path(yt_video_path).unlink(missing_ok=True)
                            except Exception:
                                pass  # best-effort cleanup — ignore if already deleted
                    else:
                        # Private/login-required or yt-dlp unavailable — fall back to
                        # ADB screenrecord. Metadata was already fetched above.
                        logger.info("yt-dlp download unavailable — falling back to ADB screenrecord")
                        try:
                            adb_analysis, screen_counts = await self.process_media_link(url)
                        except Exception as adb_err:
                            logger.error("ADB media interception failed: %s", adb_err)

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
                    else:
                        # Both yt-dlp and ADB failed — nothing to route to a desk.
                        # Notify the operative and abort rather than forwarding a bare URL.
                        logger.warning("Media interception produced no content for %s — aborting task.", url)
                        if source.startswith("signal:"):
                            _recipient = source[len("signal:") :]
                            await self.send_signal_message(
                                _recipient,
                                f"⚠️ OSIA — Collection failure\n"
                                f"No content could be extracted from:\n{url}\n\n"
                                f"yt-dlp: login required or rate-limited\n"
                                f"ADB: device offline or unavailable\n\n"
                                f"Resubmit once the ADB device is reconnected and/or cookies are refreshed.",
                            )
                        return
                except Exception as e:
                    logger.error("Media interception failed: %s", e)

            else:
                # Plain URL (news article, blog, report) — attempt direct fetch
                # before falling back to Tavily. This bypasses Tavily rate limits
                # and gives the desk the actual source text, not a search summary.
                try:
                    article_text = await self._fetch_article_url(url)
                    if article_text:
                        _article_fetched = True
                        media_analysis = article_text
                        query = f"## ARTICLE CONTENT\nSource: {url}\n\n{article_text}\n\nOriginal Request: {query}"
                    else:
                        logger.warning("Direct article fetch returned no content for %s — Tavily will be used", url)
                except Exception as e:
                    logger.error("Direct article fetch raised: %s", e)

        # 2. Automated Research (multi-turn tool calling)
        # Skip for social media ADB tasks (Instagram/TikTok/Facebook) — the ADB capture
        # and yt-dlp metadata already constitute the intelligence gathering for these URLs.
        # Running handle_research on a raw social media URL produces useless Tavily results.
        # Also skip when direct article fetch already supplied the full source text —
        # the desk has everything it needs and additional Tavily calls risk rate limits.
        _is_social_media_url = url and any(domain in url.lower() for domain in MEDIA_DOMAINS)
        if not _is_social_media_url and not _article_fetched:
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
                logger.error("Automated research failed: %s — attempting direct-tool fallback.", e)
                try:
                    research_summary, source_tracker = await self._research_fallback(
                        original_query or media_analysis or query
                    )
                    if research_summary:
                        logger.info("Direct-tool research fallback produced %d chars.", len(research_summary))
                        manifest = source_tracker.format_manifest() if source_tracker else ""
                        citation_block = build_citation_protocol()
                        query = (
                            f"Baseline research summary for this topic:\n{research_summary}\n\n"
                            f"{manifest}\n\n"
                            f"{citation_block}\n\n"
                            f"Original Request: {query}"
                        )
                except Exception as fallback_err:
                    logger.error("Direct-tool research fallback also failed: %s", fallback_err)

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

        valid_desks_list = "\n".join(f"- {s} ({self.desk_registry.get(s).name})" for s in sorted(self.valid_desks))
        plan_prompt = f"""
        {mandate}

        ---

        You are the Chief of Staff for OSIA. A new Request for Information (RFI) has come in: '{query if original_query else (media_analysis or query)}'

        Which of our specialized desks should analyze this? Choose ONE from the following list:
{valid_desks_list}

        Reply with ONLY the slug of the desk (the part before the parenthesis), nothing else.
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
                    infographic_b64 = await generate_infographic(
                        self.client,
                        self.model_id,
                        analysis,
                        assigned_desk,
                    )
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
