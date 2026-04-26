"""
OSIA Hermes Worker — validates low-confidence intel by searching for corroborating
or contradicting evidence, then updates Qdrant payload in-place and enriches the wiki.

Named after Hermes, messenger of the gods and guide across boundaries — the worker
traverses sources and carries evidence back to validate existing intelligence rather
than generating new intel from scratch. The name also reflects the model choice:
Hermes 4 (NousResearch) runs all desks.

For each desk, scrolls the Qdrant collection for points where reliability_tier is
not "A" (RSS-ingested B-tier, unknown ?, research worker outputs, etc.) that have
not been recently checked. Runs a focused research loop per point and patches the
payload in-place with a structured verdict and updated reliability_tier.

After corroboration, wiki enrichment follows a three-path strategy:
  1. Payload carries wiki_path  →  patch corroboration section + update metadata tier
  2. No wiki_path, topic known  →  search wiki for an entity page and append a note
  3. CORROBORATED HIGH/MODERATE, no wiki page found  →  create entity page; stamp wiki_path back to Qdrant

A quality maintenance pass also runs after each desk's corroboration batch:
  - Deletes empty/stub points (text < HERMES_CLEANUP_MIN_TEXT_LEN chars)
  - Purges tier-C contradiction-flagged points older than HERMES_CLEANUP_STALE_DAYS
  - Purges low-tier points checked HERMES_CLEANUP_UNVERIFIED_CHECKS+ times, always
    UNVERIFIED, and older than HERMES_CLEANUP_UNVERIFIED_AGE_DAYS

Distributed operation (multiple nodes, shared Qdrant + Redis + wiki):
  Each desk is protected by a Redis lease: osia:hermes:lease:<desk-slug>
  Workers claim a desk atomically via SET NX before processing and release via DEL on
  completion (TTL = HERMES_LEASE_TTL seconds so crashed workers don't block forever).
  Workers that find a desk already leased skip it and move on — across N desks, up to
  N workers can run fully in parallel with zero overlap.

  If Redis is unavailable, a warning is logged and the worker proceeds without
  distributed locking (single-node safe mode).

  To inspect active leases:  redis-cli KEYS 'osia:hermes:lease:*'
  To force-clear all leases: uv run python -m src.workers.hermes_worker --clear-leases

Model routing:
  All desks → nousresearch/hermes-4-70b via OpenRouter (SOTA on RefusalBench —
              no censorship, native tool calling, strong structured output)
  Fallback  → mistral-small-3-2-24b-instruct via Venice if OPENROUTER_API_KEY is not set
  Fallback  → Gemini if neither key is available

Verdict transitions:
  CORROBORATED  → missing/? → C; C → B; B → A; A stays A
  CONTRADICTED  → A → B; B → C; C stays C; sets contradiction_flag: true
  UNVERIFIED    → no tier change; marks corroboration_checked_at only

Usage:
  uv run python -m src.workers.hermes_worker [--desk <slug>] [--dry-run] [--limit N]
                                              [--no-cleanup] [--worker-id ID] [--clear-leases]

Environment variables:
  OPENROUTER_API_KEY           — required for Hermes 4 (primary)
  VENICE_API_KEY               — fallback if no OpenRouter key
  GEMINI_API_KEY               — final fallback
  QDRANT_URL, QDRANT_API_KEY, HF_TOKEN, TAVILY_API_KEY, OTX_API_KEY
  WIKIJS_URL, WIKIJS_API_KEY   — wiki enrichment (optional; skipped if not set)
  REDIS_URL                    — Redis for distributed desk leasing (default: redis://localhost:6379/0)

  HERMES_MODEL                 — override OpenRouter model ID
                                  (default: nousresearch/hermes-4-70b)
  HERMES_COOLDOWN_DAYS         — days before a point can be re-checked (default: 14)
  HERMES_BATCH_SIZE            — max points per desk per run (default: 10)
  HERMES_LEASE_TTL             — desk lease TTL in seconds; must exceed max desk runtime
                                  (default: 3600 = 1 hour)

  HERMES_CLEANUP_STALE_DAYS        — days before stale contradicted tier-C points are purged (default: 30)
  HERMES_CLEANUP_MIN_TEXT_LEN      — minimum text length to keep a point (default: 50 chars)
  HERMES_CLEANUP_UNVERIFIED_CHECKS — check count before permanently-unverifiable purge (default: 3)
  HERMES_CLEANUP_UNVERIFIED_AGE_DAYS — age in days for permanently-unverified purge (default: 90)

Manual trigger (fire-and-forget):
  sudo systemctl start --no-block osia-hermes-worker.service

Run directly:
  uv run python -m src.workers.hermes_worker
  uv run python -m src.workers.hermes_worker --desk geopolitical-and-security-desk
  uv run python -m src.workers.hermes_worker --dry-run --limit 3
  uv run python -m src.workers.hermes_worker --worker-id node-2
  uv run python -m src.workers.hermes_worker --clear-leases
"""

import argparse
import asyncio
import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from src.intelligence.wiki_client import (
    WikiClient,
    build_entity_page,
    desk_wiki_section,
    entity_wiki_path,
)

# Tool infrastructure — imported from research_worker to avoid duplication.
# Safe to import: module-level code only loads env vars and defines functions.
from src.workers.research_worker import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REACT_ONLY_MODELS,  # kept for Venice fallback path
    TOOL_SCHEMAS,
    VENICE_API_KEY,
    VENICE_BASE_URL,
    _parse_react,  # kept for Venice fallback path
    get_tool_registry,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("osia.hermes_worker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.osia.dev")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
HERMES_LEASE_TTL = int(os.getenv("HERMES_LEASE_TTL", "3600"))  # seconds; must exceed max desk runtime
_LEASE_PREFIX = "osia:hermes:lease:"

COOLDOWN_DAYS = float(os.getenv("HERMES_COOLDOWN_DAYS", "14"))
COOLDOWN_SECONDS = COOLDOWN_DAYS * 86400
BATCH_SIZE = int(os.getenv("HERMES_BATCH_SIZE", "10"))

CLEANUP_STALE_DAYS = float(os.getenv("HERMES_CLEANUP_STALE_DAYS", "30"))
CLEANUP_MIN_TEXT_LEN = int(os.getenv("HERMES_CLEANUP_MIN_TEXT_LEN", "50"))
CLEANUP_UNVERIFIED_CHECKS = int(os.getenv("HERMES_CLEANUP_UNVERIFIED_CHECKS", "3"))
CLEANUP_UNVERIFIED_AGE_DAYS = float(os.getenv("HERMES_CLEANUP_UNVERIFIED_AGE_DAYS", "90"))

# RAG contamination remediation pass config
SCORING_BATCH_SIZE = int(os.getenv("HERMES_SCORING_BATCH", "20"))
CONTRA_BATCH_SIZE = int(os.getenv("HERMES_CONTRA_BATCH", "10"))
KB_CORROBORATION_THRESHOLD = float(os.getenv("HERMES_KB_THRESHOLD", "0.75"))
CONTRADICTION_SIMILARITY_THRESHOLD = float(os.getenv("HERMES_CONTRA_SIMILARITY", "0.70"))
ECHO_RISK_THRESHOLD = float(os.getenv("HERMES_ECHO_RISK_THRESHOLD", "0.80"))
STALENESS_DAYS = float(os.getenv("HERMES_STALENESS_DAYS", "30"))
FRESH_HOURS = int(os.getenv("HERMES_FRESH_HOURS", "24"))
SCORING_DOWNGRADE_DAYS = float(os.getenv("HERMES_SCORING_DOWNGRADE_DAYS", "14"))
CONTRA_MODEL = os.getenv("HERMES_CONTRA_MODEL", "google/gemini-flash-1.5")

DESKS_DIR = Path("config/desks")

# Hermes 4 via OpenRouter — used for ALL desks.
# SOTA on RefusalBench: handles the full range of intelligence content without
# censorship or refusals. 131K context, strong structured output.
# Override with hermes-4-405b for higher accuracy at ~4x cost.
HERMES_MODEL = os.getenv("HERMES_MODEL", "nousresearch/hermes-4-70b")

# Venice fallback — only used when OPENROUTER_API_KEY is not configured.
VENICE_MODEL_FALLBACK = os.getenv("VENICE_MODEL_FALLBACK", "mistral-small-3-2-24b-instruct")

# Hermes 4 uses its own <tool_call> XML format via system prompt injection rather than
# the standard OpenAI tools API parameter. OpenRouter's routing layer has no endpoint
# for these models that supports the tools: API field, so we bypass it entirely by
# injecting tool schemas as a <tools> block in the system message and parsing
# <tool_call> tags from the response content.
_HERMES_MODELS = frozenset(
    {
        "nousresearch/hermes-4-70b",
        "nousresearch/hermes-4-405b",
    }
)

# Don't send tools: API field for Hermes (uses prompt injection) or venice-uncensored (ReAct)
_REACT_ONLY_MODELS = REACT_ONLY_MODELS | _HERMES_MODELS

MAX_ROUNDS = 8  # phase 1: up to ~5 search rounds; phase 2: verdict turn + follow-up buffer

# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


def _model_for_desk(_desk: str) -> tuple[str, str, str | None]:
    """Return (model_id, base_url, api_key).

    All desks use Hermes 4 via OpenRouter — no per-desk splits needed since
    Hermes 4 is uncensored and handles sensitive intelligence content natively.
    Falls back to Venice mistral-small-3-2-24b-instruct if OpenRouter is not configured.
    """
    if OPENROUTER_API_KEY:
        return HERMES_MODEL, OPENROUTER_BASE_URL, OPENROUTER_API_KEY

    # OpenRouter not configured — fall back to Venice
    return VENICE_MODEL_FALLBACK, VENICE_BASE_URL, VENICE_API_KEY or None


# ---------------------------------------------------------------------------
# Desk config loader
# ---------------------------------------------------------------------------


@dataclass
class DeskMeta:
    slug: str
    name: str
    collection: str
    tools: list[str]
    persona: str
    boost_collections: list[str] = field(default_factory=list)


def _load_desk_meta(slug: str) -> DeskMeta:
    path = DESKS_DIR / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No desk config found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    qdrant_cfg = raw.get("qdrant", {})
    collection = qdrant_cfg.get("collection", slug)
    tools = raw.get("tools", [])
    persona = (
        raw.get("briefing", {}).get("persona") or f"You are an intelligence analyst for the {raw.get('name', slug)}."
    )
    return DeskMeta(
        slug=slug,
        name=raw.get("name", slug),
        collection=collection,
        tools=tools,
        persona=persona.strip(),
        boost_collections=qdrant_cfg.get("boost_collections", []),
    )


def _list_all_desks() -> list[str]:
    return sorted(p.stem for p in DESKS_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Reliability tier transitions
# ---------------------------------------------------------------------------

_TIER_UP: dict[str, str] = {"?": "C", "": "C", "C": "B", "B": "A", "A": "A"}
_TIER_DOWN: dict[str, str] = {"A": "B", "B": "C", "C": "C", "?": "C", "": "C"}


def _upgrade_tier(current: str) -> str:
    return _TIER_UP.get(current, "C")


def _downgrade_tier(current: str) -> str:
    return _TIER_DOWN.get(current, "C")


# ---------------------------------------------------------------------------
# Tool schema filtering
# ---------------------------------------------------------------------------


def _filter_tool_schemas(desk_tools: list[str]) -> list[dict]:
    """Return TOOL_SCHEMAS filtered to desk-relevant tools.

    search_intel_kb is always included (internal KB check should always run first).
    search_web is always included as the default web fallback (DuckDuckGo, no quota).
    All other tools are gated by the desk's tools list.
    """
    always_included = {"search_intel_kb", "search_web"}
    allowed = always_included | set(desk_tools)
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in allowed]


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


def _parse_verdict(text: str) -> tuple[str, str, str, list[str]]:
    """Parse the structured verdict block from model output.

    Returns (verdict, confidence, reasoning, sources).
    Defaults to ("UNVERIFIED", "LOW", "", []) if the block is absent or malformed.
    """
    verdict = "UNVERIFIED"
    confidence = "LOW"
    reasoning = ""
    sources: list[str] = []

    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if "## CORROBORATION VERDICT" in stripped.upper():
            in_block = True
            continue
        if not in_block:
            continue
        if stripped.upper().startswith("VERDICT:"):
            v = stripped[len("VERDICT:") :].strip().upper()
            if v in ("CORROBORATED", "CONTRADICTED", "UNVERIFIED"):
                verdict = v
        elif stripped.upper().startswith("CONFIDENCE:"):
            c = stripped[len("CONFIDENCE:") :].strip().upper()
            if c in ("HIGH", "MODERATE", "LOW"):
                confidence = c
        elif stripped.upper().startswith("REASONING:"):
            reasoning = stripped[len("REASONING:") :].strip()
        elif stripped.upper().startswith("SOURCES:"):
            src_str = stripped[len("SOURCES:") :].strip()
            if src_str and src_str.lower() not in ("none", "none found", "n/a"):
                sources = [s.strip() for s in src_str.split(",") if s.strip()]

    return verdict, confidence, reasoning, sources


# ---------------------------------------------------------------------------
# Message sanitisation
# ---------------------------------------------------------------------------


def _sanitize_assistant_message(msg: dict) -> dict:
    """Strip the assistant message to only fields Venice accepts.

    Venice's own API response includes extra fields (e.g. 'refusal': null) that its
    Pydantic validator then rejects when the message is sent back in a subsequent round,
    producing a cascade of union-type validation errors. Whitelist only the three
    fields that are always valid in an assistant turn.
    """
    cleaned: dict = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        cleaned["tool_calls"] = msg["tool_calls"]
    return cleaned


# ---------------------------------------------------------------------------
# Hermes 4 tool calling helpers
# ---------------------------------------------------------------------------


def _build_hermes_tools_block(schemas: list[dict]) -> str:
    """Serialise filtered tool schemas as a <tools> block for Hermes 4 system prompts.

    Keeps the full OpenAI schema format (including the "type": "function" wrapper)
    as a JSON array inside <tools> tags — matching the format Hermes 4's chat template
    expects when tools are injected via the system message.
    """
    return "<tools>\n" + json.dumps(schemas) + "\n</tools>"


_KNOWN_TOOLS = frozenset(
    {
        "fetch_url",
        "search_intel_kb",
        "search_web",
        "search_tavily",
        "search_wikipedia",
        "search_arxiv",
        "search_semantic_scholar",
        "search_otx",
        "search_aleph",
    }
)


def _parse_hermes_tool_calls(content: str) -> list[tuple[str, str]]:
    """Extract (tool_name, query) pairs from Hermes 4 output.

    Hermes 4 via OpenRouter produces a variety of tool call formats depending on
    which version of the chat template OpenRouter is running. We handle all observed
    formats rather than relying on a single canonical one:

      1. <tool_call>{"name": "...", "arguments": {"query": "..."}}</tool_call>
      2. [{"name": "...", "arguments": {"query": "..."}}]  ← JSON array
      3. <search_tool query="..."/>  or  <search_tool query="..."></search_tool>
      4. search_intel_kb(query="...")  ← Python function syntax
      5. search_intel_kb: {"query": "..."}  ← JSON inline
      6. search_intel_kb: query text  ← plain text

    Priority: try all formats; deduplicate by (name, query).
    """
    calls: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(name: str, query: str) -> None:
        name = name.strip()
        query = query.strip()
        if name in _KNOWN_TOOLS and query and (name, query) not in seen:
            seen.add((name, query))
            calls.append((name, query))

    # 1. Canonical <tool_call> JSON envelope
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        try:
            data = json.loads(match.group(1).strip())
            name = data.get("name", "")
            args = data.get("arguments", {})
            query = (args.get("query") or args.get("url") or "") if isinstance(args, dict) else ""
            _add(name, query)
        except (json.JSONDecodeError, AttributeError):
            pass  # malformed JSON in <tool_call> tag — skip this match, try next

    # 2. JSON array: [{"name": "...", "arguments": {"query": "..."}}]
    # Strip <tools>...</tools> wrapper if present before trying JSON parse
    stripped = re.sub(r"</?tools>", "", content, flags=re.IGNORECASE).strip()
    for match in re.finditer(r"\[\s*\{.*?\}\s*\]", stripped, re.DOTALL):
        try:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                for item in items:
                    name = item.get("name", "")
                    args = item.get("arguments", {})
                    query = (args.get("query") or args.get("url") or "") if isinstance(args, dict) else ""
                    _add(name, query)
        except (json.JSONDecodeError, TypeError):
            pass  # regex matched something that looks like a JSON array but isn't — skip

    # 3a. XML attribute style: <search_intel_kb query="..."/>
    for match in re.finditer(
        r'<(search_\w+)\s+query=["\']([^"\']+)["\']',
        content,
        re.IGNORECASE,
    ):
        _add(match.group(1).lower(), match.group(2))

    # 3b. XML child-element style with <query> tag: <name>search_intel_kb</name><arguments><query>...</query>
    for match in re.finditer(
        r"<name>\s*(search_\w+)\s*</name>.*?<query>\s*(.*?)\s*</query>",
        content,
        re.DOTALL | re.IGNORECASE,
    ):
        _add(match.group(1).lower(), match.group(2))

    # 3c. XML child-element style with JSON <arguments>: <name>search_intel_kb</name><arguments>{"query":"..."}
    for match in re.finditer(
        r"<name>\s*(search_\w+)\s*</name>\s*<arguments>\s*(\{[^<]+\})\s*</arguments>",
        content,
        re.DOTALL | re.IGNORECASE,
    ):
        try:
            args = json.loads(match.group(2).strip())
            query = (args.get("query") or args.get("url") or "") if isinstance(args, dict) else ""
            _add(match.group(1).lower(), query)
        except (json.JSONDecodeError, AttributeError):
            pass  # <arguments> block matched but contained invalid JSON — skip this match

    # 4. Python function call: search_intel_kb(query="...") or search_intel_kb("...")
    for match in re.finditer(
        r'\b(search_\w+)\s*\(\s*(?:query\s*=\s*)?["\']([^"\']+)["\']',
        content,
    ):
        _add(match.group(1), match.group(2))

    # 5. Inline JSON: search_intel_kb: {"query": "..."}
    for match in re.finditer(
        r'\b(search_\w+)\s*:\s*\{[^}]*["\']query["\']\s*:\s*["\']([^"\']+)["\']',
        content,
    ):
        _add(match.group(1), match.group(2))

    # 6. Plain text: search_intel_kb: some query text (line-by-line, last resort)
    for line in content.splitlines():
        stripped_line = line.strip()
        for name in _KNOWN_TOOLS:
            prefix = name + ":"
            if stripped_line.lower().startswith(prefix):
                rest = stripped_line[len(prefix) :].strip()
                if rest and not rest.startswith(("{", "[")):
                    _add(name, rest)
                break

    return calls


# ---------------------------------------------------------------------------
# Corroboration research loop — OpenAI-compat (Venice / OpenRouter)
# ---------------------------------------------------------------------------


async def _run_corroboration_loop_openai_compat(
    desk: DeskMeta,
    point_id: int | str,
    topic: str,
    source: str,
    current_tier: str,
    excerpt: str,
    http: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    extra_headers: dict | None = None,
    model_override: str | None = None,
) -> str:
    model = model_override or _model_for_desk(desk.slug)[0]
    _registry = get_tool_registry(desk.slug)  # desk-aware: search_intel_kb includes boost collections
    filtered_schemas = _filter_tool_schemas(desk.tools)
    tool_names = " | ".join(s["function"]["name"] for s in filtered_schemas)
    is_hermes = model in _HERMES_MODELS

    # ---------------------------------------------------------------------------
    # Two-phase prompting strategy
    #
    # Phase 1 — search only. The system prompt and initial user message contain NO
    # verdict format. Asking for both searches and a structured verdict in the same
    # turn causes Hermes 4 to skip directly to the verdict (the training data has
    # many "think → answer" examples; the verdict block acts as a strong anchor).
    # By withholding the verdict format until *after* tool results arrive we force
    # the search phase to happen first.
    #
    # Phase 2 — verdict. Injected as a new user turn after at least one tool round
    # has completed (or after a nudge if the model skips tools entirely).
    # ---------------------------------------------------------------------------

    system = (
        f"{desk.persona}\n\n"
        "Your current assignment is EVIDENCE GATHERING for claim verification. "
        "You will receive an intelligence excerpt. Your job right now is to SEARCH for evidence — "
        "do not write any verdict or conclusion yet.\n\n"
        f"Available tools: {tool_names}\n\n"
        "To call a tool, output EXACTLY this XML format — nothing else:\n"
        '<tool_call>{"name": "search_intel_kb", "arguments": {"query": "your search query here"}}</tool_call>\n\n'
        "Search strategy:\n"
        "1. Call search_intel_kb first with the core subject of the claim.\n"
        "2. Then call 1-3 additional tools for independent external evidence.\n\n"
        "Do NOT write prose, analysis, or verdicts. Output ONLY <tool_call> tags."
    )

    # Hermes 4: inject tool schemas as a <tools> block in the system message.
    # OpenRouter has no endpoint that supports the tools: API field for these models,
    # so we bypass it entirely — inject schemas here, parse <tool_call> tags from
    # response content. Temperature 0.6 is the Hermes 4 recommended default.
    if is_hermes:
        system = system + "\n\n" + _build_hermes_tools_block(filtered_schemas)
    else:
        system = (
            system + "\n\nUse ReAct format for tool calls:\n"
            "SEARCH_INTEL_KB: <query>\nSEARCH_WEB: <query>\nSEARCH_WIKIPEDIA: <query>\n"
            "SEARCH_ARXIV: <query>\nSEARCH_DUCKDUCKGO: <query>"
        )

    # Verdict request injected as a new user turn after tool results are in.
    verdict_request = (
        "Good. Now based on all the search results above, write your corroboration assessment.\n\n"
        "## CORROBORATION VERDICT\n"
        "VERDICT: <CORROBORATED|CONTRADICTED|UNVERIFIED>\n"
        "CONFIDENCE: <HIGH|MODERATE|LOW>\n"
        "REASONING: <one paragraph — what evidence was found, what it confirmed or contradicted>\n"
        "SOURCES: <comma-separated URLs, or 'none found'>\n\n"
        "Rules:\n"
        "- CORROBORATED: independent sources confirm the claim's core facts\n"
        "- CONTRADICTED: sources directly dispute the claim\n"
        "- UNVERIFIED: searches returned no relevant independent evidence\n"
        "- CONFIDENCE reflects how strong and direct the evidence is"
    )

    phase1_user = (
        f"Search for evidence about this intelligence excerpt:\n\n"
        f"Topic: {topic}\n"
        f"Source: {source}\n"
        f"Current reliability tier: {current_tier or '?'}\n\n"
        f"--- EXCERPT ---\n{excerpt[:1500]}\n--- END EXCERPT ---\n\n"
        "Call search_intel_kb now, then use additional tools. Do NOT write a verdict yet."
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": phase1_user},
    ]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    tool_call_count = 0
    search_rounds = 0  # rounds in which tool calls were made
    tool_nudge_sent = False
    verdict_requested = False
    MAX_SEARCH_ROUNDS = 3  # cap search depth; force verdict after this many search rounds

    logger.info(
        "Corroborating point %s | desk: %s | topic: %.80s",
        str(point_id)[:8],
        desk.slug,
        topic or "(no topic)",
    )

    for round_num in range(MAX_ROUNDS):
        # Hermes 4 recommended: temp=0.6, top_p=0.95, top_k=20
        temperature = 0.6 if is_hermes else 0.2
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": temperature,
        }
        if is_hermes:
            payload["top_p"] = 0.95
        if model not in _REACT_ONLY_MODELS:
            payload["tools"] = filtered_schemas
            payload["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                resp = await http.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90.0,
                )
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    wait = 35 * (attempt + 1)
                    logger.warning("429 rate-limited — waiting %ds (attempt %d/3)", wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                logger.error(
                    "API HTTP %d on round %d — body: %s",
                    e.response.status_code,
                    round_num,
                    e.response.text[:400],
                )
                raise
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as e:
                if attempt < 2:
                    wait = 15 * (attempt + 1)
                    logger.warning(
                        "Venice connection error on round %d — retrying in %ds (attempt %d/3): %s",
                        round_num,
                        wait,
                        attempt + 1,
                        e,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]
        messages.append(_sanitize_assistant_message(message))

        tool_calls = message.get("tool_calls") or []
        content = message.get("content", "") or ""

        # --- Native OpenAI tool_calls (non-Hermes models) ---
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                query = fn_args.get("query") or fn_args.get("url") or ""
                logger.info("  → tool call: %s(%r)", fn_name, query)
                tool_fn = _registry.get(fn_name)
                result = await tool_fn(query, http) if tool_fn else f"Unknown tool: {fn_name}"
                logger.info("     result: %.120s", result.replace("\n", " "))
                tool_call_count += 1
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
            search_rounds += 1
            if search_rounds >= MAX_SEARCH_ROUNDS and not verdict_requested:
                messages.append({"role": "user", "content": verdict_request})
                verdict_requested = True
                logger.info(
                    "Point %s: %d search round(s) done, injecting verdict request",
                    str(point_id)[:8],
                    search_rounds,
                )
            continue

        # --- Hermes 4 native <tool_call> tag format ---
        if is_hermes:
            hermes_calls = _parse_hermes_tool_calls(content)
            if hermes_calls:
                response_parts = []
                for fn_name, query in hermes_calls:
                    logger.info("  → tool call: %s(%r)", fn_name, query)
                    tool_fn = _registry.get(fn_name)
                    result = await tool_fn(query, http) if tool_fn else f"Unknown tool: {fn_name}"
                    logger.info("     result: %.120s", result.replace("\n", " "))
                    tool_call_count += 1
                    response_parts.append(
                        f"<tool_response>\n{json.dumps({'tool': fn_name, 'result': result})}\n</tool_response>"
                    )
                search_rounds += 1
                if search_rounds >= MAX_SEARCH_ROUNDS:
                    # Enough evidence gathered — inject verdict request alongside tool results
                    response_parts.append(verdict_request)
                    messages.append({"role": "user", "content": "\n\n".join(response_parts)})
                    verdict_requested = True
                    logger.info(
                        "Point %s: %d search round(s) done, injecting verdict request",
                        str(point_id)[:8],
                        search_rounds,
                    )
                else:
                    messages.append({"role": "user", "content": "\n\n".join(response_parts)})
                continue

        # --- ReAct fallback — venice-uncensored and other text-only models ---
        react_calls = _parse_react(content)
        if react_calls:
            tool_results = []
            for fn_name, query in react_calls:
                logger.info("  → tool call: %s(%r)", fn_name, query)
                tool_fn = _registry.get(fn_name)
                result = await tool_fn(query, http) if tool_fn else f"Unknown tool: {fn_name}"
                logger.info("     result: %.120s", result.replace("\n", " "))
                tool_call_count += 1
                tool_results.append(f"[{fn_name}: {query}]\n{result}")
            search_rounds += 1
            if search_rounds >= MAX_SEARCH_ROUNDS and not verdict_requested:
                tool_results.append(verdict_request)
                verdict_requested = True
                logger.info(
                    "Point %s: %d search round(s) done, injecting verdict request",
                    str(point_id)[:8],
                    search_rounds,
                )
            messages.append({"role": "user", "content": "\n\n".join(tool_results)})
            continue

        # --- No tool calls this round ---
        if verdict_requested:
            # Model responded to the verdict request without more tool calls — done
            logger.info(
                "Corroboration complete: %d tool call(s) across %d round(s)",
                tool_call_count,
                round_num + 1,
            )
            return content

        if tool_call_count > 0:
            # At least one search was done — now ask for the verdict
            logger.info(
                "Point %s: %d tool call(s) done, requesting verdict",
                str(point_id)[:8],
                tool_call_count,
            )
            messages.append({"role": "user", "content": verdict_request})
            verdict_requested = True
            continue

        if not tool_nudge_sent:
            # No tools used yet — nudge once before falling back to no-tool verdict
            logger.warning(
                "Point %s: no tool calls in round %d — nudging",
                str(point_id)[:8],
                round_num + 1,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have not searched yet. "
                        f"Call search_intel_kb now with a query about: {topic or excerpt[:100]}. "
                        "Do not write a verdict until you have searched."
                    ),
                }
            )
            tool_nudge_sent = True
            continue

        # Nudge already sent and still no tools — request verdict on whatever reasoning exists
        logger.warning(
            "Point %s: skipped tool use entirely after nudge — requesting verdict anyway",
            str(point_id)[:8],
        )
        messages.append({"role": "user", "content": verdict_request})
        verdict_requested = True
        continue

    # Max rounds hit. If we used tools but never got a verdict, make one final call.
    if tool_call_count > 0 and not verdict_requested:
        logger.warning(
            "Point %s: hit max rounds after %d tool call(s) — making final verdict call",
            str(point_id)[:8],
            tool_call_count,
        )
        messages.append({"role": "user", "content": verdict_request})
        temperature = 0.6 if is_hermes else 0.2
        payload = {"model": model, "messages": messages, "max_tokens": 1500, "temperature": temperature}
        if is_hermes:
            payload["top_p"] = 0.95
        try:
            resp = await http.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content", "") or ""
            if content:
                return content
        except Exception as e:
            logger.error("Final verdict call failed for point %s: %s", str(point_id)[:8], e)

    logger.warning("Hit max rounds for point %s — returning best available content", str(point_id)[:8])
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


# ---------------------------------------------------------------------------
# Corroboration loop — Gemini fallback
# ---------------------------------------------------------------------------


async def _run_corroboration_loop_gemini(
    desk: DeskMeta,
    point_id: int | str,
    topic: str,
    source: str,
    current_tier: str,
    excerpt: str,
    http: httpx.AsyncClient,
) -> str:
    from google import genai
    from google.genai import types

    gemini = genai.Client(api_key=GEMINI_API_KEY)
    _registry = get_tool_registry(desk.slug)  # desk-aware: search_intel_kb includes boost collections

    # Build Gemini-format tool declarations from filtered schemas
    filtered_schemas = _filter_tool_schemas(desk.tools)
    declarations = []
    for schema in filtered_schemas:
        fn = schema["function"]
        declarations.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"query": types.Schema(type="STRING")},
                    required=["query"],
                ),
            )
        )
    tools = [types.Tool(function_declarations=declarations)]

    tool_names = " | ".join(s["function"]["name"] for s in filtered_schemas)
    system = (
        f"{desk.persona}\n\n"
        "Your assignment is CLAIM VERIFICATION. Determine if the excerpt can be CORROBORATED, "
        "CONTRADICTED, or remains UNVERIFIED.\n"
        "Call search_intel_kb first, then use desk-appropriate tools "
        f"({tool_names}). Make 2-4 tool calls max.\n"
        "End your response with:\n"
        "## CORROBORATION VERDICT\n"
        "VERDICT: CORROBORATED|CONTRADICTED|UNVERIFIED\n"
        "CONFIDENCE: HIGH|MODERATE|LOW\n"
        "REASONING: <paragraph>\n"
        "SOURCES: <URLs or 'none found'>"
    )
    user_content = (
        f"Verify this intelligence excerpt:\n\n"
        f"Topic: {topic}\nSource: {source}\nCurrent reliability tier: {current_tier or '?'}\n\n"
        f"--- EXCERPT ---\n{excerpt[:1500]}\n--- END EXCERPT ---"
    )

    contents = [types.Content(role="user", parts=[types.Part(text=f"{system}\n\n{user_content}")])]
    logger.info("Corroborating point %s via Gemini (desk: %s)", str(point_id)[:8], desk.slug)

    for round_num in range(MAX_ROUNDS):
        response = await asyncio.to_thread(
            gemini.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(tools=tools),
        )
        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [p for p in candidate.content.parts if p.function_call]
        if not function_calls:
            text_parts = [p.text for p in candidate.content.parts if p.text]
            result = "\n".join(text_parts)
            logger.info("Corroboration complete after %d rounds", round_num + 1)
            return result

        response_parts = []
        for part in function_calls:
            call = part.function_call
            _args = dict(call.args) if call.args else {}
            query = _args.get("query") or _args.get("url") or ""
            logger.info("Tool: %s(%r)", call.name, query)
            tool_fn = _registry.get(call.name)
            result_text = await tool_fn(query, http) if tool_fn else f"Unknown tool: {call.name}"
            response_parts.append(
                types.Part(function_response=types.FunctionResponse(name=call.name, response={"result": result_text}))
            )
        contents.append(types.Content(role="user", parts=response_parts))

    logger.warning("Hit max rounds for point %s", str(point_id)[:8])
    text_parts = [p.text for p in contents[-1].parts if hasattr(p, "text") and p.text]
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


async def _run_corroboration_loop(
    desk: DeskMeta,
    point_id: int | str,
    topic: str,
    source: str,
    current_tier: str,
    excerpt: str,
    http: httpx.AsyncClient,
) -> str:
    """Try each configured provider in order; cascade to the next on any failure."""
    errors: list[str] = []

    if OPENROUTER_API_KEY:
        try:
            return await _run_corroboration_loop_openai_compat(
                desk,
                point_id,
                topic,
                source,
                current_tier,
                excerpt,
                http,
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
                extra_headers={"HTTP-Referer": "https://osia.dev", "X-Title": "OSIA Hermes Worker"},
                model_override=HERMES_MODEL,
            )
        except Exception as exc:
            logger.warning("OpenRouter failed for desk '%s' — trying Venice: %s", desk.slug, exc)
            errors.append(f"OpenRouter: {exc}")

    if VENICE_API_KEY:
        try:
            return await _run_corroboration_loop_openai_compat(
                desk,
                point_id,
                topic,
                source,
                current_tier,
                excerpt,
                http,
                base_url=VENICE_BASE_URL,
                api_key=VENICE_API_KEY,
                model_override=VENICE_MODEL_FALLBACK,
            )
        except Exception as exc:
            logger.warning("Venice failed for desk '%s' — trying Gemini: %s", desk.slug, exc)
            errors.append(f"Venice: {exc}")

    if GEMINI_API_KEY:
        try:
            return await _run_corroboration_loop_gemini(desk, point_id, topic, source, current_tier, excerpt, http)
        except Exception as exc:
            logger.warning("Gemini failed for desk '%s': %s", desk.slug, exc)
            errors.append(f"Gemini: {exc}")

    raise RuntimeError(f"All providers failed for desk '{desk.slug}': {'; '.join(errors) or 'no API keys configured'}")


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


async def _scroll_low_confidence(
    client: AsyncQdrantClient,
    collection: str,
    scan_limit: int,
) -> list[tuple[int | str, dict]]:
    """Scroll the collection for points where reliability_tier != 'A'.

    Returns list of (point_id, payload) tuples. Point IDs keep their native type
    (int for integer IDs, str for UUID IDs) so they can be passed directly back to
    set_payload without re-serialisation issues.
    """
    try:
        results, _offset = await client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_models.Filter(
                must_not=[
                    qdrant_models.FieldCondition(
                        key="reliability_tier",
                        # Exclude all INTSUM tiers (A, A+, A-) — those are handled
                        # by the KB scoring pass, not the research corroboration loop.
                        match=qdrant_models.MatchAny(any=["A", "A+", "A-"]),
                    )
                ]
            ),
            limit=scan_limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.error("Scroll failed for collection '%s': %s", collection, e)
        return []

    return [(p.id, dict(p.payload or {})) for p in results]


async def _patch_payload(
    client: AsyncQdrantClient,
    collection: str,
    point_id: int | str,
    patches: dict,
    dry_run: bool,
) -> None:
    """Set payload fields on an existing Qdrant point without touching the vector."""
    pid_log = str(point_id)[:8]
    if dry_run:
        logger.info("[DRY-RUN] Would patch point %s in '%s': %s", pid_log, collection, patches)
        return
    try:
        await client.set_payload(
            collection_name=collection,
            payload=patches,
            points=[point_id],
        )
        logger.info("Patched point %s in '%s'", pid_log, collection)
    except Exception as e:
        logger.error("Failed to patch point %s in '%s': %s", pid_log, collection, e)


# ---------------------------------------------------------------------------
# Distributed desk leasing (Redis)
# ---------------------------------------------------------------------------


def _get_redis():
    """Return a connected Redis client, or None if Redis is unavailable.

    Uses a 3-second connect timeout so a missing Redis doesn't stall startup.
    When None is returned, all lease functions become no-ops and the worker
    runs in single-node mode (no distributed coordination).
    """
    import redis as _redis

    try:
        r = _redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=3)
        r.ping()
        return r
    except Exception as e:
        logger.warning("Redis unavailable (%s) — running without distributed desk leasing (single-node mode)", e)
        return None


def _try_claim_desk(r, slug: str, worker_id: str) -> bool:
    """Atomically claim a desk lease. Returns True if this worker now holds the lease.

    Uses SET NX so only one worker can claim a given desk. The TTL ensures that
    leases from crashed workers expire automatically after HERMES_LEASE_TTL seconds.
    If r is None (Redis unavailable) always returns True so single-node runs
    continue unimpeded.
    """
    if r is None:
        return True
    return bool(r.set(f"{_LEASE_PREFIX}{slug}", worker_id, nx=True, ex=HERMES_LEASE_TTL))


def _release_desk(r, slug: str) -> None:
    """Release a desk lease. Safe to call even if the lease has already expired."""
    if r is None:
        return
    r.delete(f"{_LEASE_PREFIX}{slug}")


def _clear_all_leases(r) -> int:
    """Delete all active Hermes desk leases. Returns count deleted."""
    if r is None:
        return 0
    keys = r.keys(f"{_LEASE_PREFIX}*")
    if not keys:
        return 0
    return r.delete(*keys)


def _list_leases(r) -> list[tuple[str, str, int]]:
    """Return active leases as [(desk_slug, worker_id, ttl_seconds)]."""
    if r is None:
        return []
    keys = r.keys(f"{_LEASE_PREFIX}*")
    leases = []
    for key in keys:
        slug = key.removeprefix(_LEASE_PREFIX)
        worker_id = r.get(key) or "unknown"
        ttl = r.ttl(key)
        leases.append((slug, worker_id, ttl))
    return sorted(leases)


# ---------------------------------------------------------------------------
# Qdrant quality maintenance
# ---------------------------------------------------------------------------


async def _cleanup_empty_points(
    client: AsyncQdrantClient,
    collection: str,
    dry_run: bool,
) -> int:
    """Delete points whose text is shorter than CLEANUP_MIN_TEXT_LEN characters."""
    try:
        results, _ = await client.scroll(
            collection_name=collection,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.error("Cleanup (empty) scroll failed for '%s': %s", collection, e)
        return 0

    to_delete = [p.id for p in results if len(((p.payload or {}).get("text") or "").strip()) < CLEANUP_MIN_TEXT_LEN]
    if not to_delete:
        return 0

    if dry_run:
        logger.info("[DRY-RUN] Would delete %d empty/stub points from '%s'", len(to_delete), collection)
        return len(to_delete)

    try:
        await client.delete(
            collection_name=collection,
            points_selector=qdrant_models.PointIdsList(points=to_delete),
        )
        logger.info("Cleanup: deleted %d empty/stub points from '%s'", len(to_delete), collection)
    except Exception as e:
        logger.error("Cleanup delete (empty) failed for '%s': %s", collection, e)
        return 0
    return len(to_delete)


async def _cleanup_stale_contradicted(
    client: AsyncQdrantClient,
    collection: str,
    dry_run: bool,
) -> int:
    """Delete tier-C contradiction-flagged points older than CLEANUP_STALE_DAYS."""
    cutoff = time.time() - (CLEANUP_STALE_DAYS * 86400)
    try:
        results, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="contradiction_flag",
                        match=qdrant_models.MatchValue(value=True),
                    ),
                    qdrant_models.FieldCondition(
                        key="reliability_tier",
                        match=qdrant_models.MatchValue(value="C"),
                    ),
                ]
            ),
            limit=200,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.error("Cleanup (contradicted) scroll failed for '%s': %s", collection, e)
        return 0

    to_delete = []
    for p in results:
        payload = p.payload or {}
        ts: float = payload.get("ingested_at_unix") or 0
        if not ts:
            checked = payload.get("corroboration_checked_at", "")
            if checked:
                try:
                    ts = datetime.fromisoformat(checked).timestamp()
                except (ValueError, OSError):
                    pass  # malformed timestamp — treat as unknown age, skip
        if ts and ts < cutoff:
            to_delete.append(p.id)

    if not to_delete:
        return 0

    if dry_run:
        logger.info("[DRY-RUN] Would purge %d stale contradicted points from '%s'", len(to_delete), collection)
        return len(to_delete)

    try:
        await client.delete(
            collection_name=collection,
            points_selector=qdrant_models.PointIdsList(points=to_delete),
        )
        logger.info("Cleanup: purged %d stale contradicted points from '%s'", len(to_delete), collection)
    except Exception as e:
        logger.error("Cleanup delete (contradicted) failed for '%s': %s", collection, e)
        return 0
    return len(to_delete)


async def _cleanup_permanently_unverifiable(
    client: AsyncQdrantClient,
    collection: str,
    dry_run: bool,
) -> int:
    """Delete low-tier points that have been checked CLEANUP_UNVERIFIED_CHECKS+ times,
    always returned UNVERIFIED, and are older than CLEANUP_UNVERIFIED_AGE_DAYS.

    These are dead-weight entries that have resisted corroboration repeatedly and no
    longer add signal to RAG retrieval.
    """
    cutoff = time.time() - (CLEANUP_UNVERIFIED_AGE_DAYS * 86400)
    try:
        results, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="corroboration_verdict",
                        match=qdrant_models.MatchValue(value="UNVERIFIED"),
                    ),
                    qdrant_models.FieldCondition(
                        key="corroboration_check_count",
                        range=qdrant_models.Range(gte=CLEANUP_UNVERIFIED_CHECKS),
                    ),
                ],
                must_not=[
                    qdrant_models.FieldCondition(
                        key="reliability_tier",
                        match=qdrant_models.MatchAny(any=["A", "B"]),
                    ),
                ],
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.error("Cleanup (unverifiable) scroll failed for '%s': %s", collection, e)
        return 0

    to_delete = []
    for p in results:
        payload = p.payload or {}
        ts: float = payload.get("ingested_at_unix") or 0
        if not ts:
            checked = payload.get("corroboration_checked_at", "")
            if checked:
                try:
                    ts = datetime.fromisoformat(checked).timestamp()
                except (ValueError, OSError):
                    pass
        if ts and ts < cutoff:
            to_delete.append(p.id)

    if not to_delete:
        return 0

    if dry_run:
        logger.info(
            "[DRY-RUN] Would purge %d permanently unverifiable points from '%s'",
            len(to_delete),
            collection,
        )
        return len(to_delete)

    try:
        await client.delete(
            collection_name=collection,
            points_selector=qdrant_models.PointIdsList(points=to_delete),
        )
        logger.info("Cleanup: purged %d permanently unverifiable points from '%s'", len(to_delete), collection)
    except Exception as e:
        logger.error("Cleanup delete (unverifiable) failed for '%s': %s", collection, e)
        return 0
    return len(to_delete)


async def _run_cleanup_pass(
    client: AsyncQdrantClient,
    collection: str,
    dry_run: bool,
) -> None:
    """Run all quality maintenance passes on a collection and log a summary."""
    empty = await _cleanup_empty_points(client, collection, dry_run)
    stale = await _cleanup_stale_contradicted(client, collection, dry_run)
    unverifiable = await _cleanup_permanently_unverifiable(client, collection, dry_run)
    if empty or stale or unverifiable:
        logger.info(
            "Cleanup '%s': empty=%d stale_contradicted=%d permanently_unverifiable=%d",
            collection,
            empty,
            stale,
            unverifiable,
        )


# ---------------------------------------------------------------------------
# RAG contamination remediation passes (tier-A INTSUM chunks)
# ---------------------------------------------------------------------------


def _extract_vector(record_vector) -> list[float] | None:
    """Normalise the .vector field from a Qdrant scroll record.

    Collections use a single anonymous vector, so record.vector is normally a
    list[float].  Named-vector collections return a dict; we take the first value.
    Returns None if the vector is absent or empty.
    """
    if record_vector is None:
        return None
    if isinstance(record_vector, dict):
        values = list(record_vector.values())
        return values[0] if values else None
    return record_vector if record_vector else None


async def _score_chunk_against_kb(
    chunk_vector: list[float],
    boost_collections: list[str],
    client: AsyncQdrantClient,
) -> tuple[float, list[str]]:
    """Search every boost collection with the chunk's own vector.

    Returns (score 0.0–1.0, list of collection names that returned a hit).
    Score = n_collections_with_hit / n_boost_collections.
    Errors on individual collections are silently swallowed (collection may not exist).
    """

    async def _probe(col: str) -> str | None:
        try:
            hits = await client.search(
                collection_name=col,
                query_vector=chunk_vector,
                limit=1,
                score_threshold=KB_CORROBORATION_THRESHOLD,
                with_vectors=False,
                with_payload=False,
            )
            return col if hits else None
        except Exception:
            return None  # collection absent or temporarily unavailable

    results = await asyncio.gather(*[_probe(col) for col in boost_collections])
    matching = [col for col in results if col is not None]
    score = len(matching) / len(boost_collections) if boost_collections else 0.0
    return score, matching


async def _corroboration_scoring_pass(
    desk: "DeskMeta",
    client: AsyncQdrantClient,
    dry_run: bool,
) -> int:
    """Vector-based KB corroboration scoring for tier-A (and A-) INTSUM chunks.

    Priority 1 — fresh chunks ingested in the last FRESH_HOURS hours.
    Priority 2 — rolling backfill of unchecked / stale chunks.

    For each chunk:
    - Searches every boost collection with the chunk's own embedding vector
    - If any KB hit scores ≥ KB_CORROBORATION_THRESHOLD → tier "A+"
    - If no KB hit and chunk is > SCORING_DOWNGRADE_DAYS old → tier "A-"
    - Writes hermes_corroboration_score, hermes_kb_sources, hermes_reviewed_at

    Returns count of chunks processed.
    """
    if not desk.boost_collections:
        logger.debug("Desk '%s' has no boost collections — skipping scoring pass", desk.slug)
        return 0

    now_ts = time.time()
    fresh_cutoff = int(now_ts - FRESH_HOURS * 3600)
    cooldown_cutoff = int(now_ts - COOLDOWN_DAYS * 86400)

    # Phase 1: fresh chunks (last FRESH_HOURS) — always re-score regardless of hermes_reviewed_at
    try:
        fresh_records, _ = await client.scroll(
            collection_name=desk.collection,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="reliability_tier",
                        match=qdrant_models.MatchAny(any=["A", "A-"]),
                    ),
                    qdrant_models.FieldCondition(
                        key="ingested_at_unix",
                        range=qdrant_models.Range(gte=fresh_cutoff),
                    ),
                ]
            ),
            limit=SCORING_BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
        )
    except Exception as e:
        logger.error("Scoring: fresh scroll failed for '%s': %s", desk.collection, e)
        fresh_records = []

    # Phase 2: backfill — tier A/A- not recently scored
    remaining_budget = SCORING_BATCH_SIZE - len(fresh_records)
    backfill_records: list = []
    if remaining_budget > 0:
        try:
            all_a, _ = await client.scroll(
                collection_name=desk.collection,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="reliability_tier",
                            match=qdrant_models.MatchAny(any=["A", "A-"]),
                        ),
                    ]
                ),
                limit=remaining_budget * 5,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as e:
            logger.error("Scoring: backfill scroll failed for '%s': %s", desk.collection, e)
            all_a = []

        fresh_ids = {r.id for r in fresh_records}
        for r in all_a:
            if r.id in fresh_ids:
                continue
            reviewed = (r.payload or {}).get("hermes_reviewed_at")
            if reviewed and int(reviewed) > cooldown_cutoff:
                continue  # scored recently, skip
            backfill_records.append(r)
            if len(backfill_records) >= remaining_budget:
                break

    candidates = list(fresh_records) + backfill_records
    if not candidates:
        return 0

    logger.info(
        "Scoring pass '%s': %d candidates (fresh=%d backfill=%d)",
        desk.slug,
        len(candidates),
        len(fresh_records),
        len(backfill_records),
    )

    processed = 0
    for record in candidates:
        vector = _extract_vector(record.vector)
        if not vector:
            continue

        payload = record.payload or {}
        ingested_ts = payload.get("ingested_at_unix") or 0
        age_days = (now_ts - ingested_ts) / 86400 if ingested_ts else 999.0
        current_tier = payload.get("reliability_tier", "A")

        score, kb_sources = await _score_chunk_against_kb(vector, desk.boost_collections, client)

        if score > 0:
            new_tier = "A+"
        elif age_days > SCORING_DOWNGRADE_DAYS:
            new_tier = "A-"
        else:
            new_tier = current_tier  # too fresh to demote yet

        patch: dict = {
            "hermes_corroboration_score": round(score, 3),
            "hermes_kb_sources": kb_sources,
            "hermes_reviewed_at": int(now_ts),
        }
        if new_tier != current_tier:
            patch["reliability_tier"] = new_tier

        if dry_run:
            logger.info(
                "[DRY-RUN] Scoring point %s: score=%.3f tier %s→%s kb=%s",
                str(record.id)[:8],
                score,
                current_tier,
                new_tier,
                kb_sources,
            )
        else:
            await _patch_payload(client, desk.collection, record.id, patch, dry_run=False)
        processed += 1

    logger.info("Scoring pass '%s': %d chunks processed", desk.slug, processed)
    return processed


async def _echo_chamber_pass(
    desk: "DeskMeta",
    client: AsyncQdrantClient,
    dry_run: bool,
) -> int:
    """Flag fresh INTSUM chunks whose top-10 similar neighbours are mostly OSIA-internal.

    An echo risk chunk is one where > ECHO_RISK_THRESHOLD of the 10 most similar
    points in the same collection came from signal: or research_worker sources,
    meaning the INTSUM was likely synthesised from prior OSIA outputs rather than
    primary KB evidence.

    Writes hermes_echo_risk: bool.  Only processes fresh chunks (last FRESH_HOURS)
    that have not already been assessed.

    Returns count of chunks assessed.
    """
    now_ts = time.time()
    fresh_cutoff = int(now_ts - FRESH_HOURS * 3600)

    try:
        fresh_records, _ = await client.scroll(
            collection_name=desk.collection,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="reliability_tier",
                        match=qdrant_models.MatchAny(any=["A", "A+", "A-"]),
                    ),
                    qdrant_models.FieldCondition(
                        key="ingested_at_unix",
                        range=qdrant_models.Range(gte=fresh_cutoff),
                    ),
                ]
            ),
            limit=SCORING_BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
        )
    except Exception as e:
        logger.error("Echo: fresh scroll failed for '%s': %s", desk.collection, e)
        return 0

    processed = 0
    for record in fresh_records:
        payload = record.payload or {}
        if payload.get("hermes_echo_risk") is not None:
            continue  # already assessed

        vector = _extract_vector(record.vector)
        if not vector:
            continue

        try:
            neighbours = await client.search(
                collection_name=desk.collection,
                query_vector=vector,
                limit=10,
                with_payload=True,
                with_vectors=False,
                query_filter=qdrant_models.Filter(must_not=[qdrant_models.HasIdCondition(has_id=[record.id])]),
            )
        except Exception as e:
            logger.warning("Echo: neighbour search failed for '%s': %s", desk.collection, e)
            continue

        if not neighbours:
            continue

        def _is_internal(src: str) -> bool:
            return src.startswith("signal:") or src in ("research_worker", "osia")

        n_internal = sum(1 for n in neighbours if _is_internal((n.payload or {}).get("source", "")))
        echo_risk = (n_internal / len(neighbours)) > ECHO_RISK_THRESHOLD

        if echo_risk:
            logger.info(
                "Echo risk: point %s — %.0f%% OSIA-internal context",
                str(record.id)[:8],
                (n_internal / len(neighbours)) * 100,
            )

        if not dry_run:
            await _patch_payload(client, desk.collection, record.id, {"hermes_echo_risk": echo_risk}, dry_run=False)
        else:
            logger.info("[DRY-RUN] Echo risk for %s: %s", str(record.id)[:8], echo_risk)

        processed += 1

    return processed


async def _llm_contradiction_check(
    text_a: str,
    text_b: str,
    http: httpx.AsyncClient,
) -> tuple[bool, str]:
    """Binary LLM contradiction check using cheapest available model.

    Returns (contradicts: bool, reason: str).
    Uses CONTRA_MODEL via OpenRouter; falls back to Venice mistral-small.
    """
    model = CONTRA_MODEL
    base_url = OPENROUTER_BASE_URL
    api_key = OPENROUTER_API_KEY
    extra_headers: dict = {"HTTP-Referer": "https://osia.dev", "X-Title": "OSIA Hermes Worker"}

    if not api_key:
        if not VENICE_API_KEY:
            return False, "no API key configured"
        base_url = VENICE_BASE_URL
        api_key = VENICE_API_KEY
        model = VENICE_MODEL_FALLBACK
        extra_headers = {}

    prompt = (
        "Are these two intelligence claims contradictory?\n\n"
        f"CLAIM A:\n{text_a[:400]}\n\n"
        f"CLAIM B:\n{text_b[:400]}\n\n"
        "Respond EXACTLY:\n"
        "VERDICT: CONTRADICTS | CONSISTENT\n"
        "REASON: <one sentence>"
    )
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        headers.update(extra_headers)
        resp = await http.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.1,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content", "") or ""
        contradicts = "VERDICT: CONTRADICTS" in content.upper()
        reason_match = re.search(r"REASON:\s*(.+)", content, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else ""
        return contradicts, reason
    except Exception as e:
        logger.warning("Contradiction LLM check failed: %s", e)
        return False, ""


async def _contradiction_detection_pass(
    desk: "DeskMeta",
    client: AsyncQdrantClient,
    http: httpx.AsyncClient,
    dry_run: bool,
) -> int:
    """Find semantically similar INTSUM chunks with conflicting claims.

    For each tier-A/A+/A- chunk with entity_tags, searches the same collection
    for similar chunks (cosine ≥ CONTRADICTION_SIMILARITY_THRESHOLD) with
    overlapping entity_tags, then runs a cheap binary LLM contradiction check on
    candidate pairs.

    Writes hermes_contradicts: list[str] and hermes_contradiction_reason: str
    to both chunks in a confirmed contradiction pair.

    Returns count of contradictions found.
    """
    if CONTRA_BATCH_SIZE <= 0:
        return 0

    try:
        records, _ = await client.scroll(
            collection_name=desk.collection,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="reliability_tier",
                        match=qdrant_models.MatchAny(any=["A", "A+", "A-"]),
                    ),
                ]
            ),
            limit=CONTRA_BATCH_SIZE * 5,
            with_payload=True,
            with_vectors=True,
        )
    except Exception as e:
        logger.error("Contradiction: scroll failed for '%s': %s", desk.collection, e)
        return 0

    # Keep only records with non-empty entity_tags and valid vectors
    tagged = [r for r in records if (r.payload or {}).get("entity_tags") and _extract_vector(r.vector) is not None]
    if not tagged:
        return 0

    checked_pairs: set[frozenset] = set()
    pairs_checked = 0
    contradictions_found = 0
    now_ts = int(time.time())

    for record in tagged:
        if pairs_checked >= CONTRA_BATCH_SIZE:
            break

        payload = record.payload or {}
        my_tags = set(payload.get("entity_tags", []))
        vector = _extract_vector(record.vector)
        if not vector:
            continue

        try:
            similar = await client.search(
                collection_name=desk.collection,
                query_vector=vector,
                limit=6,
                score_threshold=CONTRADICTION_SIMILARITY_THRESHOLD,
                with_payload=True,
                with_vectors=False,
                query_filter=qdrant_models.Filter(must_not=[qdrant_models.HasIdCondition(has_id=[record.id])]),
            )
        except Exception as e:
            logger.warning("Contradiction: similarity search failed: %s", e)
            continue

        for neighbour in similar:
            pair_key: frozenset = frozenset([record.id, neighbour.id])
            if pair_key in checked_pairs:
                continue

            nb_tags = set((neighbour.payload or {}).get("entity_tags", []))
            if not (my_tags & nb_tags):
                continue  # no overlapping entity_tags — not about the same subject

            checked_pairs.add(pair_key)
            pairs_checked += 1

            contradicts, reason = await _llm_contradiction_check(
                payload.get("text", ""),
                (neighbour.payload or {}).get("text", ""),
                http,
            )

            if not contradicts:
                continue

            contradictions_found += 1
            logger.info(
                "Contradiction found: %s ↔ %s — %s",
                str(record.id)[:8],
                str(neighbour.id)[:8],
                reason[:120],
            )

            if not dry_run:
                for pid, ep in [(record.id, payload), (neighbour.id, neighbour.payload or {})]:
                    other = str(neighbour.id) if pid == record.id else str(record.id)
                    existing = ep.get("hermes_contradicts") or []
                    if other not in existing:
                        await _patch_payload(
                            client,
                            desk.collection,
                            pid,
                            {
                                "hermes_contradicts": existing + [other],
                                "hermes_contradiction_reason": reason[:300],
                                "hermes_reviewed_at": now_ts,
                            },
                            dry_run=False,
                        )
            else:
                logger.info(
                    "[DRY-RUN] Would flag contradiction: %s ↔ %s",
                    str(record.id)[:8],
                    str(neighbour.id)[:8],
                )

    if contradictions_found:
        logger.info(
            "Contradiction pass '%s': %d found in %d pairs checked",
            desk.slug,
            contradictions_found,
            pairs_checked,
        )
    return contradictions_found


async def _staleness_pass(
    client: AsyncQdrantClient,
    collection: str,
    dry_run: bool,
) -> int:
    """Flag old uncorroborated entity-tagged INTSUM chunks as stale.

    Criteria: tier A or A-  +  older than STALENESS_DAYS  +  entity_tags non-empty
    +  no KB corroboration found (hermes_corroboration_score absent or zero)
    +  not already flagged.

    Writes hermes_stale_flag: true in bulk (no per-point LLM call).
    Returns count of chunks flagged.
    """
    cutoff = int(time.time() - STALENESS_DAYS * 86400)

    try:
        records, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="reliability_tier",
                        match=qdrant_models.MatchAny(any=["A", "A-"]),
                    ),
                    qdrant_models.FieldCondition(
                        key="ingested_at_unix",
                        range=qdrant_models.Range(lte=cutoff),
                    ),
                ],
                must_not=[
                    qdrant_models.FieldCondition(
                        key="hermes_stale_flag",
                        match=qdrant_models.MatchValue(value=True),
                    ),
                ],
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.error("Staleness scroll failed for '%s': %s", collection, e)
        return 0

    to_flag = []
    for r in records:
        payload = r.payload or {}
        if not payload.get("entity_tags"):
            continue  # no real-world entity references — skip
        score = payload.get("hermes_corroboration_score")
        if score and float(score) > 0:
            continue  # already KB-corroborated
        to_flag.append(r.id)

    if not to_flag:
        return 0

    if dry_run:
        logger.info("[DRY-RUN] Would flag %d stale chunks in '%s'", len(to_flag), collection)
        return len(to_flag)

    _WRITE_BATCH = 50
    flagged = 0
    for i in range(0, len(to_flag), _WRITE_BATCH):
        batch = to_flag[i : i + _WRITE_BATCH]
        try:
            await client.set_payload(
                collection_name=collection,
                payload={"hermes_stale_flag": True},
                points=batch,
            )
            flagged += len(batch)
        except Exception as e:
            logger.error("Staleness batch write failed for '%s': %s", collection, e)

    if flagged:
        logger.info("Staleness pass: flagged %d chunks in '%s'", flagged, collection)
    return flagged


# ---------------------------------------------------------------------------
# Per-desk processing
# ---------------------------------------------------------------------------


async def _wiki_enrich_corroboration(
    desk: "DeskMeta",
    payload: dict,
    verdict: str,
    confidence: str,
    reasoning: str,
    sources: list[str],
    checked_at: str,
    dry_run: bool,
) -> str | None:
    """Write corroboration findings to the wiki. Returns a new wiki_path if a page was created.

    Three-path strategy:
    1. payload carries wiki_path  →  patch corroboration section + update metadata reliability tier
    2. no wiki_path, topic known  →  search wiki for an entity page and append a research-notes entry
    3. CORROBORATED HIGH/MODERATE, no wiki page found  →  create a new entity page and return its path

    CONTRADICTED verdicts always append a warning entry to research-notes on any found page.
    Non-fatal — logs and returns None on any failure.
    """
    if dry_run or not os.getenv("WIKIJS_API_KEY"):
        return None

    topic: str = (payload.get("topic") or "").strip()
    wiki_path: str = (payload.get("wiki_path") or "").strip()
    new_tier: str = payload.get("reliability_tier") or "?"

    icon = {"CORROBORATED": "✅", "CONTRADICTED": "⚠️", "UNVERIFIED": "❓"}.get(verdict, "")
    sources_str = ", ".join(sources) if sources else "*none found*"
    date_str = checked_at[:10]

    corroboration_block = (
        f"**Verdict:** {icon} {verdict}  \n"
        f"**Confidence:** {confidence}  \n"
        f"**Verified:** {date_str}  \n"
        f"**Reasoning:** {reasoning or '*(not recorded)*'}  \n"
        f"**Sources:** {sources_str}"
    )

    note_icon = {"CORROBORATED": "✅", "CONTRADICTED": "⚠️"}.get(verdict, "🔍")
    research_note = (
        f"- **{date_str}** [{note_icon} {verdict} / {confidence}] "
        f"Hermes corroboration — **{topic or 'unknown topic'}**\n\n"
        f"  > {(reasoning or '')[:300].replace(chr(10), ' ')}"
    )

    try:
        async with WikiClient() as wiki:
            # --- Path 1: existing page referenced by wiki_path ---
            if wiki_path:
                existing = await wiki.get_page(wiki_path)
                if existing:
                    await wiki.patch_section(wiki_path, "corroboration", corroboration_block)
                    # Update reliability tier in the AUTO:metadata section
                    body = existing.get("content", "")
                    new_body = re.sub(
                        r"(\|\s*\*\*Reliability\*\*\s*\|)\s*[A-Z?]\s*(\|)",
                        rf"\1 {new_tier} \2",
                        body,
                    )
                    if new_body != body:
                        await wiki.update_page(
                            existing["id"],
                            new_body,
                            existing["title"],
                            existing.get("description", ""),
                            existing.get("tags", []),
                        )
                    if verdict != "UNVERIFIED":
                        await wiki.append_to_section(wiki_path, "research-notes", research_note)
                    logger.info(
                        "Wiki: corroboration+tier updated for '%s' → %s (tier: %s)",
                        wiki_path,
                        verdict,
                        new_tier,
                    )
                    return None

            # --- Path 2: no wiki_path — search by topic ---
            if not topic:
                return None

            results = await wiki.search_pages(topic[:50])
            entity_page = next(
                (r for r in results if r.get("path", "").startswith(("entities/", "desks/"))),
                None,
            )

            if entity_page:
                if verdict == "CORROBORATED" and confidence in ("HIGH", "MODERATE"):
                    text_excerpt = (payload.get("text") or "")[:500]
                    await wiki.patch_section(
                        entity_page["path"],
                        "summary",
                        f"{text_excerpt}\n\n---\n\n{corroboration_block}",
                    )
                await wiki.append_to_section(entity_page["path"], "research-notes", research_note)
                logger.info(
                    "Wiki: corroboration note appended to '%s' → %s",
                    entity_page["path"],
                    verdict,
                )
                return None

            # --- Path 3: CORROBORATED HIGH/MODERATE with no wiki presence → create entity page ---
            if verdict == "CORROBORATED" and confidence in ("HIGH", "MODERATE"):
                text_excerpt = (payload.get("text") or "")[:500]
                _desk_section = desk_wiki_section(desk.slug)
                new_path = entity_wiki_path("Organisation", topic)
                content = build_entity_page(
                    entity_type="Organisation",
                    desk_name=desk.name,
                    desk_section=_desk_section,
                    first_seen=date_str,
                    summary=f"{text_excerpt}\n\n---\n\n{corroboration_block}",
                )
                created = await wiki.create_page(
                    new_path,
                    topic,
                    content,
                    description=f"Entity — corroborated {date_str} by Hermes ({confidence} confidence)",
                    tags=["entity", "organisation", "hermes", "corroborated"],
                )
                if created:
                    logger.info("Wiki: created entity page '%s' (corroborated, %s)", new_path, confidence)
                    return new_path

    except Exception as e:
        logger.warning(
            "Wiki corroboration enrichment failed for '%s' (non-fatal): %s",
            topic or wiki_path,
            e,
        )

    return None


async def process_desk(
    desk: DeskMeta,
    limit: int,
    dry_run: bool,
    run_cleanup: bool = True,
) -> tuple[int, int, int]:
    """Corroborate low-confidence points in one desk's collection, then run quality maintenance.

    Returns (corroborated, contradicted, unverified) counts.
    """
    client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, port=None)

    try:
        # Verify collection exists
        if not await client.collection_exists(desk.collection):
            logger.warning("Collection '%s' does not exist — skipping desk %s", desk.collection, desk.slug)
            return 0, 0, 0
    except Exception as e:
        logger.error("Cannot check collection '%s': %s", desk.collection, e)
        return 0, 0, 0

    # Scroll for candidate points (over-fetch to allow for cooldown filtering)
    candidates = await _scroll_low_confidence(client, desk.collection, scan_limit=limit * 5)
    logger.info("Desk '%s': %d candidate points scrolled (limit: %d)", desk.slug, len(candidates), limit)

    # Filter by cooldown
    now = time.time()
    targets: list[tuple[int | str, dict]] = []
    for point_id, payload in candidates:
        checked_at = payload.get("corroboration_checked_at")
        if checked_at:
            try:
                age = now - datetime.fromisoformat(checked_at).timestamp()
                if age < COOLDOWN_SECONDS:
                    continue
            except (ValueError, OSError):
                pass  # malformed ISO timestamp in corroboration_checked_at — treat as unchecked
        targets.append((point_id, payload))
        if len(targets) >= limit:
            break

    logger.info("Desk '%s': %d points to corroborate after cooldown filter", desk.slug, len(targets))

    corroborated = 0
    contradicted = 0
    unverified = 0

    async with httpx.AsyncClient(timeout=120.0) as http:
        # ---- Research corroboration loop (tier B/C/?) ----
        for point_id, payload in targets:
            topic = payload.get("topic", "")
            source = payload.get("source", "unknown")
            current_tier = payload.get("reliability_tier", "")
            text = payload.get("text", "")

            if not text:
                logger.warning("Point %s has no text — skipping", str(point_id)[:8])
                continue

            try:
                raw_output = await _run_corroboration_loop(
                    desk=desk,
                    point_id=point_id,
                    topic=topic,
                    source=source,
                    current_tier=current_tier,
                    excerpt=text,
                    http=http,
                )
            except Exception as e:
                logger.error("Corroboration loop failed for point %s: %s", str(point_id)[:8], e)
                continue

            verdict, confidence, reasoning, sources = _parse_verdict(raw_output)
            logger.info("Point %s verdict: %s (%s confidence)", str(point_id)[:8], verdict, confidence)

            checked_now = datetime.now(UTC).isoformat()
            patch: dict = {
                "corroboration_verdict": verdict,
                "corroboration_confidence": confidence,
                "corroboration_checked_at": checked_now,
                "corroboration_reasoning": reasoning[:500] if reasoning else "",
                "corroboration_sources": sources,
                "corroboration_check_count": payload.get("corroboration_check_count", 0) + 1,
            }

            if verdict == "CORROBORATED":
                patch["reliability_tier"] = _upgrade_tier(current_tier)
                corroborated += 1
            elif verdict == "CONTRADICTED":
                patch["reliability_tier"] = _downgrade_tier(current_tier)
                patch["contradiction_flag"] = True
                contradicted += 1
            else:
                unverified += 1

            enriched_payload = {**payload, **patch}
            new_wiki_path = await _wiki_enrich_corroboration(
                desk, enriched_payload, verdict, confidence, reasoning, sources, checked_now, dry_run
            )
            if new_wiki_path:
                patch["wiki_path"] = new_wiki_path

            await _patch_payload(client, desk.collection, point_id, patch, dry_run)
            await asyncio.sleep(2)

        # ---- RAG contamination remediation passes (tier A/A+/A-) ----
        await _corroboration_scoring_pass(desk, client, dry_run)
        await _echo_chamber_pass(desk, client, dry_run)
        await _contradiction_detection_pass(desk, client, http, dry_run)
        await _staleness_pass(client, desk.collection, dry_run)

    if run_cleanup:
        await _run_cleanup_pass(client, desk.collection, dry_run)

    return corroborated, contradicted, unverified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="OSIA Hermes Worker")
    parser.add_argument(
        "--desk",
        metavar="SLUG",
        help="Process only this desk slug (default: all desks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Research and parse verdicts but do not write back to Qdrant",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=f"Max points to corroborate per desk (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip Qdrant quality maintenance passes (empty/stale/unverifiable point cleanup)",
    )
    parser.add_argument(
        "--worker-id",
        default=socket.gethostname(),
        metavar="ID",
        help="Identifier stamped into Redis desk leases (default: hostname). "
        "Use to distinguish workers in multi-node deployments.",
    )
    parser.add_argument(
        "--clear-leases",
        action="store_true",
        help="Delete all active Hermes desk leases from Redis and exit. Use to unblock desks after a crashed worker.",
    )
    args = parser.parse_args()

    # --clear-leases: admin action — connect, clear, exit immediately.
    if args.clear_leases:
        r = _get_redis()
        if r is None:
            logger.error("Cannot clear leases — Redis unavailable")
            return
        n = _clear_all_leases(r)
        if n:
            logger.info("Cleared %d Hermes desk lease(s)", n)
        else:
            logger.info("No active Hermes desk leases found")
        return

    if not VENICE_API_KEY and not OPENROUTER_API_KEY and not GEMINI_API_KEY:
        logger.error("No API key set (VENICE_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY) — cannot run")
        return

    # Connect to Redis once for the entire run; None = single-node mode.
    redis_client = _get_redis()
    if redis_client is not None:
        active = _list_leases(redis_client)
        if active:
            logger.info(
                "Active leases at startup: %s",
                ", ".join(f"{slug}@{wid}({ttl}s)" for slug, wid, ttl in active),
            )

    desk_slugs = [args.desk] if args.desk else _list_all_desks()
    logger.info(
        "=== OSIA Hermes Worker starting === worker=%s desks=%d limit=%d dry_run=%s cleanup=%s",
        args.worker_id,
        len(desk_slugs),
        args.limit,
        args.dry_run,
        not args.no_cleanup,
    )

    total_corroborated = 0
    total_contradicted = 0
    total_unverified = 0
    total_skipped = 0
    total_leased_out = 0

    for slug in desk_slugs:
        try:
            desk = _load_desk_meta(slug)
        except FileNotFoundError as e:
            logger.error("Desk config not found: %s", e)
            total_skipped += 1
            continue

        # Claim the desk lease before doing any work.
        if not _try_claim_desk(redis_client, slug, args.worker_id):
            logger.info(
                "Desk '%s' is held by another worker — skipping (lease TTL: %ds)",
                slug,
                redis_client.ttl(f"{_LEASE_PREFIX}{slug}") if redis_client else 0,
            )
            total_leased_out += 1
            continue

        logger.info("--- Processing desk: %s (%s) [lease claimed] ---", desk.name, desk.collection)
        try:
            c, x, u = await process_desk(
                desk,
                limit=args.limit,
                dry_run=args.dry_run,
                run_cleanup=not args.no_cleanup,
            )
        except Exception as e:
            logger.error("Desk '%s' failed: %s", slug, e)
            total_skipped += 1
            continue
        finally:
            # Always release — even on exception or dry-run.
            _release_desk(redis_client, slug)

        total_corroborated += c
        total_contradicted += x
        total_unverified += u
        logger.info(
            "Desk '%s' done: corroborated=%d contradicted=%d unverified=%d",
            slug,
            c,
            x,
            u,
        )
        await asyncio.sleep(5)

    logger.info(
        "=== Batch complete: corroborated=%d contradicted=%d unverified=%d skipped=%d leased_out=%d ===",
        total_corroborated,
        total_contradicted,
        total_unverified,
        total_skipped,
        total_leased_out,
    )


if __name__ == "__main__":
    asyncio.run(main())
