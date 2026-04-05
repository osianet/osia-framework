"""
OSIA Hermes Worker — validates low-confidence intel by searching for corroborating
or contradicting evidence, then updates Qdrant payload in-place.

Named after Hermes, messenger of the gods and guide across boundaries — the worker
traverses sources and carries evidence back to validate existing intelligence rather
than generating new intel from scratch. The name also reflects the model choice:
Hermes 4 (NousResearch) runs all desks.

For each desk, scrolls the Qdrant collection for points where reliability_tier is
not "A" (RSS-ingested B-tier, unknown ?, research worker outputs, etc.) that have
not been recently checked. Runs a focused research loop per point and patches the
payload in-place with a structured verdict and updated reliability_tier.

Model routing:
  All desks → nousresearch/hermes-4-70b via OpenRouter (SOTA on RefusalBench —
              no censorship, native tool calling, strong structured output)
  Fallback  → mistral-31-24b via Venice if OPENROUTER_API_KEY is not set
  Fallback  → Gemini if neither key is available

Verdict transitions:
  CORROBORATED  → missing/? → C; C → B; B → A; A stays A
  CONTRADICTED  → A → B; B → C; C stays C; sets contradiction_flag: true
  UNVERIFIED    → no tier change; marks corroboration_checked_at only

Usage:
  uv run python -m src.workers.hermes_worker [--desk <slug>] [--dry-run] [--limit N]

Environment variables:
  OPENROUTER_API_KEY           — required for Hermes 4 (primary)
  VENICE_API_KEY               — fallback if no OpenRouter key
  GEMINI_API_KEY               — final fallback
  QDRANT_URL, QDRANT_API_KEY, HF_TOKEN, TAVILY_API_KEY, OTX_API_KEY

  HERMES_MODEL                 — override OpenRouter model ID
                                  (default: nousresearch/hermes-4-70b)
  HERMES_COOLDOWN_DAYS         — days before a point can be re-checked (default: 14)
  HERMES_BATCH_SIZE            — max points per desk per run (default: 10)

Manual trigger (fire-and-forget):
  sudo systemctl start --no-block osia-hermes-worker.service

Run directly:
  uv run python -m src.workers.hermes_worker
  uv run python -m src.workers.hermes_worker --desk geopolitical-and-security-desk
  uv run python -m src.workers.hermes_worker --dry-run --limit 3
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

# Tool infrastructure — imported from research_worker to avoid duplication.
# Safe to import: module-level code only loads env vars and defines functions.
from src.workers.research_worker import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REACT_ONLY_MODELS,  # kept for Venice fallback path
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    VENICE_API_KEY,
    VENICE_BASE_URL,
    _parse_react,  # kept for Venice fallback path
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

COOLDOWN_DAYS = float(os.getenv("HERMES_COOLDOWN_DAYS", "14"))
COOLDOWN_SECONDS = COOLDOWN_DAYS * 86400
BATCH_SIZE = int(os.getenv("HERMES_BATCH_SIZE", "10"))

DESKS_DIR = Path("config/desks")

# Hermes 4 via OpenRouter — used for ALL desks.
# SOTA on RefusalBench: handles the full range of intelligence content without
# censorship or refusals. 131K context, strong structured output.
# Override with hermes-4-405b for higher accuracy at ~4x cost.
HERMES_MODEL = os.getenv("HERMES_MODEL", "nousresearch/hermes-4-70b")

# Venice fallback — only used when OPENROUTER_API_KEY is not configured.
VENICE_MODEL_FALLBACK = os.getenv("VENICE_MODEL_FALLBACK", "mistral-31-24b")

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
    Falls back to Venice mistral-31-24b if OpenRouter is not configured.
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


def _load_desk_meta(slug: str) -> DeskMeta:
    path = DESKS_DIR / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No desk config found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    collection = raw.get("qdrant", {}).get("collection", slug)
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
    search_duckduckgo is always included as a web fallback.
    All other tools are gated by the desk's tools list.
    """
    always_included = {"search_intel_kb", "search_duckduckgo"}
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
        "search_intel_kb",
        "search_web",
        "search_duckduckgo",
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
    import re

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
            query = args.get("query", "") if isinstance(args, dict) else ""
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
                    query = args.get("query", "") if isinstance(args, dict) else ""
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
            query = args.get("query", "") if isinstance(args, dict) else ""
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
) -> str:
    model, _, _ = _model_for_desk(desk.slug)
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
                query = fn_args.get("query", "")
                logger.info("  → tool call: %s(%r)", fn_name, query)
                tool_fn = TOOL_REGISTRY.get(fn_name)
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
                    tool_fn = TOOL_REGISTRY.get(fn_name)
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
                tool_fn = TOOL_REGISTRY.get(fn_name)
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
            query = dict(call.args).get("query", "") if call.args else ""
            logger.info("Tool: %s(%r)", call.name, query)
            tool_fn = TOOL_REGISTRY.get(call.name)
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
    model, base_url, api_key = _model_for_desk(desk.slug)

    if api_key:
        extra_headers = (
            {"HTTP-Referer": "https://osia.dev", "X-Title": "OSIA Hermes Worker"}
            if base_url == OPENROUTER_BASE_URL
            else None
        )
        return await _run_corroboration_loop_openai_compat(
            desk,
            point_id,
            topic,
            source,
            current_tier,
            excerpt,
            http,
            base_url=base_url,
            api_key=api_key,
            extra_headers=extra_headers,
        )

    if GEMINI_API_KEY:
        logger.warning("No Venice/OpenRouter key available for desk '%s' — falling back to Gemini", desk.slug)
        return await _run_corroboration_loop_gemini(desk, point_id, topic, source, current_tier, excerpt, http)

    raise RuntimeError("No API key set: VENICE_API_KEY, OPENROUTER_API_KEY, or GEMINI_API_KEY required")


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
                        match=qdrant_models.MatchValue(value="A"),
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
# Per-desk processing
# ---------------------------------------------------------------------------


async def process_desk(desk: DeskMeta, limit: int, dry_run: bool) -> tuple[int, int, int]:
    """Corroborate low-confidence points in one desk's collection.

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
    if not targets:
        return 0, 0, 0

    corroborated = 0
    contradicted = 0
    unverified = 0

    async with httpx.AsyncClient(timeout=120.0) as http:
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

            # Build payload patch
            checked_now = datetime.now(UTC).isoformat()
            patch: dict = {
                "corroboration_verdict": verdict,
                "corroboration_confidence": confidence,
                "corroboration_checked_at": checked_now,
                "corroboration_reasoning": reasoning[:500] if reasoning else "",
                "corroboration_sources": sources,
            }

            if verdict == "CORROBORATED":
                patch["reliability_tier"] = _upgrade_tier(current_tier)
                corroborated += 1
            elif verdict == "CONTRADICTED":
                patch["reliability_tier"] = _downgrade_tier(current_tier)
                patch["contradiction_flag"] = True
                contradicted += 1
            else:
                # UNVERIFIED — no tier change, keep existing or leave absent
                unverified += 1

            await _patch_payload(client, desk.collection, point_id, patch, dry_run)
            await asyncio.sleep(2)  # rate-limit buffer between points

    return corroborated, contradicted, unverified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="OSIA Corroboration Worker")
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
    args = parser.parse_args()

    if not VENICE_API_KEY and not OPENROUTER_API_KEY and not GEMINI_API_KEY:
        logger.error("No API key set (VENICE_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY) — cannot run")
        return

    desk_slugs = [args.desk] if args.desk else _list_all_desks()
    logger.info(
        "=== OSIA Corroboration Worker starting === desks=%d limit=%d dry_run=%s",
        len(desk_slugs),
        args.limit,
        args.dry_run,
    )

    total_corroborated = 0
    total_contradicted = 0
    total_unverified = 0
    total_skipped = 0

    for slug in desk_slugs:
        try:
            desk = _load_desk_meta(slug)
        except FileNotFoundError as e:
            logger.error("Desk config not found: %s", e)
            total_skipped += 1
            continue

        logger.info("--- Processing desk: %s (%s) ---", desk.name, desk.collection)
        try:
            c, x, u = await process_desk(desk, limit=args.limit, dry_run=args.dry_run)
        except Exception as e:
            logger.error("Desk '%s' failed: %s", slug, e)
            total_skipped += 1
            continue

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
        # Brief pause between desks to avoid 429s when processing all desks in sequence
        await asyncio.sleep(5)

    logger.info(
        "=== Batch complete: corroborated=%d contradicted=%d unverified=%d skipped_desks=%d ===",
        total_corroborated,
        total_contradicted,
        total_unverified,
        total_skipped,
    )


if __name__ == "__main__":
    asyncio.run(main())
