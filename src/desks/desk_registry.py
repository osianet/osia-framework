"""
OSIA Desk Registry

Loads all desk configurations from config/desks/*.yaml, assembles system prompts,
and dispatches model invocations directly via native SDKs or OpenAI-compatible HTTP.

Environment variables:
  GEMINI_API_KEY       — Google Gemini API key
  ANTHROPIC_API_KEY    — Anthropic API key
  OPENAI_API_KEY       — OpenAI API key
  HF_TOKEN             — HuggingFace token (HF endpoint auth + HFEndpointManager)
  HF_NAMESPACE         — HuggingFace namespace (HFEndpointManager)
  OSIA_DIRECTIVES_FILE — Path to analytical mandate file (default: DIRECTIVES.md)
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

from src.desks.hf_endpoint_manager import HFEndpointManager
from src.intelligence.source_tracker import build_citation_protocol

load_dotenv()

logger = logging.getLogger("osia.desk_registry")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PROVIDERS = {"gemini", "anthropic", "openai", "hf_endpoint", "openrouter", "venice"}
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONTEXT_TOP_K = 5
DEFAULT_CROSS_DESK_TOP_K = 3
REQUEST_TIMEOUT = 300.0
RETRY_COUNT = 3
RETRY_DELAY = 15.0

# HF endpoints can take several minutes to load a 70B model after reporting
# "running" — use a longer delay and more attempts, and also retry on 422
# which HF returns while the model server is still warming up.
HF_RETRY_COUNT = 6
HF_RETRY_DELAY = 30.0
HF_RETRIABLE_STATUSES = frozenset({422, 503})

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    hf_endpoint_name: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass
class QdrantConfig:
    collection: str
    context_top_k: int = DEFAULT_CONTEXT_TOP_K
    cross_desk_search: bool = True
    cross_desk_top_k: int = DEFAULT_CROSS_DESK_TOP_K


@dataclass
class DeskConfig:
    slug: str
    name: str
    prompt_file: str
    prompt_text: str
    model_primary: ModelConfig
    model_fallback: ModelConfig | None
    qdrant: QdrantConfig
    tools: list[str]
    mcp_servers: list[str]
    system_prompt: str  # fully assembled: prompt + mandate + citation protocol
    entity_research_target: bool = True


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------


def _parse_model_config(data: dict, file_path: str, key_prefix: str) -> ModelConfig:
    """Parse a model config block (primary or fallback)."""
    provider = data.get("provider")
    if not provider:
        raise ValueError(f"{file_path}: missing required key '{key_prefix}.provider'")
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"{file_path}: invalid provider '{provider}' in '{key_prefix}.provider'. "
            f"Must be one of: {', '.join(sorted(VALID_PROVIDERS))}"
        )

    model_id = data.get("model_id")
    if not model_id:
        raise ValueError(f"{file_path}: missing required key '{key_prefix}.model_id'")

    hf_endpoint_name = data.get("hf_endpoint_name") or None
    if provider == "hf_endpoint" and not hf_endpoint_name:
        raise ValueError(f"{file_path}: '{key_prefix}.hf_endpoint_name' is required when provider is 'hf_endpoint'")

    return ModelConfig(
        provider=provider,
        model_id=model_id,
        hf_endpoint_name=hf_endpoint_name,
        temperature=float(data.get("temperature", DEFAULT_TEMPERATURE)),
        max_tokens=int(data.get("max_tokens", DEFAULT_MAX_TOKENS)),
    )


def _parse_desk_yaml(path: Path, mandate_text: str, citation_protocol: str) -> DeskConfig:
    """Load and validate a single desk YAML file into a DeskConfig."""
    file_str = str(path)

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{file_str}: YAML root must be a mapping")

    # Validate required top-level keys
    required_keys = ["slug", "name", "prompt_file", "model", "qdrant", "tools", "mcp_servers", "entity_research_target"]
    for key in required_keys:
        if key not in raw:
            raise ValueError(f"{file_str}: missing required key '{key}'")

    slug: str = raw["slug"]
    name: str = raw["name"]
    prompt_file: str = raw["prompt_file"]

    # Load prompt text
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found for desk '{slug}': {prompt_file}")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    # Parse model block
    model_block = raw["model"]
    if not isinstance(model_block, dict) or "primary" not in model_block:
        raise ValueError(f"{file_str}: missing required key 'model.primary'")

    model_primary = _parse_model_config(model_block["primary"], file_str, "model.primary")
    model_fallback: ModelConfig | None = None
    if "fallback" in model_block and model_block["fallback"]:
        model_fallback = _parse_model_config(model_block["fallback"], file_str, "model.fallback")

    # Parse qdrant block
    qdrant_block = raw["qdrant"]
    if not isinstance(qdrant_block, dict) or "collection" not in qdrant_block:
        raise ValueError(f"{file_str}: missing required key 'qdrant.collection'")

    qdrant_cfg = QdrantConfig(
        collection=qdrant_block["collection"],
        context_top_k=int(qdrant_block.get("context_top_k", DEFAULT_CONTEXT_TOP_K)),
        cross_desk_search=bool(qdrant_block.get("cross_desk_search", True)),
        cross_desk_top_k=int(qdrant_block.get("cross_desk_top_k", DEFAULT_CROSS_DESK_TOP_K)),
    )

    tools: list[str] = list(raw["tools"] or [])
    mcp_servers: list[str] = list(raw["mcp_servers"] or [])
    entity_research_target: bool = bool(raw["entity_research_target"])

    # Assemble full system prompt
    system_prompt = _assemble_system_prompt(prompt_text, mandate_text, citation_protocol)

    return DeskConfig(
        slug=slug,
        name=name,
        prompt_file=prompt_file,
        prompt_text=prompt_text,
        model_primary=model_primary,
        model_fallback=model_fallback,
        qdrant=qdrant_cfg,
        tools=tools,
        mcp_servers=mcp_servers,
        system_prompt=system_prompt,
        entity_research_target=entity_research_target,
    )


def _assemble_system_prompt(prompt_text: str, mandate_text: str, citation_protocol: str) -> str:
    """Combine desk prompt + mandate + citation protocol into the full system prompt."""
    parts = [prompt_text.rstrip()]

    if mandate_text and "## ANALYTICAL MANDATE" not in prompt_text:
        parts.append("\n\n## ANALYTICAL MANDATE\n" + mandate_text.strip())

    parts.append(citation_protocol)
    return "".join(parts)


# ---------------------------------------------------------------------------
# DeskRegistry
# ---------------------------------------------------------------------------


class DeskRegistry:
    """
    Loads all desk configs from config/desks/*.yaml, assembles system prompts,
    and dispatches model invocations via native SDKs or OpenAI-compatible HTTP.
    """

    def __init__(self, desks_dir: str = "config/desks") -> None:
        self._desks: dict[str, DeskConfig] = {}
        self._hf_manager = HFEndpointManager()
        self._http_client: httpx.AsyncClient | None = None

        # Load analytical mandate
        mandate_text = self._load_mandate()

        # Load citation protocol
        citation_protocol = build_citation_protocol()

        # Discover and load all desk YAML files
        desks_path = Path(desks_dir)
        yaml_files = sorted(desks_path.glob("*.yaml"))
        if not yaml_files:
            raise RuntimeError(
                f"No desk configuration files found in '{desks_dir}'. At least one *.yaml file is required."
            )

        for yaml_path in yaml_files:
            desk = _parse_desk_yaml(yaml_path, mandate_text, citation_protocol)
            self._desks[desk.slug] = desk

        self._log_startup_summary()

    # ------------------------------------------------------------------
    # Mandate loading
    # ------------------------------------------------------------------

    def _load_mandate(self) -> str:
        """Load the analytical mandate from OSIA_DIRECTIVES_FILE."""
        directives_file = os.getenv("OSIA_DIRECTIVES_FILE", "DIRECTIVES.md")

        if directives_file == "":
            logger.warning("OSIA_DIRECTIVES_FILE is set to empty string — no analytical mandate will be appended.")
            return ""

        mandate_path = Path(directives_file)
        if not mandate_path.exists():
            raise FileNotFoundError(
                f"Analytical mandate file not found: '{directives_file}' (set OSIA_DIRECTIVES_FILE to override)"
            )

        text = mandate_path.read_text(encoding="utf-8")
        logger.info("Loaded analytical mandate from '%s' (%d chars)", directives_file, len(text))
        return text

    # ------------------------------------------------------------------
    # Startup summary
    # ------------------------------------------------------------------

    def _log_startup_summary(self) -> None:
        logger.info("DeskRegistry loaded %d desks:", len(self._desks))
        for slug, desk in self._desks.items():
            primary = desk.model_primary
            fallback_str = ""
            if desk.model_fallback:
                fb = desk.model_fallback
                fallback_str = f" | fallback: {fb.provider}/{fb.model_id}"
            logger.info(
                "  [%s] %s — primary: %s/%s%s | collection: %s | tools: %d",
                slug,
                desk.name,
                primary.provider,
                primary.model_id,
                fallback_str,
                desk.qdrant.collection,
                len(desk.tools),
            )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get(self, slug: str) -> DeskConfig:
        """Return the DeskConfig for the given slug. Raises KeyError if unknown."""
        if slug not in self._desks:
            raise KeyError(f"Unknown desk slug: '{slug}'")
        return self._desks[slug]

    def list_slugs(self) -> list[str]:
        """Return all loaded desk slugs."""
        return list(self._desks.keys())

    # ------------------------------------------------------------------
    # HTTP client (lazy init, shared across invocations)
    # ------------------------------------------------------------------

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        return self._http_client

    async def close(self) -> None:
        """Shut down HTTP clients."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            logger.info("DeskRegistry HTTP client closed.")

    # ------------------------------------------------------------------
    # Model invocation
    # ------------------------------------------------------------------

    async def invoke(
        self,
        slug: str,
        user_message: str,
        context_block: str | None = None,
    ) -> tuple[str, dict]:
        """
        Invoke the desk's configured model and return (response_text, metadata).
        metadata contains: {model_used: "primary"|"fallback", model_id: str}

        Raises RuntimeError / provider exceptions on unrecoverable failure.
        """
        desk = self.get(slug)

        assembled_message = self._assemble_user_message(user_message, context_block)

        # Primary attempt
        try:
            text = await self._invoke_model(desk, desk.model_primary, assembled_message)
            return text, {"model_used": "primary", "model_id": desk.model_primary.model_id}
        except Exception as primary_exc:
            if desk.model_fallback is None:
                raise

            logger.warning(
                "Desk '%s': primary model %s/%s failed (%s). Attempting fallback %s/%s.",
                slug,
                desk.model_primary.provider,
                desk.model_primary.model_id,
                primary_exc,
                desk.model_fallback.provider,
                desk.model_fallback.model_id,
            )

        # Fallback attempt
        try:
            text = await self._invoke_model(desk, desk.model_fallback, assembled_message)
            return text, {"model_used": "fallback", "model_id": desk.model_fallback.model_id}
        except Exception as fallback_exc:
            logger.error(
                "Desk '%s': fallback model %s/%s also failed (%s).",
                slug,
                desk.model_fallback.provider,
                desk.model_fallback.model_id,
                fallback_exc,
            )
            raise fallback_exc

    # ------------------------------------------------------------------
    # Internal: assemble user message
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_user_message(user_message: str, context_block: str | None) -> str:
        if context_block and context_block.strip():
            return f"## INTELLIGENCE CONTEXT\n{context_block.strip()}\n\n{user_message}"
        return user_message

    # ------------------------------------------------------------------
    # Internal: dispatch to provider with retry
    # ------------------------------------------------------------------

    async def _invoke_model(
        self,
        desk: DeskConfig,
        model_cfg: ModelConfig,
        assembled_message: str,
    ) -> str:
        """Dispatch to the correct provider with retry on HTTP 503."""
        if model_cfg.provider == "hf_endpoint":
            return await self._invoke_hf_endpoint(desk, model_cfg, assembled_message)
        elif model_cfg.provider == "gemini":
            return await self._invoke_with_retry(lambda: self._call_gemini(desk, model_cfg, assembled_message))
        elif model_cfg.provider == "anthropic":
            return await self._invoke_with_retry(lambda: self._call_anthropic(desk, model_cfg, assembled_message))
        elif model_cfg.provider == "openai":
            return await self._invoke_with_retry(
                lambda: self._call_openai_compat(
                    desk,
                    model_cfg,
                    assembled_message,
                    base_url="https://api.openai.com",
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                )
            )
        elif model_cfg.provider == "openrouter":
            return await self._invoke_with_retry(
                lambda: self._call_openai_compat(
                    desk,
                    model_cfg,
                    assembled_message,
                    base_url="https://openrouter.ai/api",
                    api_key=os.getenv("OPENROUTER_API_KEY", ""),
                    extra_headers={"HTTP-Referer": "https://osia.dev", "X-Title": "OSIA Intelligence Framework"},
                )
            )
        elif model_cfg.provider == "venice":
            return await self._invoke_with_retry(
                lambda: self._call_openai_compat(
                    desk,
                    model_cfg,
                    assembled_message,
                    base_url="https://api.venice.ai/api",
                    api_key=os.getenv("VENICE_API_KEY", ""),
                )
            )
        else:
            raise ValueError(f"Unknown provider: {model_cfg.provider}")

    async def _invoke_with_retry(
        self,
        fn,
        retry_count: int = RETRY_COUNT,
        retry_delay: float = RETRY_DELAY,
        retriable_statuses: frozenset[int] = frozenset({503}),
    ) -> str:
        """Call fn up to retry_count times, retrying on retriable_statuses."""
        last_exc: Exception | None = None
        for attempt in range(1, retry_count + 1):
            try:
                return await fn()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in retriable_statuses and attempt < retry_count:
                    logger.warning(
                        "HTTP %d on attempt %d/%d — retrying in %ds",
                        exc.response.status_code,
                        attempt,
                        retry_count,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    last_exc = exc
                else:
                    raise
            except Exception as exc:
                raise exc
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _call_gemini(
        self,
        desk: DeskConfig,
        model_cfg: ModelConfig,
        assembled_message: str,
    ) -> str:
        from google import genai
        from google.genai import types as genai_types

        api_key = os.getenv("GEMINI_API_KEY", "")
        client = genai.Client(api_key=api_key)

        config = genai_types.GenerateContentConfig(
            system_instruction=desk.system_prompt,
            temperature=model_cfg.temperature,
            max_output_tokens=model_cfg.max_tokens,
        )

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model_cfg.model_id,
            contents=assembled_message,
            config=config,
        )
        return response.text

    async def _call_anthropic(
        self,
        desk: DeskConfig,
        model_cfg: ModelConfig,
        assembled_message: str,
    ) -> str:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)

        response = await asyncio.to_thread(
            client.messages.create,
            model=model_cfg.model_id,
            system=desk.system_prompt,
            messages=[{"role": "user", "content": assembled_message}],
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )
        return response.content[0].text

    async def _call_openai_compat(
        self,
        desk: DeskConfig,
        model_cfg: ModelConfig,
        assembled_message: str,
        base_url: str,
        api_key: str,
        extra_headers: dict | None = None,
    ) -> str:
        http = self._get_http_client()
        payload = {
            "model": model_cfg.model_id,
            "messages": [
                {"role": "system", "content": desk.system_prompt},
                {"role": "user", "content": assembled_message},
            ],
            "temperature": model_cfg.temperature,
            "max_tokens": model_cfg.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }
        resp = await http.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def _invoke_hf_endpoint(
        self,
        desk: DeskConfig,
        model_cfg: ModelConfig,
        assembled_message: str,
    ) -> str:
        """Wake HF endpoint, then dispatch with retry."""
        ready = await self._hf_manager.ensure_ready(desk.slug)
        if not ready:
            raise RuntimeError(
                f"Desk '{desk.slug}': HF endpoint '{model_cfg.hf_endpoint_name}' failed to become ready."
            )

        endpoint_url = await self._resolve_hf_endpoint_url(model_cfg)

        return await self._invoke_with_retry(
            lambda: self._call_openai_compat(
                desk,
                model_cfg,
                assembled_message,
                base_url=endpoint_url,
                api_key=os.getenv("HF_TOKEN", ""),
            ),
            retry_count=HF_RETRY_COUNT,
            retry_delay=HF_RETRY_DELAY,
            retriable_statuses=HF_RETRIABLE_STATUSES,
        )

    async def _resolve_hf_endpoint_url(self, model_cfg: ModelConfig) -> str:
        """
        Resolve the HF endpoint base URL.
        When HFEndpointManager is disabled, falls back to env vars.
        When enabled, fetches the URL from the HF hub via asyncio.to_thread.
        """
        if not self._hf_manager.enabled:
            name = (model_cfg.hf_endpoint_name or "").lower()
            if "dolphin" in name:
                url = os.getenv("HF_ENDPOINT_DOLPHIN_24B", "")
            else:
                url = os.getenv("HF_ENDPOINT_HERMES_70B", "")
            if not url:
                raise RuntimeError(
                    f"HFEndpointManager is disabled and no fallback URL env var is set "
                    f"for endpoint '{model_cfg.hf_endpoint_name}'"
                )
            return url.rstrip("/")

        # Fetch URL from HF hub without blocking the event loop
        url = await asyncio.to_thread(self._hf_manager._wake_and_wait, model_cfg.hf_endpoint_name)
        if not url:
            raise RuntimeError(f"Could not resolve URL for HF endpoint '{model_cfg.hf_endpoint_name}'")
        return url.rstrip("/")
