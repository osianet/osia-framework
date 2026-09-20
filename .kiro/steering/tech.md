# Tech Stack & Build System

## Runtime

- Python 3.11+ (`.python-version` specifies 3.11)
- Package manager: `uv` (modern Python package manager)
- Project config: `pyproject.toml` (PEP 621)
- Lock file: `uv.lock`

## Core Dependencies

- `httpx` — async HTTP client (OpenAI-compatible LLM/vision providers, Signal API, AnythingLLM API)
- `src/intelligence/text_client.py` / `vision_client.py` — provider-agnostic text + image/video clients (Venice → OpenRouter, OpenAI-compatible). **Google-free by default.**
- `google-genai` — Gemini SDK. OPTIONAL for LLM/vision (opt-in last-resort backend only); still used by the research/hermes tool-calling loops' Gemini fallback path.
- `redis` — async Redis client for task queue (`redis.asyncio`)
- `websockets` — Signal WebSocket listener
- `fastapi` + `uvicorn` — HTTP bridges (phone bridge, MCP SSE bridge)
- `mcp[cli]` — Model Context Protocol client/server SDK
- `feedparser` + `beautifulsoup4` — RSS feed parsing
- `yt-dlp` — YouTube transcript extraction
- `python-dotenv` — environment variable loading
- `pandas`, `matplotlib`, `openpyxl` — data analysis and charting
- `huggingface-hub` — HuggingFace Inference Endpoints management (provisioning, scale-to-zero)

## Infrastructure

- Redis (via Docker) — task queue and state tracking
- Queue API (`queue.osia.dev`) — authenticated HTTP wrapper around Redis for remote worker access
- Signal CLI REST API (via Docker) — encrypted messaging gateway
- AnythingLLM — isolated LLM workspaces acting as intelligence desks
- Qdrant (`qdrant.osia.dev`) — vector database for intelligence storage and RAG retrieval
- HuggingFace Jobs — on-demand batch compute for research workers (CPU Basic, ~$0.01/hr)
- HuggingFace Spaces — (removed, replaced by HF Jobs)
- systemd — service management (see `systemd/` directory)
- Nginx — reverse proxy with Let's Encrypt wildcard certs
- HuggingFace Inference Endpoints — dedicated scale-to-zero GPU endpoints for uncensored models (Dolphin 3.0, Hermes 3)

## Hardware Target

- Orange Pi 5 Plus (ARM64) — primary host
- Moto g06 (Android) — physical phone gateway via USB/ADB
- RTX 3080 Ti — local GPU compute for uncensored models

## Common Commands

```bash
# Install dependencies
uv sync

# Run the orchestrator
uv run python main.py

# Run individual gateways
uv run python -m src.gateways.signal_ingress
uv run python -m src.gateways.rss_ingress
uv run python -m src.gateways.phone_bridge

# Run the Queue API (Redis HTTP wrapper for remote workers)
uv run python src/gateways/queue_api.py

# Run the research worker locally (for testing)
uv run python -m src.workers.research_worker

# Run the MCP stdio-to-SSE bridge
uv run python src/mcp_bridge.py --command <cmd> --port <port> --name <name>

# Trigger a daily SITREP manually
uv run python -m src.cron.daily_sitrep

# Start infrastructure services
docker compose up -d

# Manage HuggingFace Inference Endpoints (uncensored models)
uv run python scripts/provision_hf_endpoints.py            # provision
uv run python scripts/provision_hf_endpoints.py --status    # check
uv run python scripts/provision_hf_endpoints.py --pause     # stop billing
```

## Environment Configuration

All secrets and config live in `.env` (git-ignored). See `.env.example` for required variables. Key ones:

- `VENICE_API_KEY` / `OPENROUTER_API_KEY` — primary LLM + vision providers (at least one required; Google-free stack)
- `GEMINI_API_KEY` — Google Gemini API key. OPTIONAL — opt-in last-resort backend only (vision needs `OSIA_VISION_ALLOW_GEMINI=1`)
- `SIGNAL_SENDER_NUMBER` — registered Signal number
- `ANYTHINGLLM_API_KEY` — AnythingLLM workspace access
- `REDIS_URL` — Redis connection string
- `OSIA_BASE_DIR` — project root path
- `MCP_TOOLS_BASE` — parent directory for MCP tool installations
- `HF_TOKEN` — HuggingFace write-scoped token (for Inference Endpoints and embedding API)
- `HF_NAMESPACE` — HuggingFace username or org
- `QUEUE_API_TOKEN` — bearer token for the Queue API
- `QUEUE_API_UA_SENTINEL` — user-agent sentinel for Queue API requests
- `TAVILY_API_KEY` — Tavily web search API key (used by research worker)
- `RESEARCH_BATCH_THRESHOLD` — min queue depth before triggering an HF Job (default: 3)
- `RESEARCH_JOB_FLAVOR` — HF Jobs hardware flavor (default: cpu-basic, ~$0.01/hr)
- `RESEARCH_JOB_TIMEOUT` — max job runtime before HF auto-cancels (default: 2h)
