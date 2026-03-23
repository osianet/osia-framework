# Tech Stack & Build System

## Runtime

- Python 3.11+ (`.python-version` specifies 3.11)
- Package manager: `uv` (modern Python package manager)
- Project config: `pyproject.toml` (PEP 621)
- Lock file: `uv.lock`

## Core Dependencies

- `google-genai` — Gemini API client (primary LLM, vision, and tool-calling)
- `redis` — async Redis client for task queue (`redis.asyncio`)
- `httpx` — async HTTP client (Signal API, AnythingLLM API)
- `websockets` — Signal WebSocket listener
- `fastapi` + `uvicorn` — HTTP bridges (phone bridge, MCP SSE bridge)
- `mcp[cli]` — Model Context Protocol client/server SDK
- `feedparser` + `beautifulsoup4` — RSS feed parsing
- `yt-dlp` — YouTube transcript extraction
- `python-dotenv` — environment variable loading
- `pandas`, `matplotlib`, `openpyxl` — data analysis and charting

## Infrastructure

- Redis (via Docker) — task queue and state tracking
- Signal CLI REST API (via Docker) — encrypted messaging gateway
- AnythingLLM — isolated LLM workspaces acting as intelligence desks
- Qdrant — vector database for intelligence storage
- systemd — service management (see `systemd/` directory)
- Nginx — reverse proxy with Let's Encrypt wildcard certs

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

# Run the MCP stdio-to-SSE bridge
uv run python src/mcp_bridge.py --command <cmd> --port <port> --name <name>

# Trigger a daily SITREP manually
uv run python -m src.cron.daily_sitrep

# Start infrastructure services
docker compose up -d
```

## Environment Configuration

All secrets and config live in `.env` (git-ignored). See `.env.example` for required variables. Key ones:

- `GEMINI_API_KEY` — Google Gemini API key
- `SIGNAL_SENDER_NUMBER` — registered Signal number
- `ANYTHINGLLM_API_KEY` — AnythingLLM workspace access
- `REDIS_URL` — Redis connection string
- `OSIA_BASE_DIR` — project root path
- `MCP_TOOLS_BASE` — parent directory for MCP tool installations
