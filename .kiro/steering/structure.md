# Project Structure

```
osia-framework/
├── main.py                  # Entry point — boots the orchestrator
├── pyproject.toml           # Project metadata and dependencies (uv/PEP 621)
├── uv.lock                  # Locked dependency versions
├── docker-compose.yml       # Redis + Signal CLI REST API containers
├── DIRECTIVES.md            # Socialist Intelligence Mandate (analytical lens)
├── HANDOVER.md              # Project context and mission briefing
├── .env.example             # Template for required environment variables
│
├── src/
│   ├── orchestrator.py      # Central brain — Redis consumer, Gemini tool-calling loop, desk routing
│   ├── mcp_bridge.py        # Generic STDIO→SSE bridge for exposing MCP tools over HTTP
│   │
│   ├── agents/
│   │   └── social_media_agent.py   # Vision-driven Android phone agent (screenshot→Gemini→action loop)
│   │
│   ├── desks/
│   │   └── anythingllm_client.py   # HTTP client for AnythingLLM workspace chat and document ingestion
│   │
│   ├── gateways/
│   │   ├── signal_ingress.py       # WebSocket listener — Signal messages → Redis queue
│   │   ├── rss_ingress.py          # RSS feed poller — new articles → Collection Directorate
│   │   ├── phone_bridge.py         # FastAPI endpoint for remote ADB commands (screenshot, record)
│   │   ├── adb_device.py           # Low-level ADB wrapper (tap, swipe, type, screenshot, record)
│   │   ├── mcp_dispatcher.py       # MCP client — manages persistent sessions to STDIO MCP servers
│   │   └── queue_api.py            # Authenticated HTTP wrapper around Redis (for remote workers)
│   │
│   ├── workers/
│   │   └── research_worker.py      # Async research worker — polls queue, runs Gemini tool loop, writes Qdrant
│   │
│   └── cron/
│       └── daily_sitrep.py         # Pushes a daily SITREP task to the Redis queue
│
├── hf-spaces/
│   └── research-worker/            # HuggingFace Jobs batch script — synced to HF dataset repo via GitHub Actions
│       ├── research_batch.py       # Self-contained batch job: wakes endpoints, drains queue, writes Qdrant
│       └── requirements.txt        # Job dependencies (httpx, huggingface-hub)
│
└── systemd/                 # systemd unit files for all long-running services and timers
```

## Architecture Patterns

- **Event-driven queue**: All work flows through a Redis list (`osia:task_queue`). Ingress gateways push tasks, the orchestrator pops and processes them.
- **Research queue**: A second queue (`osia:research_queue`) feeds async research workers. Workers pop jobs, run multi-source Gemini tool loops, and write chunked results into Qdrant for RAG retrieval.
- **Gateway pattern**: Each external system (Signal, RSS, ADB phone, MCP tools, Queue API) has a dedicated gateway module in `src/gateways/`.
- **Desk abstraction**: Intelligence analysis desks are AnythingLLM workspaces accessed via a single HTTP client (`AnythingLLMDesk`). The orchestrator routes tasks to desks by slug name.
- **MCP tool dispatch**: The orchestrator declares Gemini function-call tools that map 1:1 to MCP server calls via `MCPDispatcher`. The research loop supports multi-turn tool calling.
- **Vision-action loop**: The social media agent uses a `screenshot → Gemini Vision → execute action` cycle to autonomously drive a physical Android phone without hardcoded coordinates.
- **Remote workers**: HuggingFace Jobs run `research_batch.py` on-demand. The Pi-side systemd timer checks queue depth every 2 hours and fires an HF Job when the threshold is met. The job wakes the appropriate HF Inference Endpoint (Dolphin R1 or Hermes 3), drains the queue, writes results to Qdrant, and exits. Billing stops on exit.

## Code Conventions

- Async-first: all I/O operations use `async`/`await`. Blocking calls (ADB subprocess) are wrapped with `asyncio.to_thread`.
- Logging via `logging` module with hierarchical names (`osia.orchestrator`, `osia.mcp`, etc.).
- Config via environment variables loaded with `python-dotenv`. No hardcoded secrets or paths.
- Type hints used throughout. Union types use `X | Y` syntax (Python 3.10+).
- Dataclasses for structured return types (`ActionResult`, `ScreenState`).
- No test suite currently exists in the repository.
