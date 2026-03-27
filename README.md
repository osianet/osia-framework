# OSIA Framework — Open Source Intelligence Agency

An event-driven, multi-agent intelligence orchestration framework that automates the collection, analysis, and reporting of open-source intelligence.

OSIA models the structure of a real intelligence agency: a **Chief of Staff** orchestrator routes incoming requests to specialist AI **Desks**, each running a purpose-selected model with its own analytical mandate. Every task passes through a research loop, a RAG context injection, and a final synthesis stage before a finished report is delivered back via Signal.

---

## Intelligence Lifecycle

### 1. Ingress
Requests enter the system through two channels:
- **Signal Gateway** — operatives send URLs or queries to a Signal group; the gateway pushes them to the Redis task queue.
- **RSS Ingress** — a scheduled poller watches configured feeds and enqueues items automatically.

### 2. Research & Collection
The **Chief of Staff** (Venice `venice-uncensored`) reads the incoming task and kicks off a multi-turn research loop using MCP tools:

| Tool | Purpose |
|------|---------|
| Tavily | Live web search |
| Wikipedia | Encyclopaedic background |
| ArXiv | Academic / scientific papers |
| Semantic Scholar | Citation-graph research |
| YouTube (yt-dlp) | Video transcript extraction |

**Media Interception (PHINT):** If a social media link is received, a physical Moto g06 Android device connected via ADB records the screen for the duration of the video. Gemini Vision analyses the recording. Post metadata and comments are extracted first via `yt-dlp` (no phone required); ADB is only used as a fallback.

### 3. Background Research Worker
A local oneshot service (`osia-research-worker.timer`, every 2 hours) drains a separate `osia:research_queue` from Redis. It runs a multi-turn Venice AI research loop per topic, then chunks and embeds the results into the `osia_research_cache` Qdrant collection. Topic deduplication is enforced via TTL-keyed Redis entries (default 24h cooldown).

### 4. Desk Routing & RAG
After the research loop completes, the Chief of Staff selects the most appropriate desk. Each desk query is enriched with:
- **Per-desk RAG** — top-K results from that desk's Qdrant collection.
- **Cross-desk RAG** — always fans out across all registered collections, including the Epstein Files knowledge base. Falls back to the raw query when entity extraction produces nothing, so sensitive topics are never blocked from retrieval.
- **Research cache** — recent background research on relevant entities.
- **Real UTC timestamp** — injected at the top of every message; no tool call needed.

### 5. INTSUM Synthesis
The **Watch Floor** receives all desk reports and synthesises them into a final **INTSUM** (Intelligence Summary) — a structured briefing with sourced citations, reliability ratings, and an overall confidence assessment. The finished report is delivered back to the Signal group.

---

## Intelligence Desks

Desks are invoked directly via `DeskRegistry` — no middleware layer. Each desk has its own system prompt, model config, and Qdrant collection.

| Desk | Provider | Model | Purpose |
|------|----------|-------|---------|
| Geopolitical & Security | OpenRouter | `anthropic/claude-sonnet-4-6` | Statecraft, military, international relations |
| Cultural & Theological | Venice | `venice-uncensored` | Sociological & religious drivers (uncensored) |
| Science & Technology | OpenRouter | `anthropic/claude-sonnet-4-6` | Technical validation, R&D analysis |
| Human Intelligence | Venice | `venice-uncensored` | Behavioural profiling, network mapping (uncensored) |
| Finance & Economics | OpenRouter | `openai/gpt-4o-mini` | Markets, sanctions, internal cost auditing |
| Cyber Intelligence & Warfare | Venice | `mistral-31-24b` | Nation-state cyber ops, digital threats |
| The Watch Floor | OpenRouter | `anthropic/claude-sonnet-4-6` | INTSUM synthesis & Signal dispatch |

**Chief of Staff routing** uses Venice `venice-uncensored` — an uncensored model is used deliberately so that no query about a sensitive subject is misrouted due to guardrails.

**Uncensored routing policy:** Venice (`venice-uncensored`) is used for desk routing and the HUMINT/Cultural/Cyber desks. The Watch Floor uses Claude Sonnet 4.6 (capable on sensitive investigative topics) rather than a guardrailed model so final INTSUM synthesis is never sanitised. Entity extraction also uses Venice as primary.

All desks fall back to `openrouter/google/gemini-2.5-flash`. The Watch Floor falls back to `openrouter/google/gemini-2.5-pro`.

Venice desks (`venice-uncensored`, `mistral-31-24b`) call the Venice AI API directly — these models are not available via OpenRouter.

---

## Architecture

```
Signal / RSS
     │
     ▼
Redis Task Queue (osia:task_queue)
     │
     ▼
OSIA Orchestrator (Chief of Staff)
     ├── MCP Research Loop (Tavily, Wikipedia, ArXiv, Semantic Scholar, YouTube)
     ├── PHINT Media Pipeline (ADB → Moto g06 → Gemini Vision)  ← social media URLs
     ├── yt-dlp social metadata (comments, captions, stats)       ← primary for social posts
     ├── Entity Extractor → research jobs → osia:research_queue
     ├── Desk Router (Venice uncensored)
     │
     ▼
DeskRegistry
     ├── Geopolitical & Security  (OpenRouter / Claude Sonnet 4.6)
     ├── Cultural & Theological   (Venice / venice-uncensored)
     ├── Science & Technology     (OpenRouter / Claude Sonnet 4.6)
     ├── Human Intelligence       (Venice / venice-uncensored)
     ├── Finance & Economics      (OpenRouter / GPT-4o mini)
     └── Cyber Intelligence       (Venice / mistral-31-24b)
           │
           ▼
     The Watch Floor (OpenRouter / Claude Sonnet 4.6)
           │
           ▼
     Signal Group — INTSUM Briefing

Background:
osia-research-worker.timer (every 2h)
     └── Venice AI research loop → Qdrant osia_research_cache

Qdrant Collections (RAG namespaces):
     ├── per-desk collections (one per desk slug)
     ├── osia_research_cache  (background research worker output)
     └── epstein-files        (declassified government documents — see below)
```

---

## Tech Stack

- **Hardware:** Orange Pi 5 Plus (ARM64), Moto g06 (Android ADB gateway)
- **Runtime:** Python 3.12 (`uv`), Redis, Qdrant
- **Cloud AI:** Venice AI (`venice-uncensored`, `mistral-31-24b`), OpenRouter (Claude Sonnet 4.6, Gemini 2.5 Pro, GPT-4o mini, Gemini 2.5 Flash)
- **MCP Servers:** Tavily, Wikipedia, ArXiv, Semantic Scholar, YouTube (yt-dlp)
- **Protocol:** Signal (E2EE ingress/egress), ADB (media interception)

---

## Services

| Service | Description |
|---------|-------------|
| `osia-orchestrator` | Main task router and research loop |
| `osia-signal-ingress` | Signal group message handler |
| `osia-rss-ingress` | RSS feed poller |
| `osia-queue-api` | Redis queue HTTP wrapper |
| `osia-persona-daemon` | Ghost persona — social media activity on the ADB device |
| `osia-research-worker.timer` | Background research batch worker (every 2h) |
| `osia-daily-sitrep.timer` | Scheduled daily SITREP (07:00 UTC) |

ADB device access between the orchestrator and persona daemon is coordinated via a Redis lock (`osia:adb:lock`). The orchestrator holds priority; the persona yields on every ADB action while the lock is held.

---

## Setup

```bash
# 1. Clone and install Python dependencies
git clone https://github.com/osianet/osia-framework
cd osia-framework
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and fill in your API keys (see below)

# 3. Install and enable all systemd services
sudo ./install.sh

# 4. Start everything
sudo ./install.sh --start
```

The install script symlinks all service and timer unit files into `/etc/systemd/system/`, substituting the install path and user automatically. It skips services that are no longer in use (AnythingLLM, Kali bridge, etc.).

```
sudo ./install.sh              # install/enable only
sudo ./install.sh --start      # install, enable, and start
sudo ./install.sh --uninstall  # disable and remove all unit files
```

### Required environment variables

```
VENICE_API_KEY=
OPENROUTER_API_KEY=
GEMINI_API_KEY=
SIGNAL_PHONE=
SIGNAL_GROUP_ID=
ADB_DEVICE_MEDIA_INTERCEPT=
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
```

---

## Epstein Files Knowledge Base

A dedicated Qdrant collection (`epstein-files`) holds declassified government documents sourced from public releases aggregated on HuggingFace. These are indexed for cross-desk RAG retrieval on any query involving the Epstein network.

**Sources:**

| Dataset | HuggingFace ID | Size | Description |
|---------|---------------|------|-------------|
| Epstein Emails | `notesbymuneeb/epstein-emails` | ~5K threads | Structured email threads from House Oversight Committee release; parsed and OCR-corrected |
| House Oversight Docs | `tensonaut/EPSTEIN_FILES_20K` | ~25K files | Plain text documents from House Oversight Committee |
| Full Index | `theelderemo/FULL_EPSTEIN_INDEX` | ~8.5K rows | Grand jury materials, FBI documents, witness statements |
| DOJ Library | `Nikity/Epstein-Files` | 4.11M rows | Full DOJ Epstein Library — court filings, FBI reports, financial records |

**Ingestion:**

```bash
# Recommended run order (emails first — highest intel density)
uv run python scripts/ingest_epstein_files.py --dataset emails
uv run python scripts/ingest_epstein_files.py --dataset oversight
uv run python scripts/ingest_epstein_files.py --dataset index

# DOJ Library (large — run in passes)
uv run python scripts/ingest_epstein_files.py --dataset nikity --limit 50000
uv run python scripts/ingest_epstein_files.py --dataset nikity --limit 50000 --resume

# Test without writing anything
uv run python scripts/ingest_epstein_files.py --dataset emails --dry-run --limit 20
```

Entity extraction uses Venice (`venice-uncensored`) to ensure person names in sensitive documents are extracted without refusal. Each novel `Person` entity is automatically enqueued to the HUMINT research queue. Chunks are embedded via the HuggingFace Inference API and upserted with full provenance metadata (source, document type, online URL where available).

---

## Development

```bash
uv run ruff check src/          # lint
uv run pyright                  # type check
uv run pytest                   # tests
```

Never commit `.env` or `config/youtube_cookies.txt`.
