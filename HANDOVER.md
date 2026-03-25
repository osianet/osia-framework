# OSIA Project Handover & Mission Context

## 🎯 The Mission
OSIA (Open Source Intelligence Agency) is a principled, counter-hegemonic intelligence framework founded on the **Decolonial & Socialist Mandate**. It prioritizes anti-imperialism, labor rights, and national sovereignty.

## 🏛️ Organizational Structure (AnythingLLM Desks)
- **Collection Directorate:** (NPU) Raw data dump and ingestion. Qdrant Tag: `collection_raw`.
- **Geopolitical & Security Desk:** (Cloud) Strategic forecasting and conflict analysis. Qdrant Tag: `geopolitical_intel`.
- **Cultural & Theological Desk:** (Cloud) Sociological and religious drivers. Qdrant Tag: `cultural_intel`.
- **Science & Tech Desk:** (Claude Sonnet 4.6) Technical accuracy and breakthroughs. Qdrant Tag: `science_intel`.
- **Human Intelligence Desk:** (Ollama - Dolphin) Uncensored behavioral profiling. Qdrant Tag: `human_intel`.
- **Finance & Economics Desk:** (GPT-5.4-mini) Market dynamics and internal auditing. Qdrant Tag: `finance_intel`.
- **Cyber Intelligence & Warfare Desk:** (Claude Sonnet 4.6) Digital infrastructure and threat analysis. Qdrant Tag: `cyber_intel`.
- **The Watch Floor:** (Gemini 3.1 Pro) Final INTSUM synthesis and dispatch. Qdrant Tag: `watch_floor`.

## 📡 Technical Architecture
- **Hardware:** Orange Pi 5 Plus (Master), Moto g06 (Android Gateway), RTX 3080 Ti (Local GPU).
- **Core:** Python 3.12 (`uv`), Redis (Task Queue), AnythingLLM (Context), Qdrant (Vectors).
- **Networking:** `*.osia.dev` subdomains with Let's Encrypt (Wildcard) and Nginx reverse proxy.
- **Ingress:** Signal ([REDACTED]), RSS Hourly Ingress, ADB Physical Screen Capture (PHINT).
- **Egress:** Signal Group Broadcasts (OSIA Briefings Group).
- **Remote Workers:** HuggingFace Spaces run async research workers that poll `osia:research_queue`, execute multi-source Gemini tool loops, and write chunked intel into Qdrant for RAG retrieval.

## 🛠️ Integrated MCP Tools & Custom Agent Skills
**MCP Tools** (`mcp.osia.dev`):
- **ArXiv & Semantic Scholar:** Academic/Technical search.
- **Wikipedia:** Factual baseline.
- **Tavily:** Real-time web search.
- **Time:** UTC precision for all reports.
- **YouTube:** Native `yt-dlp` transcription (Premium authenticated).

**Native AnythingLLM Skills** (`plugins/agent-skills`):
- **`osia-cyber-ip-intel`**: IP Geolocation & ASN lookup.
- **`osia-finance-stock-intel`**: Real-time ticker and market data.
- **`osia-stash-writer`**: Allows agents to write intelligence reports to a shared filesystem text file (`osia_shared_stash.txt`).
*(Note: Agents are automatically triggered by the python framework prepending `@agent` to incoming messages.)*

## 🔐 Credentials & Configs
- **Location:** `/home/ubuntu/osia-framework/.env` (Ignored by Git).
- **YouTube Cookies:** `/home/ubuntu/osia-framework/config/youtube_cookies.txt` (Premium).
- **Signal Number:** `[REDACTED]`.
- **Systemd:** 4 active services + 2 timers (Signal, Orchestrator, RSS, SITREP).
- **Management Scripts:** Run `scripts/update_prompts.py` to auto-sync directives, prompts, and model configs to AnythingLLM workspaces.

## ☁️ HuggingFace Inference Endpoints (Uncensored Cloud Models)

Dedicated HF Inference Endpoints provide on-demand access to uncensored models for desks that need unfiltered analysis. Endpoints use Text Generation Inference (TGI) with an OpenAI-compatible API and scale-to-zero billing.

**Provisioned endpoints:**
- `osia-dolphin-r1-24b` — `cognitivecomputations/Dolphin3.0-R1-Mistral-24B` on L4 GPU ($0.80/hr active, $0 idle). Serves HUMINT and Cultural desks. R1 reasoning variant gives auditable analytical chains.
- `osia-hermes-70b` — `NousResearch/Hermes-3-Llama-3.1-70B` on 2x A100 ($5.00/hr active, $0 idle). Serves Cyber desk. Purpose-built for reliable agentic tool-calling.

**Management:**
```bash
uv run python scripts/provision_hf_endpoints.py            # create endpoints
uv run python scripts/provision_hf_endpoints.py --status    # check status
uv run python scripts/provision_hf_endpoints.py --pause     # force-pause (stop billing)
uv run python scripts/provision_hf_endpoints.py --resume    # wake up paused endpoints
```

## 🔬 Queue API & Research Workers

The Queue API (`queue.osia.dev`, port 8098) is an authenticated HTTP wrapper around Redis that allows remote workers to push/pop jobs without direct Redis access. It exposes only the operations needed: push, pop, queue length, seen-check, seen-add.

Research workers run as HuggingFace Gradio Spaces (`hf-spaces/research-worker/`). Each worker:
1. Polls `osia:research_queue` via the Queue API
2. Runs a multi-turn Gemini research loop using direct HTTP tools (Tavily, Wikipedia, ArXiv, Semantic Scholar)
3. Chunks and embeds the output using `all-MiniLM-L6-v2` (matching AnythingLLM's embedder)
4. Writes results into Qdrant collection `osia:research_cache` with desk/topic metadata

**Deploying a research worker Space:**
1. Create a new Space at `huggingface.co/spaces/BadIdeasRory/osia-research-worker` (Gradio SDK)
2. Push the contents of `hf-spaces/research-worker/` to the Space repo
3. Set secrets in Space Settings: `QUEUE_API_URL`, `QUEUE_API_TOKEN`, `QUEUE_API_UA_SENTINEL`, `QDRANT_URL`, `QDRANT_API_KEY`, `GEMINI_API_KEY`, `TAVILY_API_KEY`, `HF_TOKEN`

**Pushing a research job manually:**
```bash
curl -X POST https://queue.osia.dev/queue/push \
  -A "osia-worker/1" \
  -H "Authorization: Bearer $QUEUE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"queue":"osia:research_queue","payload":{"topic":"Bolivia lithium 2025","desk":"finance-and-economics-directorate"}}'
```

## 🚀 Active Objectives
1. **Intelligence Dashboard:** Automating Telegram/Discord monitoring via the ADB gateway.
2. **Deep Dossiering:** Building multi-tool search workflows for target profiling.
3. **Internal Auditor:** Implementing cost/token logging for the Finance Desk.