# OSIA Project Handover & Mission Context

## 🎯 The Mission
OSIA (Open Source Intelligence Agency) is a principled, counter-hegemonic intelligence framework founded on the **Socialist Intelligence Mandate**. It prioritizes anti-imperialism, labor rights, and national sovereignty.

## 🏛️ Organizational Structure (AnythingLLM Desks)
- **Collection Directorate:** (NPU) Raw data dump and ingestion. Qdrant Tag: `collection_raw`.
- **Geopolitical & Security Desk:** (Cloud) Strategic forecasting and conflict analysis. Qdrant Tag: `geopolitical_intel`.
- **Cultural & Theological Desk:** (Cloud) Sociological and religious drivers. Qdrant Tag: `cultural_intel`.
- **Science & Tech Desk:** (Claude 3.5 Sonnet) Technical accuracy and breakthroughs. Qdrant Tag: `science_intel`.
- **Human Intelligence Desk:** (Ollama - Dolphin) Uncensored behavioral profiling. Qdrant Tag: `human_intel`.
- **Finance & Economics Desk:** (GPT-4o) Market dynamics and internal auditing. Qdrant Tag: `finance_intel`.
- **Cyber Intelligence & Warfare Desk:** (Claude 3.5 Sonnet) Digital infrastructure and threat analysis. Qdrant Tag: `cyber_intel`.
- **The Watch Floor:** (Gemini 2.5 Pro) Final INTSUM synthesis and dispatch. Qdrant Tag: `watch_floor`.

## 📡 Technical Architecture
- **Hardware:** Orange Pi 5 Plus (Master), Moto g06 (Android Gateway), RTX 3080 Ti (Local GPU).
- **Core:** Python 3.12 (`uv`), Redis (Task Queue), AnythingLLM (Context), Qdrant (Vectors).
- **Networking:** `*.osia.dev` subdomains with Let's Encrypt (Wildcard) and Nginx reverse proxy.
- **Ingress:** Signal ([REDACTED]), RSS Hourly Ingress, ADB Physical Screen Capture (PHINT).
- **Egress:** Signal Group Broadcasts (OSIA Briefings Group).

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

Dedicated HF Inference Endpoints provide on-demand access to uncensored models (Dolphin 3.0) for desks that need unfiltered analysis. Endpoints use Text Generation Inference (TGI) and expose an OpenAI-compatible API.

**How billing works:** Endpoints are configured with scale-to-zero. After 10 minutes of inactivity, replicas drop to 0 and billing stops. A cold start of ~2-5 minutes occurs on the next request. You provision once, then leave them — no need to tear down.

**Provisioned endpoints:**
- `osia-dolphin-8b` — Dolphin 3.0 8B on L4 GPU ($0.80/hr active, $0 idle). Lightweight fallback.
- `osia-dolphin-70b` — Dolphin 3.0 70B on 2x A100 ($5.00/hr active, $0 idle). Primary uncensored model.

**Target desks:** Human Intelligence, Cyber Intelligence, Cultural & Theological.

**Management:**
```bash
uv run python scripts/provision_hf_endpoints.py            # create endpoints
uv run python scripts/provision_hf_endpoints.py --status    # check status
uv run python scripts/provision_hf_endpoints.py --pause     # force-pause (stop billing)
uv run python scripts/provision_hf_endpoints.py --resume    # wake up paused endpoints
```

**AnythingLLM integration:** Configure target desks to use "Generic OpenAI" provider with:
- Base URL: the endpoint URL from `--status` output (append `/v1`)
- API Key: your `HF_TOKEN`
- Model: `cognitivecomputations/Dolphin3.0-Llama3.1-70B` (or 8B)

**Env vars:** `HF_TOKEN`, `HF_NAMESPACE` — see `.env.example`.

## 🚀 Active Objectives
1. **Intelligence Dashboard:** Automating Telegram/Discord monitoring via the ADB gateway.
2. **Deep Dossiering:** Building multi-tool search workflows for target profiling.
3. **Internal Auditor:** Implementing cost/token logging for the Finance Desk.