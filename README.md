# OSIA Framework — Open Source Intelligence Agency

An event-driven, multi-agent intelligence orchestration framework that automates the collection, analysis, and reporting of open-source intelligence.

OSIA models the structure of a real intelligence agency: a **Chief of Staff** orchestrator routes incoming requests to specialist AI **Desks**, each running a purpose-selected model with its own analytical mandate. Every task passes through a research loop, a RAG context injection, and a final synthesis stage before a finished report is delivered back via Signal.

---

## Intelligence Lifecycle

### 1. Ingress
Requests enter the system through three channels:
- **Signal Gateway** — operatives send URLs or queries to a Signal group; the gateway pushes them to the Redis task queue (`osia:task_queue`) for the orchestrator to process.
- **RSS Ingress** — a scheduled poller watches configured feeds. Each new article is deduplicated, cleaned, and summarised by Gemini, then upserted **directly into Qdrant** (`collection-directorate`) — bypassing the task queue and orchestrator entirely. Extracted entities are enqueued to `osia:research_queue` for the research worker. Summaries are also staged in Redis (`osia:rss:daily_digest`) for the 07:00 UTC SITREP.
- **Ingress API** — an authenticated HTTPS endpoint for programmatic submission from internal tooling or external callers (see [Ingress API](#ingress-api)).

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
A local oneshot service (`osia-research-worker.timer`, every 2 hours) drains a separate `osia:research_queue` from Redis. It runs a multi-turn Venice AI research loop per topic, then chunks and embeds the results **directly into the originating desk's Qdrant collection** (falling back to `osia_research_cache` if no desk is specified). Every chunk payload carries `entity_tags` (the researched topic plus significant tokens) and `ingested_at_unix` (unix timestamp) for temporal decay scoring. Topic deduplication is enforced via TTL-keyed Redis entries (default 72h cooldown).

### 4. Desk Routing & RAG
After the research loop completes, the Chief of Staff selects the most appropriate desk. Each desk query is enriched with three tiers of Qdrant context, all scored with a **70-day half-life temporal decay** so recent intelligence ranks above stale entries:

- **Per-desk RAG** — top-K results from that desk's own Qdrant collection (contains past INTSUMs, research worker output, and RSS summaries routed here).
- **Cross-desk RAG** — always fans out across all 17 registered collections; uses extracted entity names when available, falls back to the raw query so collections like `epstein-files` and `wikileaks-cables` are searched even on sensitive topics.
- **Boost collections** — each desk has a YAML-configured list of knowledge base collections that are searched with a *guaranteed* per-collection quota, independent of global cross-desk ranking. This ensures domain-specific KBs always contribute context:
  - Cyber desk boosts: `mitre-attack`, `cve-database`, `hackerone-reports`, `ttp-mappings`, `cti-reports`, `cybersecurity-attacks`
  - Geopolitical and HUMINT desks boost: `wikileaks-cables`, `epstein-files`, `collection-directorate`, `iran-israel-war-2026`
  - Finance and Cultural desks boost: `wikileaks-cables`, `collection-directorate`
  - Watch Floor boosts: `collection-directorate`, `osia_research_cache`, `wikileaks-cables`, `epstein-files`, `iran-israel-war-2026`
- **Real UTC timestamp** — injected at the top of every message; no tool call needed.

### 5. INTSUM Synthesis
The **Watch Floor** receives all desk reports and synthesises them into a final **INTSUM** (Intelligence Summary) — a structured briefing with sourced citations, reliability ratings, and an overall confidence assessment. The finished INTSUM is then **written back into the desk's Qdrant collection**, creating a self-reinforcing knowledge accumulation loop where every analysis enriches future RAG context. The finished report is also archived as a PDF and delivered back to the Signal group.

### 6. Daily SITREP
The 07:00 UTC daily SITREP (`osia-daily-sitrep.timer`) now runs **Qdrant pre-seeding** before building its prompt: it fans out four standing intelligence queries (geopolitics, cyber, finance, technology) across the full cross-desk store with temporal decay, pulling accumulated OSIA intelligence from all research archives and knowledge bases. This historical context is injected as an "ACCUMULATED OSIA INTELLIGENCE" block alongside the 24-hour RSS digest, so six months of embedded research informs every morning briefing — even on days with sparse RSS coverage.

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
Signal / Ingress API                    RSS Ingress (osia-rss-ingress.timer)
     │                                       │
     ▼                                       ├──▶ Qdrant: collection-directorate
Redis Task Queue (osia:task_queue)          ├──▶ Redis: osia:rss:daily_digest
     │                                       └──▶ Redis: osia:research_queue (entities)
     ▼
OSIA Orchestrator (Chief of Staff)
     ├── MCP Research Loop (Tavily, Wikipedia, ArXiv, Semantic Scholar, YouTube)
     ├── PHINT Media Pipeline (ADB → Moto g06 → Gemini Vision)  ← social media URLs
     ├── yt-dlp social metadata (comments, captions, stats)       ← primary for social posts
     ├── Entity Extractor → research jobs → osia:research_queue
     ├── Desk Router (Venice uncensored)
     ├── RAG Context Builder ─────────────────────────────────────────┐
     │     ├── [1] Desk primary collection (top-K, 70-day decay)      │
     │     ├── [2] Cross-desk fan-out (all 17 collections, top-3)     │
     │     └── [3] Boost collections (desk-specific KB quota,         │
     │               guaranteed hits from MITRE/CVE/WikiLeaks/etc.)   │
     │                                                                 │
     ▼                                                                 │
DeskRegistry ◀──────── INTELLIGENCE CONTEXT injected ────────────────┘
     ├── Geopolitical & Security  (OpenRouter / Claude Sonnet 4.6)
     │     boost: wikileaks-cables, epstein-files, collection-directorate,
     │            iran-israel-war-2026
     ├── Cultural & Theological   (Venice / venice-uncensored)
     │     boost: etymology-database, collection-directorate, wikileaks-cables
     ├── Science & Technology     (OpenRouter / Claude Sonnet 4.6)
     │     boost: cve-database, mitre-attack, collection-directorate
     ├── Human Intelligence       (Venice / venice-uncensored)
     │     boost: epstein-files, wikileaks-cables, iran-israel-war-2026
     ├── Finance & Economics      (OpenRouter / GPT-4o mini)
     │     boost: wikileaks-cables, epstein-files, collection-directorate
     └── Cyber Intelligence       (Venice / mistral-31-24b)
           boost: mitre-attack, cve-database, hackerone-reports,
                  ttp-mappings, cti-reports, cybersecurity-attacks
           │
           ▼
     The Watch Floor (OpenRouter / Claude Sonnet 4.6)
           boost: collection-directorate, osia_research_cache,
                  wikileaks-cables, epstein-files,
                  iran-israel-war-2026
           │
           ├──▶ Signal Group — INTSUM Briefing
           ├──▶ PDF Archive (reports/)
           └──▶ Qdrant write-back (analysis → desk collection)

Background:
osia-research-worker.timer (every 2h)
     └── Venice AI research loop → Qdrant desk collection
           (entity_tags + ingested_at_unix stamped on every chunk)

osia-daily-sitrep.timer (07:00 UTC)
     ├── Drain Redis osia:rss:daily_digest
     ├── Qdrant pre-seed: 4 standing queries × cross-desk search (70-day decay)
     └── Push enriched SITREP task → osia:task_queue

Qdrant Collections (RAG namespaces):
     ├── per-desk collections (one per desk slug — primary + research worker + INTSUM write-back)
     ├── collection-directorate (RSS summaries, routing fallback)
     ├── osia_research_cache   (fallback for unrouted research jobs)
     ├── epstein-files         (declassified government documents)
     ├── wikileaks-cables      (124K US diplomatic cables 1966–2010)
     ├── cybersecurity-attacks (13K global cyber incidents)
     ├── hackerone-reports     (12.6K disclosed bug bounty reports)
     ├── mitre-attack          (~1,400 ATT&CK techniques, groups, malware, mitigations)
     ├── cti-reports           (9.7K NER-annotated CTI report texts)
     ├── ttp-mappings          (20.7K threat report snippets with ATT&CK labels)
     ├── cve-database          (280K NVD CVEs 1999–2025)
     ├── cti-bench             (5.6K analyst benchmark scenarios)
     └── etymology-database    (historical word/term/concept origins)
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
| `osia-ingress-api` | Authenticated HTTPS ingress for programmatic task submission |
| `osia-queue-api` | Redis queue HTTP wrapper |
| `osia-status-api` | Service health, metrics, and log access |
| `osia-persona-daemon` | Ghost persona — social media activity on the ADB device |
| `osia-research-worker.timer` | Background research batch worker (every 2h) |
| `osia-daily-sitrep.timer` | Scheduled daily SITREP (07:00 UTC) |

ADB device access between the orchestrator and persona daemon is coordinated via a Redis lock (`osia:adb:lock`). The orchestrator holds priority; the persona yields on every ADB action while the lock is held.

---

## Weekly Department Briefings

Each intelligence desk produces a weekly video briefing — a narrated slide deck presented by a fictional department head character. The pipeline runs every Monday at 08:00 UTC and generates both landscape (16:9, YouTube) and portrait (9:16, Shorts/Reels) formats.

### Pipeline

1. **Intel retrieval** — queries Qdrant for the past 7 days of intelligence from each desk's collection
2. **Slide generation** — the desk's LLM produces a structured JSON briefing (title slide, 3–5 content slides, closing slide) with full narration scripts
3. **Background images** — Venice AI (`flux-2-pro`) generates a dark, cinematic background image per slide, composited behind content with a gradient overlay for text legibility
4. **Narration** — ElevenLabs TTS renders each slide's narration script to audio, with per-desk voice IDs matching each department head character
5. **Video assembly** — ffmpeg combines slide PNGs + audio into MP4 segments, then concatenates them
6. **YouTube upload** — landscape videos upload to YouTube as background tasks (uploads run concurrently with subsequent desk generation)

### Department Head Portraits

Each desk has a `portrait_prompt` in its YAML config describing the department head's appearance. Portraits are generated once via Venice AI and reused across all future briefings, appearing on the title slide.

```bash
# Generate all portraits
uv run python scripts/generate_portraits.py

# Single desk
uv run python scripts/generate_portraits.py --desk cyber-intelligence-and-warfare-desk

# Regenerate existing portraits
uv run python scripts/generate_portraits.py --force
```

Portraits are saved to `assets/portraits/<desk-slug>.png`.

### YouTube Upload Setup

Briefing videos can be automatically uploaded to YouTube. This requires a one-time OAuth 2.0 consent flow:

1. Create OAuth 2.0 credentials in [Google Cloud Console](https://console.cloud.google.com/apis/credentials) (Desktop app type)
2. Enable the YouTube Data API v3
3. Download the client secret JSON → `config/youtube_client_secret.json`
4. Run the auth flow:
   ```bash
   uv run python -m src.intelligence.youtube_uploader --auth
   ```
5. Set `youtube.enabled: true` in `config/weekly_briefing.yaml`

Uploads default to `unlisted`. Both `youtube_client_secret.json` and `youtube_token.json` are git-ignored.

### Configuration

All briefing settings live in `config/weekly_briefing.yaml`:
- Schedule, lookback window, slide timing
- ElevenLabs TTS model and voice parameters
- Venice image generation model and toggle
- YouTube upload privacy, category, and tags
- Video encoding (hardware-accelerated via Rockchip MPP on the Orange Pi)

Per-desk voice, persona, and portrait prompt config lives in `config/desks/<desk-slug>.yaml` under the `briefing:` block.

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
| House Oversight Docs | `theelderemo/epstein-files-nov-2025` | ~25.8K files | Plain text documents from House Oversight Committee (Nov 2025 release) |
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

## Cybersecurity Attacks Knowledge Base

A dedicated Qdrant collection (`cybersecurity-attacks`) holds 13,407 documented global cybersecurity incidents sourced from the public `vinitvek/cybersecurityattacks` HuggingFace dataset. Incidents span 163 countries and 22 industry sectors and cover nation-state operations, criminal ransomware, hacktivist campaigns, and more. The collection is included in cross-desk RAG retrieval, enriching any query that touches threat actors, TTPs, or targeted sectors.

**Dataset:** [`vinitvek/cybersecurityattacks`](https://huggingface.co/datasets/vinitvek/cybersecurityattacks) — 13,407 rows, ~6 MB, Unlicense

**Fields indexed per incident:**

| Field | Description |
|-------|-------------|
| `affected_organization` | Victim organisation |
| `affected_country` | Target country (163 unique) |
| `affected_industry` | Sector (22 categories: Healthcare, Finance, etc.) |
| `event_type` / `event_subtype` | Attack classification + method (86 subtypes) |
| `motive` | Financial, Political-Espionage, Sabotage, Protest, etc. |
| `actor` / `actor_type` / `actor_country` | Threat actor name, classification, and origin |
| `description` | Incident narrative |
| `source_url` | Reference source |

**Ingestion:**

```bash
# Full ingest (~13K records, completes in a few minutes)
uv run python scripts/ingest_cybersecurity_attacks.py

# Test without writing anything
uv run python scripts/ingest_cybersecurity_attacks.py --limit 100 --dry-run

# Skip enqueuing threat actors to the Cyber desk research queue
uv run python scripts/ingest_cybersecurity_attacks.py --skip-actors

# Resume an interrupted run
uv run python scripts/ingest_cybersecurity_attacks.py --resume
```

Each novel threat actor (`Nation-State`, `Criminal`, `Hacktivist`, `Terrorist`) is automatically enqueued to the **Cyber Intelligence & Warfare desk** research queue for follow-up investigation. Actors are Redis-deduplicated across runs. Embedding and upserts are idempotent — re-running the script is safe.

---

## HackerOne Disclosed Reports Knowledge Base

A dedicated Qdrant collection (`hackerone-reports`) holds 12,618 publicly disclosed bug bounty reports from the [`Hacker0x01/hackerone_disclosed_reports`](https://huggingface.co/datasets/Hacker0x01/hackerone_disclosed_reports) dataset. Reports span a wide range of weakness types (XSS, SSRF, auth bypass, etc.), affected organisations, and severity levels. The collection enriches Cyber desk queries with real-world vulnerability context — past disclosures, affected assets, and exploitability narratives.

**Dataset:** `Hacker0x01/hackerone_disclosed_reports` — 10,094 train / 1,262 validation / 1,262 test rows

**Fields indexed per report:**

| Field | Description |
|-------|-------------|
| `title` | Vulnerability report title |
| `weakness` | CWE weakness classification (e.g. Improper Authorization, SSRF) |
| `team_handle` | Target organisation's HackerOne handle |
| `asset_identifier` | Affected asset (domain, IP, app) |
| `max_severity` | Declared severity (critical / high / medium / low) |
| `substate` | Report outcome (resolved, duplicate, informative, not-applicable) |
| `has_bounty` | Whether a bounty was awarded |
| `reporter` | Researcher username |
| `disclosed_at` | Public disclosure date |
| `vulnerability_information` | Full technical vulnerability narrative |

**Ingestion:**

```bash
# Full ingest — all three splits (~12.6K reports)
uv run python scripts/ingest_hackerone_reports.py

# Single split
uv run python scripts/ingest_hackerone_reports.py --splits train

# Skip reports with no vulnerability text (visibility=no-content)
uv run python scripts/ingest_hackerone_reports.py --skip-no-content

# Test without writing anything
uv run python scripts/ingest_hackerone_reports.py --limit 100 --dry-run

# Resume an interrupted run
uv run python scripts/ingest_hackerone_reports.py --resume
```

Upserts are idempotent (deterministic point IDs from report ID). Re-running the script is safe and will update any changed payloads.

---

## WikiLeaks Cables Knowledge Base

A dedicated Qdrant collection (`wikileaks-cables`) holds 124,747 US diplomatic cables spanning 1966–2010, sourced from the WikiLeaks cable release. Cables are chunked with a paragraph-aware splitter (1,500-char chunks, 225-char overlap), with the cable header (ID, origin, classification, date, references) prepended to every chunk so each embedding is fully self-contained.

**Dataset:** [`fn5/wikileaks-cables`](https://huggingface.co/datasets/fn5/wikileaks-cables) — 124,747 rows, ~1 GB Parquet, MIT

**Fields indexed per cable:**

| Field | Description |
|-------|-------------|
| `cable_id` | Cable identifier (e.g. `66BUENOSAIRES2481`) |
| `origin` | Originating embassy or consulate (272 unique origins) |
| `classification` | UNCLASSIFIED / CONFIDENTIAL / SECRET / TOP SECRET |
| `datetime` | Cable timestamp |
| `references` | Referenced cable IDs |
| `body` | Full cable text (chunked if > 1,500 chars) |

**Ingestion:**

```bash
# Full ingest — all 124K cables
uv run python scripts/ingest_wikileaks_cables.py --resume

# Classified cables only
uv run python scripts/ingest_wikileaks_cables.py --classification SECRET CONFIDENTIAL --resume

# Enqueue SECRET/CONFIDENTIAL cables to Geopolitical desk research queue
uv run python scripts/ingest_wikileaks_cables.py --classification SECRET CONFIDENTIAL --enqueue-classified

# Test without writing anything
uv run python scripts/ingest_wikileaks_cables.py --limit 500 --dry-run
```

Long bodies are chunked — expect significantly more Qdrant points than source records. `--enqueue-classified` is rate-controlled: each cable is TTL-deduplicated in Redis (30-day window) to avoid re-enqueueing across runs.

---

## MITRE ATT&CK Knowledge Base

A dedicated Qdrant collection (`mitre-attack`) holds the full MITRE ATT&CK knowledge base downloaded directly from the [mitre-attack/attack-stix-data](https://github.com/mitre-attack/attack-stix-data) GitHub repository (Apache 2.0). All entity types — techniques, threat actor groups, software (malware/tools), and mitigations — are ingested with relationship data resolved: group documents list their attributed techniques and malware; technique documents list known groups and software that use them.

**Coverage (enterprise domain):**

| Entity Type | Count | Description |
|-------------|-------|-------------|
| Techniques | ~700 | ATT&CK techniques and sub-techniques with tactic, platform, detection, and data source metadata |
| Groups | ~140 | APT and criminal group profiles with aliases, descriptions, and attributed TTPs |
| Software | ~600 | Malware families and offensive tools with platform and usage attribution |
| Mitigations | ~40 | M1xxx mitigation controls |

**Ingestion:**

```bash
# Enterprise domain only (default)
uv run python scripts/ingest_mitre_attack.py

# All three domains
uv run python scripts/ingest_mitre_attack.py --domains enterprise mobile ics

# Use locally pre-downloaded JSON files (skips GitHub download)
uv run python scripts/ingest_mitre_attack.py --local-dir /tmp/attack-stix

# Test without writing anything
uv run python scripts/ingest_mitre_attack.py --dry-run
```

The script downloads each domain's STIX bundle via HTTPS, builds the full relationship graph in memory, then flushes all entity documents to Qdrant in a single pass. Re-running is safe — point IDs are deterministic hashes of the STIX object ID.

---

## CTI Reports Knowledge Base

A dedicated Qdrant collection (`cti-reports`) holds 9,732 expert NER-annotated CTI (Cyber Threat Intelligence) report texts from the [`mrmoor/cyber-threat-intelligence-splited`](https://huggingface.co/datasets/mrmoor/cyber-threat-intelligence-splited) dataset. Each point stores the raw CTI text augmented with a structured entity summary block, and carries per-type metadata arrays for Qdrant payload filtering.

**Dataset:** `mrmoor/cyber-threat-intelligence-splited` — 6,810 train / 1,460 validation / 1,460 test, CC BY 4.0

**Entity types extracted per document:**

| Category | Types |
|----------|-------|
| Semantic | `malware`, `threat-actor`, `attack-pattern`, `identity`, `campaign` |
| IOC | `IPV4`, `DOMAIN`, `SHA1`, `SHA256`, `MD5`, `FILEPATH`, `EMAIL`, `CVE`, `URL` |

**Ingestion:**

```bash
# All splits
uv run python scripts/ingest_cti_reports.py --resume

# Training split only
uv run python scripts/ingest_cti_reports.py --splits train

# Test without writing anything
uv run python scripts/ingest_cti_reports.py --limit 200 --dry-run
```

---

## TTP Mappings Knowledge Base

A dedicated Qdrant collection (`ttp-mappings`) holds 20,736 threat report text snippets from the [`tumeteor/Security-TTP-Mapping`](https://huggingface.co/datasets/tumeteor/Security-TTP-Mapping) dataset, each labelled with one or more MITRE ATT&CK technique IDs by domain experts. ATT&CK technique IDs are stored as a payload array (`ttp_ids`) on each point, enabling Qdrant payload filter queries such as "all documents referencing T1566".

**Dataset:** `tumeteor/Security-TTP-Mapping` — 14,900 train / 2,630 validation / 3,170 test, CC BY 4.0

**Ingestion:**

```bash
# All splits
uv run python scripts/ingest_ttp_mappings.py --resume

# Training split only
uv run python scripts/ingest_ttp_mappings.py --splits train

# Test without writing anything
uv run python scripts/ingest_ttp_mappings.py --limit 500 --dry-run
```

---

## CVE Database Knowledge Base

A dedicated Qdrant collection (`cve-database`) holds 280,694 NVD CVE records from 1999 through May 2025, sourced from the [`stasvinokur/cve-and-cwe-dataset-1999-2025`](https://huggingface.co/datasets/stasvinokur/cve-and-cwe-dataset-1999-2025) dataset. Each point stores the CVE ID, CVSS scores, CWE classification, severity, and full NVD description. CVSS severity is stored as both a string and an ordinal integer (`severity_score`) for range filtering.

**Dataset:** `stasvinokur/cve-and-cwe-dataset-1999-2025` — 280,694 rows, ~103 MB CSV, CC0-1.0 (public domain)

**Fields indexed per CVE:**

| Field | Description |
|-------|-------------|
| `cve_id` | CVE identifier (e.g. `CVE-2021-44228`) |
| `severity` | CRITICAL / HIGH / MEDIUM / LOW / NONE |
| `severity_score` | Ordinal: CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1, NONE=0 |
| `cvss_v4` / `cvss_v3` / `cvss_v2` | CVSS scores |
| `cwe_id` | CWE weakness classification |
| `text` | Full NVD description |

**Ingestion:**

```bash
# Full ingest — all 280K CVEs (use --resume; checkpoints every 2,000 records)
uv run python scripts/ingest_cve_database.py --resume

# High-priority CVEs first
uv run python scripts/ingest_cve_database.py --severity CRITICAL HIGH --resume

# Test without writing anything
uv run python scripts/ingest_cve_database.py --limit 500 --dry-run
```

At 280K records this is the largest ingest job. Recommended approach: run CRITICAL/HIGH first to populate the high-value subset immediately, then run the full ingest in a background pass.

---

## CTI-Bench Knowledge Base

A dedicated Qdrant collection (`cti-bench`) holds 5,610 analyst-grade CTI evaluation scenarios from the [`AI4Sec/cti-bench`](https://huggingface.co/datasets/AI4Sec/cti-bench) dataset (CC BY-NC-SA 4.0 — internal use only). Each of the six task configurations produces a different document format optimised for RAG retrieval.

**Task configurations:**

| Task | Rows | Format | Intel value |
|------|------|--------|-------------|
| `cti-ate` | 60 | Malware report → ATT&CK technique IDs (ground truth) | Highest — real reports with expert TTP labels |
| `cti-taa` | 50 | Threat report → threat actor attribution | Highest — actor attribution examples |
| `cti-mcq` | 2,500 | Cybersecurity knowledge MCQ with answer | Broad — Q&A knowledge base |
| `cti-rcm` | 1,000 | CVE description → CWE mapping (2024 CVEs) | CVE classification context |
| `cti-rcm-2021` | 1,000 | CVE description → CWE mapping (2021 CVEs) | CVE classification context |
| `cti-vsp` | 1,000 | Vulnerability description → CVSS severity | Severity prediction context |

**Ingestion:**

```bash
# High-value tasks first
uv run python scripts/ingest_cti_bench.py --tasks cti-ate cti-taa

# All tasks
uv run python scripts/ingest_cti_bench.py

# Test without writing anything
uv run python scripts/ingest_cti_bench.py --dry-run
```

---

## Ingress API

A dedicated FastAPI service (`src/gateways/ingress_api.py`) provides an authenticated HTTPS endpoint for submitting intelligence tasks programmatically — from internal scripts, automation pipelines, or any external caller with a valid token.

**Authentication:** Every request (except `/health`) requires a `Bearer` token in the `Authorization` header and a correct `User-Agent` sentinel. Wrong UA returns 404 so the service is invisible to scanners.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Submit an intelligence query or URL for immediate processing → `osia:task_queue` |
| `POST` | `/research` | Enqueue a deep-research topic → `osia:research_queue` (24h TTL dedup) |
| `GET` | `/queue/status` | Current depth of both queues |
| `GET` | `/health` | Liveness probe (UA-gated, no auth required) |

**`POST /ingest` body:**

```json
{
  "query": "Latest developments in hypersonic glide vehicles",
  "label": "automation",
  "priority": "normal"
}
```

- `query` — the intelligence question or URL to analyse (required)
- `label` — caller identifier, alphanumeric (optional; defaults to `external`). Appears as the task source in logs and Qdrant metadata as `api:<label>`.
- `priority` — `"normal"` (default, appended to queue) or `"high"` (prepended, runs ahead of backlog)

**`POST /research` body:**

```json
{
  "topic": "Iranian drone proliferation networks",
  "label": "daily-brief"
}
```

Topics are MD5-deduplicated against a TTL key in Redis — duplicate submissions within the cooldown window (`RESEARCH_COOLDOWN_HOURS`, default 24h) are silently skipped and return `"queued": false`.

**Token management:**

```bash
uv run python scripts/manage_ingress_token.py           # show token + example curl commands
uv run python scripts/manage_ingress_token.py --rotate  # generate and save a new token
```

After rotating, restart the service: `sudo systemctl restart osia-ingress-api.service`

**Environment variables:**

```
INGRESS_API_TOKEN=          # Bearer token (generate with manage_ingress_token.py --rotate)
INGRESS_API_UA_SENTINEL=osia-ingress/1
INGRESS_API_PORT=8097
```

---

## Development

```bash
uv run ruff check src/          # lint
uv run pyright                  # type check
uv run pytest                   # tests
```

Never commit `.env`, `config/youtube_client_secret.json`, or `config/youtube_token.json`.
