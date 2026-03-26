# Design Document: OSIA Core Rebuild

## Overview

This rebuild replaces AnythingLLM as the middleware layer between the OSIA orchestrator and its LLM models/vector store. The new architecture is a direct, pure-Python system that eliminates the HTTP→AnythingLLM→model indirection, gives OSIA full ownership of Qdrant (filtered search, cross-desk retrieval, metadata queries), and introduces an entity extraction pipeline that continuously enriches the knowledge base.

AnythingLLM may remain running for other purposes but OSIA will not depend on it after this rebuild.

### Key Goals

- Direct model dispatch via native SDKs (Gemini, Anthropic, OpenAI-compat httpx)
- Full Qdrant control: filtered search, cross-desk retrieval, deterministic upserts
- Entity extraction pipeline feeding the research queue automatically
- Desk configuration driven entirely by YAML files — no code changes to add/modify desks
- Preserve all existing capabilities: Signal I/O, ADB pipeline, MCP tools, RSS ingress, HF endpoint management, source tracking, citation audit, infographic generation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGRESS LAYER                               │
│  signal_ingress.py ──┐                                              │
│  rss_ingress.py ─────┼──► osia:task_queue (Redis)                  │
│  daily_sitrep.py ────┘                                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR                                  │
│  src/orchestrator.py                                                │
│                                                                     │
│  1. Media interception (YouTube/ADB)                                │
│  2. MCP research loop (Gemini tool-calling)                         │
│  3. Entity extraction  ──────────────────► osia:research_queue      │
│  4. Qdrant context injection                                        │
│  5. Desk routing (Gemini Chief-of-Staff)                            │
│  6. Desk_Registry.invoke()                                          │
│  7. Source audit + Signal reply                                     │
└──────────────┬──────────────────────────────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────────────────────────────────────────┐
│ Qdrant      │  │              DESK REGISTRY                       │
│ Store       │  │  src/desks/desk_registry.py                      │
│             │  │                                                  │
│ search()    │  │  Loads config/desks/*.yaml                       │
│ upsert()    │  │  Assembles system prompts                        │
│ cross_desk_ │  │  Dispatches to:                                  │
│   search()  │  │   ├─ Gemini SDK (google-genai)                   │
│ ensure_     │  │   ├─ Anthropic SDK                               │
│   collection│  │   ├─ OpenAI-compat httpx                         │
└─────────────┘  │   └─ HF Endpoint (via HFEndpointManager)         │
                 └──────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        [gemini-3-flash] [claude-sonnet] [Dolphin/Hermes]
        [gemini-3.1-pro] [gpt-5.4-mini]  [HF Endpoints]

┌─────────────────────────────────────────────────────────────────────┐
│                    BACKGROUND RESEARCH                              │
│  osia:research_queue ──► research_worker.py / research_batch.py    │
│                          └─► Qdrant (osia:research_cache)          │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Task Processing (Updated)

```
Signal/RSS message
      │
      ▼
[1] Media interception (YouTube transcript / ADB capture)
      │
      ▼
[2] MCP research loop (multi-turn Gemini tool-calling)
      │
      ▼
[3] Entity extraction (EntityExtractor.extract)
      │  └─► enqueue_research_jobs → osia:research_queue
      ▼
[4] Qdrant context injection
      │  ├─ QdrantStore.search(desk_collection, query, top_k=5)
      │  └─ QdrantStore.cross_desk_search(entity_names, top_k=3)
      ▼
[5] Chief-of-Staff routing (Gemini selects desk slug)
      │
      ▼
[6] DeskRegistry.invoke(slug, user_message, context_block)
      │  ├─ Assemble: system_prompt + mandate + citation_protocol
      │  ├─ Prepend: ## INTELLIGENCE CONTEXT block
      │  ├─ HFEndpointManager.ensure_ready() if hf_endpoint desk
      │  ├─ Primary model call (with retry on 503/429/timeout)
      │  └─ Fallback model call if primary exhausts retries
      ▼
[7] audit_report() + SourceTracker manifest
      │
      ▼
[8] Signal reply + optional infographic
```

---

## Components and Interfaces

### New Components

| File | Role |
|------|------|
| `config/desks/<slug>.yaml` | Per-desk configuration (model, Qdrant, tools, MCP servers) |
| `config/osia.yaml` | Global system config (default_desk, etc.) |
| `src/desks/desk_registry.py` | Loads desk configs, assembles prompts, dispatches to models |
| `src/intelligence/qdrant_store.py` | Direct Qdrant client: search, upsert, cross-desk, bootstrap |
| `src/intelligence/entity_extractor.py` | Named entity extraction + research job enqueuing |

### Preserved Components (unchanged)

| File | Role |
|------|------|
| `src/desks/hf_endpoint_manager.py` | HF Inference Endpoint lifecycle (wake, probe, timeout) |
| `src/intelligence/source_tracker.py` | Provenance tracking and citation audit |
| `src/gateways/mcp_dispatcher.py` | MCP session management and tool dispatch |
| `src/workers/research_worker.py` | Background research loop writing to Qdrant |
| `hf-spaces/research-worker/research_batch.py` | HF Jobs batch research script |

### Modified Components

| File | Change |
|------|--------|
| `src/orchestrator.py` | Remove AnythingLLMDesk; add EntityExtractor + QdrantStore + DeskRegistry |
| `src/gateways/rss_ingress.py` | Replace AnythingLLM ingest with QdrantStore.upsert + EntityExtractor |

### Removed Components

| File | Replacement |
|------|-------------|
| `src/desks/anythingllm_client.py` | `src/desks/desk_registry.py` |

---

## Data Models

### DeskConfig (dataclass)

```python
@dataclass
class ModelConfig:
    provider: str           # "gemini" | "anthropic" | "openai" | "hf_endpoint"
    model_id: str
    hf_endpoint_name: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096

@dataclass
class QdrantConfig:
    collection: str
    context_top_k: int = 5
    cross_desk_search: bool = True
    cross_desk_top_k: int = 3

@dataclass
class DeskConfig:
    slug: str
    name: str
    prompt_file: str
    prompt_text: str        # loaded from prompt_file at startup
    model_primary: ModelConfig
    model_fallback: ModelConfig | None
    qdrant: QdrantConfig
    tools: list[str]
    mcp_servers: list[str]
    entity_research_target: bool = True
```

### Desk YAML Config Schema

```yaml
# config/desks/geopolitical-and-security-desk.yaml
slug: geopolitical-and-security-desk
name: "Geopolitical & Security Desk"
prompt_file: templates/prompts/geopolitical-and-security-desk.txt

model:
  primary:
    provider: gemini
    model_id: gemini-3-flash
    temperature: 0.3
    max_tokens: 8192
  fallback:
    provider: gemini
    model_id: gemini-2.5-flash
    temperature: 0.3
    max_tokens: 8192

qdrant:
  collection: geopolitical_intel
  context_top_k: 5
  cross_desk_search: true
  cross_desk_top_k: 3

tools:
  - search_web
  - search_wikipedia
  - search_arxiv

mcp_servers:
  - tavily
  - wikipedia
  - arxiv

entity_research_target: true
```

HF endpoint desk example:

```yaml
# config/desks/cyber-intelligence-and-warfare-desk.yaml
slug: cyber-intelligence-and-warfare-desk
name: "Cyber Intelligence & Warfare Desk"
prompt_file: templates/prompts/cyber-intelligence-and-warfare-desk.txt

model:
  primary:
    provider: hf_endpoint
    model_id: Hermes-3-Llama-3.1-70B
    hf_endpoint_name: osia-hermes-70b
    temperature: 0.4
    max_tokens: 4096
  fallback:
    provider: gemini
    model_id: gemini-3-flash
    temperature: 0.3
    max_tokens: 4096

qdrant:
  collection: cyber_intel
  context_top_k: 5
  cross_desk_search: true
  cross_desk_top_k: 3

tools:
  - search_web
  - search_arxiv

mcp_servers:
  - tavily
  - arxiv

entity_research_target: true
```

### Global Config Schema (`config/osia.yaml`)

```yaml
# config/osia.yaml
default_desk: the-watch-floor   # fallback when routing returns unknown slug
```

### Desk-to-Model Mapping (all seven desks)

| Slug | Primary Provider | Primary Model | Fallback |
|------|-----------------|---------------|---------|
| geopolitical-and-security-desk | gemini | gemini-3-flash | gemini-2.5-flash |
| cultural-and-theological-intelligence-desk | hf_endpoint (osia-dolphin-r1-24b) | Dolphin3.0-R1-Mistral-24B | gemini-3-flash |
| science-technology-and-commercial-desk | anthropic | claude-sonnet-4-6 | gemini-3-flash |
| human-intelligence-and-profiling-desk | hf_endpoint (osia-dolphin-r1-24b) | Dolphin3.0-R1-Mistral-24B | gemini-3-flash |
| finance-and-economics-directorate | openai | gpt-5.4-mini | gemini-3-flash |
| cyber-intelligence-and-warfare-desk | hf_endpoint (osia-hermes-70b) | Hermes-3-Llama-3.1-70B | gemini-3-flash |
| the-watch-floor | gemini | gemini-3.1-pro-preview | gemini-3-flash |

### Qdrant Collection Schema

**Vector config**: 384 dimensions, Cosine distance (`sentence-transformers/all-MiniLM-L6-v2`)

**Payload fields** (all upserts must include):

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | Primary content field (matches Research_Worker format) |
| `desk` | str | Source desk slug |
| `topic` | str | Subject/title of the intel |
| `source` | str | Origin URL, feed URL, or tool name |
| `reliability_tier` | str | Admiralty scale: A–E |
| `timestamp` | str | ISO-8601 collection time |
| `entity_tags` | list[str] | Extracted entity names |
| `triggered_by` | str | What initiated this upsert (e.g. "rss_ingress", signal recipient) |

**Collections bootstrapped at startup**:

| Collection | Desk |
|-----------|------|
| `geopolitical_intel` | geopolitical-and-security-desk |
| `cultural_intel` | cultural-and-theological-intelligence-desk |
| `science_intel` | science-technology-and-commercial-desk |
| `human_intel` | human-intelligence-and-profiling-desk |
| `finance_intel` | finance-and-economics-directorate |
| `cyber_intel` | cyber-intelligence-and-warfare-desk |
| `watch_floor` | the-watch-floor |
| `collection_raw` | collection-directorate (RSS ingress) |
| `osia:research_cache` | Research_Worker output (existing) |

### SearchResult (dataclass)

```python
@dataclass
class SearchResult:
    text: str
    score: float
    collection: str
    metadata: dict
```

### Entity (dataclass)

```python
@dataclass
class Entity:
    name: str
    entity_type: str    # "Person" | "Organisation" | "Location" | "Event" | "Technology"
    context: str        # sentence/phrase where entity appeared
    source_desk: str
```

### Research Job Payload (pushed to `osia:research_queue`)

```python
{
    "job_id": "<uuid4>",
    "topic": "<entity name>",
    "desk": "<routed desk slug>",
    "priority": "normal",
    "directives_lens": True,
    "triggered_by": "<feed URL or signal recipient>"
}
```

Entity-to-desk routing:

| Entity Type | Target Desk |
|-------------|-------------|
| Person | human-intelligence-and-profiling-desk |
| Organisation | geopolitical-and-security-desk |
| Location | geopolitical-and-security-desk |
| Event | geopolitical-and-security-desk |
| Technology | science-technology-and-commercial-desk |

---

## Model Dispatch Logic

### Primary → Retry → Fallback

```
invoke(slug, user_message, context_block)
  │
  ├─ Assemble system_prompt (prompt_text + mandate + citation_protocol)
  ├─ If context_block: prepend "## INTELLIGENCE CONTEXT\n{context_block}"
  │
  ├─ If hf_endpoint provider:
  │    └─ HFEndpointManager.ensure_ready(slug)
  │         └─ If False: raise RuntimeError
  │
  ├─ PRIMARY attempt (up to 3 retries on HTTP 503, 15s delay):
  │    ├─ gemini: google-genai SDK, GenerateContentConfig(temperature, max_tokens)
  │    ├─ anthropic: anthropic SDK, messages.create(system=..., messages=[...])
  │    ├─ openai: httpx POST /v1/chat/completions (OpenAI API)
  │    └─ hf_endpoint: httpx POST {endpoint_url}/v1/chat/completions
  │
  ├─ On primary failure (5xx / 429 / timeout / SDK error after retries):
  │    ├─ If fallback defined:
  │    │    └─ Log warning (desk, failure reason, fallback model)
  │    │    └─ FALLBACK attempt (same assembled message + context)
  │    │         └─ On fallback failure: raise fallback exception (log both)
  │    └─ If no fallback: raise primary exception immediately
  │
  └─ Return response text + metadata {model_used: "primary"|"fallback", model_id: str}
```

### Prompt Assembly Order

```
[desk system prompt text]

## ANALYTICAL MANDATE
[mandate text from OSIA_DIRECTIVES_FILE]

--- CITATION PROTOCOL ---
[build_citation_protocol() output]
```

If `## ANALYTICAL MANDATE` already appears in the desk prompt file, the mandate block is not appended again.

---

## Qdrant Context Injection Flow

```
Before desk invocation:
  │
  ├─ desk_collection = DeskConfig.qdrant.collection
  ├─ top_k = DeskConfig.qdrant.context_top_k (default 5)
  │
  ├─ Try: results = await qdrant_store.search(desk_collection, query, top_k)
  │    └─ On QdrantError: log warning, results = []
  │
  ├─ If cross_desk_search enabled and entity_names extracted:
  │    └─ Try: cross = await qdrant_store.cross_desk_search(
  │                entity_names_query, cross_desk_top_k)
  │         └─ On QdrantError: log warning, cross = []
  │
  ├─ combined = results + cross  (deduplicated by point ID)
  │
  ├─ If combined is empty:
  │    └─ context_block = None  (invoke desk without context)
  │
  └─ Else: format context_block:
       "## INTELLIGENCE CONTEXT\n\n"
       for each result:
         "[{result.collection}] (Reliability: {metadata.reliability_tier}) "
         "{metadata.timestamp}\n{result.text}\n\n"
```

---

## RSS Ingress Updated Flow

```
For each new article:
  │
  ├─ [existing] Gemini summarization
  │
  ├─ [NEW] EntityExtractor.extract(ai_summary, "collection-directorate")
  │    └─ On failure: log warning, entity_tags = []
  │
  ├─ [NEW] EntityExtractor.enqueue_research_jobs(entities, triggered_by=feed_url)
  │
  ├─ [CHANGED] QdrantStore.upsert("collection_raw", intel_record, metadata={
  │      desk: "collection-directorate",
  │      topic: title,
  │      source: feed_url,
  │      reliability_tier: "B",
  │      timestamp: ISO-8601,
  │      entity_tags: [e.name for e in entities],
  │      triggered_by: "rss_ingress"
  │   })
  │
  ├─ [existing] Redis rpush(DAILY_DIGEST_KEY, intel_record)
  └─ [existing] Redis sadd(SEEN_KEY, link)
```

---

## New/Changed Dependencies

Add to `pyproject.toml`:

```toml
"anthropic>=0.40.0",        # Anthropic SDK (new)
"qdrant-client>=1.12.0",    # Qdrant Python client (new)
"pyyaml>=6.0.2",            # YAML config loading (new)
```

`qdrant-client` provides the typed Python client with async support. The existing `httpx`-based Qdrant calls in `research_worker.py` remain unchanged — the new `QdrantStore` uses `qdrant-client` for richer query support (filters, cross-collection search).

Install:
```bash
uv add anthropic qdrant-client pyyaml
```

---

## Migration Notes

### AnythingLLM

- `systemd/osia-anythingllm.service` can be stopped and disabled after migration is validated
- `src/desks/anythingllm_client.py` is deleted
- `ANYTHINGLLM_BASE_URL` and `ANYTHINGLLM_API_KEY` env vars become unused (can be removed from `.env` after validation)
- AnythingLLM's Qdrant collections used `vectorTag` as the collection name. The new system uses the same collection names (`geopolitical_intel`, `cultural_intel`, etc.) that AnythingLLM was already writing to — **no data migration required**

### Existing Qdrant Data

- All existing points written by AnythingLLM and the Research_Worker are immediately readable by `QdrantStore.search` — the `text` payload field is the same
- The new `QdrantStore.upsert` uses deterministic IDs (SHA-256 hash of content) — this means re-ingesting the same article will overwrite the existing point cleanly rather than creating duplicates
- AnythingLLM used `Xenova/all-MiniLM-L6-v2` (same model, different runtime) — embeddings are compatible

### Environment Variables

New variable added:
- `OSIA_DIRECTIVES_FILE` — path to analytical mandate file (defaults to `DIRECTIVES.md`)

No existing variables are removed or renamed. The `.env.example` should add `OSIA_DIRECTIVES_FILE=DIRECTIVES.md` as a documented optional.

### Systemd Services

No new services required. The orchestrator, RSS ingress, and research worker all run under their existing unit files. After migration:
- `osia-anythingllm.service` — can be disabled
- All other services — unchanged

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Desk YAML missing required key | `ValueError` at startup, identifies file + key |
| `prompt_file` path not found | `FileNotFoundError` at startup |
| Invalid `provider` value | `ValueError` at startup |
| `hf_endpoint` provider without `hf_endpoint_name` | `ValueError` at startup |
| `config/desks/` empty | `RuntimeError` at startup |
| Mandate file not found | `FileNotFoundError` at startup |
| `OSIA_DIRECTIVES_FILE` set to empty string | Log warning, no mandate appended |
| HF endpoint wake-up fails | `RuntimeError` from `DeskRegistry.invoke` |
| Primary model HTTP 503 | Retry up to 3× with 15s delay, then fallback |
| Primary model fails, fallback defined | Log warning, invoke fallback |
| Primary model fails, no fallback | Raise exception immediately |
| Both primary and fallback fail | Raise fallback exception, log both failures |
| Qdrant unavailable during context injection | Log warning, proceed without context block |
| HF embedding API fails | Log error, return zero vectors (384-dim), continue |
| `ensure_collection` fails | Log error + re-raise, orchestrator aborts startup |
| Entity extraction Gemini returns malformed JSON | Log warning, return empty list |
| RSS entity extraction fails | Log warning, upsert article without entity tags |
| Unknown desk slug in task | Route to `default_desk` (from `config/osia.yaml`), log warning |

---

## Testing Strategy

### Dual Testing Approach

Both unit tests and property-based tests are required. Unit tests cover specific examples, integration points, and error conditions. Property tests verify universal correctness across generated inputs.

**Property-based testing library**: `hypothesis` (Python). Each property test runs a minimum of 100 iterations.

Install:
```bash
uv add --dev hypothesis pytest pytest-asyncio
```

**Tag format for property tests**:
```python
# Feature: osia-core-rebuild, Property {N}: {property_text}
```

### Unit Tests

Focus areas:
- Startup validation: missing keys, invalid providers, missing files
- Provider dispatch: one test per provider type using mocked HTTP/SDK clients
- HF endpoint wake-up failure path
- Qdrant unavailability graceful degradation
- Entity extraction malformed JSON handling
- RSS ingress entity extraction failure path
- `config/osia.yaml` default_desk fallback

### Property Tests

Each correctness property below maps to a single property-based test.


---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

---

### Property 1: Desk registry round-trip

*For any* set of valid desk config dicts written to YAML files in a temp directory, loading the `DeskRegistry` from that directory and calling `list_slugs()` should return exactly the set of slugs that were written, and `get(slug)` should return a `DeskConfig` with matching fields for each slug.

**Validates: Requirements 1.1, 1.8, 1.9**

---

### Property 2: Schema validation rejects missing required keys

*For any* required top-level key in the desk config schema (`slug`, `name`, `prompt_file`, `model`, `qdrant`, `tools`, `mcp_servers`, `entity_research_target`), loading a config file that is missing that key should raise a `ValueError` identifying the file and the missing key.

**Validates: Requirements 1.2, 1.6, 13.2, 13.3**

---

### Property 3: Prompt file loading

*For any* desk config pointing to a temp file containing arbitrary text, the loaded `DeskConfig.prompt_text` should equal the contents of that file exactly.

**Validates: Requirements 1.5**

---

### Property 4: Context block is prepended under correct heading

*For any* non-empty context block string and any user message string, the assembled message passed to the model should contain the substring `## INTELLIGENCE CONTEXT` followed by the context block text, appearing before the user message content.

**Validates: Requirements 2.7, 5.3**

---

### Property 5: Fallback is invoked when primary fails

*For any* desk config that defines a fallback model, if the primary model invocation raises an exception (simulated via mock), the `DeskRegistry.invoke` call should invoke the fallback model with the same assembled message and context block, and the returned metadata should indicate `model_used = "fallback"`.

**Validates: Requirements 2.12, 2.13, 2.16**

---

### Property 6: Upsert produces deterministic point IDs

*For any* text string, calling `QdrantStore.upsert` twice with the same text should produce the same point ID both times (deterministic SHA-256 hash of content).

**Validates: Requirements 3.2**

---

### Property 7: ensure_collection is idempotent

*For any* collection name, calling `QdrantStore.ensure_collection` N times (N ≥ 1) should result in exactly one collection existing with the correct vector config — subsequent calls should be no-ops and not raise errors.

**Validates: Requirements 3.4, 9.2, 9.3**

---

### Property 8: SearchResult fields are always complete

*For any* search query against a collection containing at least one point, every `SearchResult` returned by `QdrantStore.search` should have non-None values for `text`, `score`, `collection`, and `metadata`.

**Validates: Requirements 3.10, 10.1, 10.2**

---

### Property 9: Entity pipeline fields are complete

*For any* entity returned by `EntityExtractor.extract`, the entity should have non-empty `name`, `entity_type`, `context`, and `source_desk` fields. For any such entity enqueued via `enqueue_research_jobs`, the resulting job payload should contain `job_id`, `topic`, `desk`, `priority`, `directives_lens`, and `triggered_by`.

**Validates: Requirements 4.3, 4.6**

---

### Property 10: Entity-to-desk routing is exhaustive

*For any* entity whose `entity_type` is one of `Person`, `Organisation`, `Location`, `Event`, or `Technology`, the desk assigned in the research job payload should match the entity-to-desk routing table exactly.

**Validates: Requirements 4.7**

---

### Property 11: Prompt assembly always includes mandate and citation protocol

*For any* desk config and any mandate text, the assembled system prompt passed to the model should contain the mandate text (under `## ANALYTICAL MANDATE`) and the citation protocol output. If the desk's prompt file already contains `## ANALYTICAL MANDATE`, the mandate text should appear exactly once.

**Validates: Requirements 8.2, 8.3, 8.4, 8.7**

---

### Property 12: Dynamic routing from registry

*For any* set of loaded desk configs, the orchestrator's valid routing targets should equal exactly `DeskRegistry.list_slugs()`. For any task payload whose desk slug is not in that set, the task should be routed to the configured `default_desk`.

**Validates: Requirements 6.13, 12.1, 12.3**

---

### Property 13: Invalid provider values are rejected

*For any* string that is not one of `gemini`, `anthropic`, `openai`, `hf_endpoint`, using it as `model.primary.provider` in a desk config should cause `DeskRegistry` startup to raise a `ValueError`.

**Validates: Requirements 13.2**

---

### Property 14: RSS upsert metadata is complete

*For any* RSS article processed by the updated `RSSIngress`, the metadata dict passed to `QdrantStore.upsert` should contain all required fields: `desk`, `topic`, `source`, `reliability_tier`, `timestamp`, `entity_tags`, and `triggered_by`.

**Validates: Requirements 14.2**
