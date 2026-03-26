# Requirements Document

## Introduction

This document specifies the requirements for a major architectural rebuild of the OSIA intelligence framework. The rebuild replaces AnythingLLM as the middleware layer between the orchestrator and the LLM models/Qdrant vector store with a direct, pure-Python architecture. The new architecture eliminates the HTTP→AnythingLLM→model indirection, gives OSIA full control over Qdrant (filtered search, cross-desk retrieval, metadata queries), introduces an entity extraction pipeline that continuously enriches the knowledge base, and preserves all existing capabilities (Signal I/O, ADB pipeline, MCP tools, RSS ingress, HF endpoint management, source tracking, citation audit, infographic generation).

AnythingLLM may remain running for other purposes but OSIA must not depend on it after this rebuild.

---

## Glossary

- **Orchestrator**: `src/orchestrator.py` — the central Redis consumer and task router ("Chief of Staff").
- **Desk**: A named intelligence analysis unit with a defined system prompt, model provider, Qdrant collection, and tool access list.
- **Desk_Config_File**: A YAML file under `config/desks/<slug>.yaml` that fully describes a single desk — its model, Qdrant settings, tools, MCP servers, and HF endpoint configuration.
- **Desk_Registry**: `src/desks/desk_registry.py` — the new Python-native desk configuration and dispatch layer replacing `AnythingLLMDesk`. Loads all desks from `config/desks/`.
- **DeskConfig**: A Python dataclass populated by loading a Desk_Config_File.
- **Qdrant_Store**: `src/intelligence/qdrant_store.py` — the direct Qdrant client replacing AnythingLLM's vector store abstraction.
- **Entity_Extractor**: `src/intelligence/entity_extractor.py` — the named-entity extraction pipeline that converts incoming intel into research jobs.
- **HFEndpointManager**: `src/desks/hf_endpoint_manager.py` — the existing HuggingFace Inference Endpoint lifecycle manager (preserved unchanged).
- **Research_Worker**: `src/workers/research_worker.py` — the existing async research worker that polls `osia:research_queue` and writes to Qdrant.
- **Research_Queue**: Redis list `osia:research_queue` — the queue of background research jobs.
- **Task_Queue**: Redis list `osia:task_queue` — the primary task queue consumed by the Orchestrator.
- **Intelligence_Context**: A block of top-K Qdrant search results injected into a desk prompt before model invocation.
- **Analytical_Mandate**: The operator-defined analytical lens loaded from the file specified by `OSIA_DIRECTIVES_FILE` (defaults to `DIRECTIVES.md`). Appended to every desk's system prompt. Can be any plain Markdown file — the current OSIA deployment uses the Decolonial & Socialist Mandate.
- **HF_Endpoint**: A HuggingFace Inference Endpoint hosting an uncensored model (Dolphin R1 24B or Hermes 3 70B) with scale-to-zero billing.
- **Embedding_Model**: `sentence-transformers/all-MiniLM-L6-v2` via the HuggingFace Inference API — the shared embedding model used by all Qdrant writes.
- **EARS**: Easy Approach to Requirements Syntax — the pattern language used for all acceptance criteria.
- **Collection_Directorate**: The desk responsible for raw data acquisition and ingestion (slug: `collection-directorate`).
- **Watch_Floor**: The desk responsible for final INTSUM synthesis (slug: `the-watch-floor`).
- **INTSUM**: Intelligence Summary — a synthesised report produced by the Watch Floor.
- **PHINT**: Physical Intelligence — ADB-captured media from a physical Android device.
- **MCP**: Model Context Protocol — the tool-calling protocol used by the Orchestrator's research loop.
- **Source_Tracker**: `src/intelligence/source_tracker.py` — the existing provenance and citation audit module (preserved unchanged).

---

## Requirements

### Requirement 1: Desk Configuration Files

**User Story:** As an OSIA operator, I want each intelligence desk to be fully described by a YAML configuration file, so that I can add, modify, or reconfigure desks without touching Python code.

#### Acceptance Criteria

1. EACH desk SHALL be defined by a YAML file at `config/desks/<slug>.yaml`. The Desk_Registry SHALL discover and load all files matching this pattern at startup.
2. EACH Desk_Config_File SHALL contain the following top-level keys:
   - `slug` (str) — unique identifier matching the filename stem
   - `name` (str) — human-readable display name
   - `prompt_file` (str) — path relative to project root of the desk's system prompt (e.g. `templates/prompts/geopolitical-and-security-desk.txt`)
   - `model` (object) — model configuration block (see criterion 3)
   - `qdrant` (object) — Qdrant configuration block (see criterion 4)
   - `tools` (list[str]) — list of MCP tool names this desk may call (e.g. `["search_web", "search_wikipedia"]`)
   - `mcp_servers` (list[str]) — list of MCP server names this desk has access to (e.g. `["tavily", "wikipedia", "arxiv"]`)
   - `entity_research_target` (bool) — whether entities extracted from this desk's output should be enqueued for background research
3. THE `model` block SHALL contain:
   - `primary` (object) — the primary model configuration (see criterion 3a)
   - `fallback` (object, optional) — the fallback model configuration, used when the primary fails (same schema as primary)

   3a. Each model configuration object (primary or fallback) SHALL contain:
   - `provider` (str) — one of: `gemini`, `anthropic`, `openai`, `hf_endpoint`
   - `model_id` (str) — the model identifier string
   - `hf_endpoint_name` (str, optional) — the HF Inference Endpoint name (required when provider is `hf_endpoint`)
   - `temperature` (float, optional, default 0.3)
   - `max_tokens` (int, optional, default 4096)
4. THE `qdrant` block SHALL contain:
   - `collection` (str) — the Qdrant collection name for this desk's intel
   - `context_top_k` (int, optional, default 5) — number of context results to inject per query
   - `cross_desk_search` (bool, optional, default true) — whether to include cross-desk results in context injection
   - `cross_desk_top_k` (int, optional, default 3)
5. THE Desk_Registry SHALL load the system prompt text from the path specified in `prompt_file` at startup.
6. WHEN a Desk_Config_File is missing a required key, THE Desk_Registry SHALL raise a `ValueError` identifying the file path and the missing key.
7. WHEN a `prompt_file` path specified in a Desk_Config_File does not exist on disk, THE Desk_Registry SHALL raise a `FileNotFoundError` identifying the missing file.
8. THE Desk_Registry SHALL expose a `get(slug: str) -> DeskConfig` method returning the loaded configuration, raising `KeyError` for unknown slugs.
9. THE Desk_Registry SHALL expose a `list_slugs() -> list[str]` method returning all loaded desk slugs.
10. THE Desk_Registry SHALL ship with pre-populated config files for all seven desks: `geopolitical-and-security-desk`, `cultural-and-theological-intelligence-desk`, `science-technology-and-commercial-desk`, `human-intelligence-and-profiling-desk`, `finance-and-economics-directorate`, `cyber-intelligence-and-warfare-desk`, `the-watch-floor` — with model assignments matching the desk-to-model mapping table.
11. WHEN a new YAML file is added to `config/desks/` and the Orchestrator is restarted, THE new desk SHALL be available for routing without any code changes.

---

### Requirement 2: Direct Model Dispatch

**User Story:** As the OSIA Orchestrator, I want to call desk models directly using their native SDKs or OpenAI-compatible HTTP endpoints, so that I eliminate the AnythingLLM HTTP middleware and gain full control over prompts, context windows, and response handling.

#### Acceptance Criteria

1. THE Desk_Registry SHALL expose an async `invoke(slug: str, user_message: str, context_block: str | None) -> str` method that calls the desk's configured model directly and returns the model's text response.
2. WHEN `invoke` is called for a `gemini`-provider desk, THE Desk_Registry SHALL call the model using the `google-genai` SDK with the desk's system prompt and the assembled user message.
3. WHEN `invoke` is called for an `anthropic`-provider desk, THE Desk_Registry SHALL call the model using the `anthropic` Python SDK with the desk's system prompt and the assembled user message.
4. WHEN `invoke` is called for an `openai`-provider desk, THE Desk_Registry SHALL call the model using an OpenAI-compatible `httpx` POST to the OpenAI API with the desk's system prompt and the assembled user message.
5. WHEN `invoke` is called for an `hf_endpoint`-provider desk, THE Desk_Registry SHALL call `HFEndpointManager.ensure_ready(slug)` before dispatching the request, and SHALL call the model using an OpenAI-compatible `httpx` POST to the endpoint's `/v1/chat/completions` path.
6. IF `HFEndpointManager.ensure_ready(slug)` returns `False`, THEN THE Desk_Registry SHALL raise a `RuntimeError` identifying the desk slug and the failed endpoint name.
7. WHEN `context_block` is provided and non-empty, THE Desk_Registry SHALL prepend the context block to the user message under a clearly labelled `## INTELLIGENCE CONTEXT` heading before dispatching to the model.
8. THE Desk_Registry SHALL apply the `max_tokens` and `temperature` values from the desk's primary model config to every primary invocation.
9. THE Desk_Registry SHALL apply a request timeout of 300 seconds to all model invocations.
10. IF a model invocation raises an HTTP error with status 503, THEN THE Desk_Registry SHALL retry the request up to 3 times with a 15-second delay between attempts before attempting the fallback.
11. THE Desk_Registry SHALL only pass tool schemas to the model for tools listed in the desk's `tools` config field — desks cannot call tools not declared in their config file.
12. WHEN a primary model invocation fails after exhausting retries (due to HTTP 5xx, rate limit 429, timeout, or provider SDK error), AND the desk config defines a `fallback` model, THE Desk_Registry SHALL log a warning identifying the desk, the failure reason, and the fallback model being used, then invoke the fallback model using the same assembled message and context block.
13. WHEN invoking a fallback model, THE Desk_Registry SHALL apply the fallback model's own `temperature` and `max_tokens` values from the config.
14. IF the fallback model invocation also fails, THEN THE Desk_Registry SHALL raise the fallback's exception, logging both the primary and fallback failure reasons.
15. WHEN a desk config does not define a `fallback` model and the primary invocation fails after exhausting retries, THE Desk_Registry SHALL raise the exception immediately without attempting any further recovery.
16. THE Desk_Registry SHALL record in the returned response metadata which model (primary or fallback) produced the response, so the Orchestrator can include this in the source audit.

---

### Requirement 3: Qdrant Intelligence Store

**User Story:** As the OSIA system, I want a direct Qdrant client with rich metadata support, so that I can perform filtered semantic search, cross-desk retrieval, and structured upserts without going through AnythingLLM's opaque vector store abstraction.

#### Acceptance Criteria

1. THE Qdrant_Store SHALL expose an async `search(collection: str, query: str, top_k: int, filters: dict | None) -> list[SearchResult]` method that embeds the query using the Embedding_Model and returns the top-K semantically similar points from the specified collection, optionally filtered by payload metadata.
2. THE Qdrant_Store SHALL expose an async `upsert(collection: str, text: str, metadata: dict) -> str` method that embeds the text, generates a deterministic point ID from the text content hash, and upserts the point with the provided metadata payload, returning the point ID.
3. THE Qdrant_Store SHALL expose an async `cross_desk_search(query: str, top_k: int, entity_tags: list[str] | None) -> list[SearchResult]` method that searches across all registered desk collections simultaneously and returns the top-K results ranked by score.
4. THE Qdrant_Store SHALL expose an async `ensure_collection(name: str)` method that creates the collection with 384-dimensional Cosine vectors if it does not already exist, and is a no-op if it does.
5. THE Qdrant_Store SHALL expose an async `collection_stats(name: str) -> dict` method returning point count and vector count for the named collection.
6. WHEN embedding a batch of texts, THE Qdrant_Store SHALL call the HuggingFace Inference API for `sentence-transformers/all-MiniLM-L6-v2` using the `HF_TOKEN` environment variable, matching the embedding model used by the Research_Worker.
7. IF the HuggingFace embedding API call fails, THEN THE Qdrant_Store SHALL log the error and return zero vectors of dimension 384 so that the calling operation can continue without crashing.
8. THE Qdrant_Store SHALL accept the following metadata fields on upsert: `desk` (str), `topic` (str), `source` (str), `reliability_tier` (str, Admiralty A–E), `timestamp` (ISO-8601 str), `entity_tags` (list[str]), and `triggered_by` (str).
9. THE Qdrant_Store SHALL use the `QDRANT_URL` and `QDRANT_API_KEY` environment variables for all Qdrant API calls.
10. THE SearchResult dataclass SHALL contain: `text` (str), `score` (float), `collection` (str), `metadata` (dict).

---

### Requirement 4: Entity Extraction Pipeline

**User Story:** As the OSIA system, I want to extract named entities from every incoming task and RSS article, so that I can automatically enqueue background research jobs that continuously enrich the Qdrant knowledge base.

#### Acceptance Criteria

1. THE Entity_Extractor SHALL expose an async `extract(text: str, source_desk: str) -> list[Entity]` method that calls a Gemini model to identify named entities in the provided text and returns a structured list.
2. THE Entity_Extractor SHALL extract entities of the following types: Person (name, role, affiliation), Organisation (name, type), Location (name, region), Event (name, date if present), Technology (name, category).
3. THE Entity_Extractor SHALL return each entity as a dataclass containing: `name` (str), `entity_type` (str), `context` (str — the sentence or phrase in which the entity appeared), and `source_desk` (str).
4. WHEN `extract` is called and the Gemini model returns malformed or unparseable JSON, THE Entity_Extractor SHALL log a warning and return an empty list rather than raising an exception.
5. THE Entity_Extractor SHALL expose an async `enqueue_research_jobs(entities: list[Entity], triggered_by: str)` method that pushes a research job to `osia:research_queue` for each entity whose normalised name has not already been seen in `osia:research:seen_topics`.
6. WHEN `enqueue_research_jobs` is called, THE Entity_Extractor SHALL construct each research job payload with: `job_id` (UUID4), `topic` (entity name), `desk` (derived from entity type using the entity-to-desk routing table), `priority` ("normal"), `directives_lens` (True), and `triggered_by` (the provided triggered_by string).
7. THE Entity_Extractor SHALL route entity research jobs to desks using the following mapping: Person → `human-intelligence-and-profiling-desk`, Organisation → `geopolitical-and-security-desk`, Location → `geopolitical-and-security-desk`, Event → `geopolitical-and-security-desk`, Technology → `science-technology-and-commercial-desk`.
8. WHEN `enqueue_research_jobs` is called with an empty entity list, THE Entity_Extractor SHALL perform no queue operations and return immediately.
9. THE Entity_Extractor SHALL use the `GEMINI_API_KEY` and `GEMINI_MODEL_ID` environment variables for all model calls.

---

### Requirement 5: Qdrant Context Injection

**User Story:** As the OSIA Orchestrator, I want to fetch relevant Qdrant context before routing a task to a desk, so that desk reports are grounded in both live research and accumulated historical intelligence.

#### Acceptance Criteria

1. WHEN the Orchestrator routes a task to a desk, THE Orchestrator SHALL call `Qdrant_Store.search` on the desk's configured Qdrant collection using the task query as the search term, retrieving the top 5 results.
2. WHEN the Orchestrator routes a task to a desk and entity names have been extracted from the task, THE Orchestrator SHALL additionally call `Qdrant_Store.cross_desk_search` using the entity names as the query, retrieving the top 3 results.
3. THE Orchestrator SHALL format the combined Qdrant results as an `## INTELLIGENCE CONTEXT` block containing each result's text, source desk, reliability tier, and collection timestamp before passing it to `Desk_Registry.invoke`.
4. WHEN the Qdrant search returns no results for a given query, THE Orchestrator SHALL invoke the desk without an Intelligence_Context block rather than injecting an empty block.
5. WHILE the Qdrant_Store is unavailable, THE Orchestrator SHALL log a warning and proceed to invoke the desk without an Intelligence_Context block rather than failing the task.

---

### Requirement 6: Orchestrator Refactor — AnythingLLM Removal

**User Story:** As the OSIA system, I want the Orchestrator to use the new Desk_Registry and Qdrant_Store instead of AnythingLLMDesk, so that all desk routing and vector store operations are direct and under OSIA's control.

#### Acceptance Criteria

1. THE Orchestrator SHALL import and instantiate `Desk_Registry` and `Qdrant_Store` at startup, and SHALL NOT import or instantiate `AnythingLLMDesk`.
2. THE Orchestrator SHALL perform entity extraction on every incoming task payload before routing to a desk, using `Entity_Extractor.extract` on the task's query or article text.
3. THE Orchestrator SHALL call `Entity_Extractor.enqueue_research_jobs` after entity extraction for every task, passing the task's Signal recipient or RSS feed URL as `triggered_by`.
4. THE Orchestrator SHALL fetch Qdrant context (Requirement 5) before every desk invocation and pass the resulting context block to `Desk_Registry.invoke`.
5. THE Orchestrator SHALL call `Desk_Registry.invoke` to produce intelligence reports, replacing all existing calls to `AnythingLLMDesk.send_task`.
6. THE Orchestrator SHALL preserve the existing Signal I/O pipeline (`send_signal_message`, `send_signal_image`) without modification.
7. THE Orchestrator SHALL preserve the existing ADB/social media pipeline (`process_media_link`, `SocialMediaAgent`) without modification.
8. THE Orchestrator SHALL preserve the existing MCP tool dispatch loop (`handle_research`, `_dispatch_tool`) without modification.
9. THE Orchestrator SHALL preserve the existing infographic generation pipeline (`generate_infographic`, `_generate_infographic_via_phone`) without modification.
10. THE Orchestrator SHALL preserve the existing Source_Tracker integration (`build_citation_protocol`, `audit_report`) without modification.
11. THE Orchestrator SHALL preserve the existing RSS ingress routing to the Collection Directorate without modification.
12. WHEN the Orchestrator shuts down, THE Orchestrator SHALL close the `Desk_Registry` HTTP clients in addition to all existing shutdown operations.
13. THE Orchestrator's `VALID_DESKS` set SHALL be populated from `Desk_Registry.list_slugs()` rather than a hardcoded set literal, so that adding a new desk config file automatically makes it a valid routing target after restart.

---

### Requirement 7: HF Endpoint Integration in Desk Layer

**User Story:** As the OSIA system, I want the HFEndpointManager to be integrated into the new Desk_Registry, so that HF-backed desks are automatically woken before dispatch without requiring the Orchestrator to manage endpoint lifecycle directly.

#### Acceptance Criteria

1. THE Desk_Registry SHALL instantiate `HFEndpointManager` at startup and hold a reference to it.
2. WHEN `Desk_Registry.invoke` is called for a desk with `requires_hf_endpoint = True`, THE Desk_Registry SHALL call `HFEndpointManager.ensure_ready(slug)` before constructing the model request.
3. THE Desk_Registry SHALL use the endpoint URL returned by `HFEndpointManager` as the base URL for the OpenAI-compatible POST request to the HF-backed desk.
4. THE Desk_Registry SHALL preserve the existing `HFEndpointManager` wake-up logic, polling intervals, readiness probe, and timeout behaviour without modification.
5. WHEN `HFEndpointManager` is disabled (HF_TOKEN or HF_NAMESPACE not set), THE Desk_Registry SHALL log a warning and attempt to call the HF endpoint using the `HF_ENDPOINT_DOLPHIN_24B` or `HF_ENDPOINT_HERMES_70B` environment variable as a fallback URL.

---

### Requirement 8: Configurable Analytical Mandate

**User Story:** As an OSIA operator, I want the analytical mandate (the lens through which all intelligence is analysed) to be defined in a standalone configuration file, so that the mandate can be changed, replaced, or extended without modifying any code or desk-specific prompt files.

#### Acceptance Criteria

1. THE system SHALL load the active analytical mandate from a file at the path specified by the `OSIA_DIRECTIVES_FILE` environment variable, defaulting to `DIRECTIVES.md` if the variable is not set.
2. THE Desk_Registry SHALL read the mandate file at instantiation time and store its contents as a string.
3. THE Desk_Registry SHALL append the full mandate text to every desk's system prompt before the first model invocation, under a clearly labelled `## ANALYTICAL MANDATE` heading.
4. WHEN a desk's system prompt file already contains the `## ANALYTICAL MANDATE` heading, THE Desk_Registry SHALL NOT append the mandate a second time, allowing desks to embed a custom mandate override directly in their prompt file.
5. WHEN the mandate file specified by `OSIA_DIRECTIVES_FILE` does not exist, THE Desk_Registry SHALL raise a `FileNotFoundError` at startup identifying the missing path.
6. WHEN `OSIA_DIRECTIVES_FILE` is set to an empty string, THE Desk_Registry SHALL operate without appending any mandate, logging a warning that no analytical mandate is configured.
7. THE Desk_Registry SHALL append the citation protocol (from `Source_Tracker.build_citation_protocol()`) to every desk's system prompt, after the mandate block.
8. THE mandate file format SHALL be plain Markdown — no special schema is required, allowing operators to write mandates in natural language.

---

### Requirement 9: Qdrant Collection Bootstrapping

**User Story:** As the OSIA system, I want all required Qdrant collections to be created automatically on first run, so that the system works out of the box without any manual Qdrant provisioning, and existing data is never touched on subsequent starts.

#### Acceptance Criteria

1. WHEN the Orchestrator starts, THE Qdrant_Store SHALL call `ensure_collection` for each of the following collections: `geopolitical_intel`, `cultural_intel`, `science_intel`, `human_intel`, `finance_intel`, `cyber_intel`, `watch_floor`, `collection_raw`, and `osia:research_cache`.
2. WHEN `ensure_collection` is called for a collection that does not yet exist, THE Qdrant_Store SHALL create it with 384-dimensional Cosine vectors and log that the collection was created.
3. WHEN `ensure_collection` is called for a collection that already exists, THE Qdrant_Store SHALL perform no write operations, make no schema changes, and log that the collection is already present and ready.
4. THE bootstrapping process SHALL be fully idempotent — running it on a fresh install, a partially initialised install, or a fully populated production system SHALL always result in the same outcome: all required collections exist and contain whatever data was already there.
5. IF `ensure_collection` fails for any collection due to a Qdrant API error, THEN THE Qdrant_Store SHALL log the error with the collection name and re-raise the exception so the Orchestrator can abort startup cleanly.

---

### Requirement 10: Research Worker Compatibility

**User Story:** As the OSIA system, I want the Research_Worker and HF batch job to continue writing to Qdrant in a format compatible with the new Qdrant_Store, so that existing research output is immediately retrievable by the new context injection pipeline.

#### Acceptance Criteria

1. THE Qdrant_Store.search method SHALL be able to retrieve points written by the existing Research_Worker and HF batch job from the `osia:research_cache` collection without schema migration.
2. THE Qdrant_Store SHALL treat the `text` payload field as the primary content field for all search result formatting, matching the field name used by the Research_Worker.
3. WHEN the Orchestrator performs cross-desk search and the `osia:research_cache` collection is included, THE Qdrant_Store SHALL include `osia:research_cache` results in the ranked output alongside desk-specific collection results.
4. THE Research_Worker SHALL NOT be modified as part of this rebuild; all compatibility SHALL be achieved through the Qdrant_Store's read interface.

---

### Requirement 11: Environment Variable Continuity

**User Story:** As the OSIA operator, I want the new architecture to use the same environment variables as the existing system where possible, so that the `.env` file requires minimal changes during migration.

#### Acceptance Criteria

1. THE Desk_Registry SHALL use `GEMINI_API_KEY` for all Gemini model calls.
2. THE Desk_Registry SHALL use `ANTHROPIC_API_KEY` for all Anthropic model calls.
3. THE Desk_Registry SHALL use `OPENAI_API_KEY` for all OpenAI model calls.
4. THE Desk_Registry SHALL use `HF_TOKEN` and `HF_NAMESPACE` for all HF endpoint operations, matching the variables used by `HFEndpointManager`.
5. THE Qdrant_Store SHALL use `QDRANT_URL` and `QDRANT_API_KEY`, matching the variables used by the Research_Worker.
6. THE Qdrant_Store SHALL use `HF_TOKEN` for embedding API calls, matching the variable used by the Research_Worker.
7. THE new architecture SHALL introduce one new optional environment variable: `OSIA_DIRECTIVES_FILE` (path to the analytical mandate file, defaults to `DIRECTIVES.md`). All other required variables already exist in `.env` and `.env.example`.

---

### Requirement 12: Backward-Compatible Desk Slug Routing

**User Story:** As the OSIA Orchestrator, I want valid desk slugs to be derived entirely from the loaded desk config files, so that adding or removing a desk requires only adding or removing a config file — no code changes needed.

#### Acceptance Criteria

1. THE Orchestrator's set of valid routing targets SHALL be populated exclusively from `Desk_Registry.list_slugs()`, which reflects all YAML files discovered in `config/desks/` at startup.
2. THE seven default desk config files shipped with OSIA SHALL use the same slugs currently used as AnythingLLM workspace slugs (`geopolitical-and-security-desk`, `cultural-and-theological-intelligence-desk`, `science-technology-and-commercial-desk`, `human-intelligence-and-profiling-desk`, `finance-and-economics-directorate`, `cyber-intelligence-and-warfare-desk`, `the-watch-floor`), ensuring backward compatibility with existing Signal ingress, RSS ingress, and task payload formats.
3. WHEN a task payload specifies a desk slug that is not present in the loaded config files, THE Orchestrator SHALL route the task to the desk configured as the default fallback and log a warning identifying the unknown slug.
4. THE default fallback desk SHALL be configurable via a `default_desk` key in a top-level `config/osia.yaml` file, defaulting to `the-watch-floor` if the file or key is absent.
5. WHEN a new YAML file is added to `config/desks/` and the Orchestrator is restarted, the new desk slug SHALL automatically become a valid routing target with no code changes.

---

### Requirement 13: Desk Config File Validation and Schema

**User Story:** As an OSIA operator, I want the system to validate desk config files at startup and provide clear error messages, so that misconfigured desks are caught before they cause runtime failures.

#### Acceptance Criteria

1. THE Desk_Registry SHALL validate every loaded Desk_Config_File against a defined schema at startup, before any model clients are initialised.
2. WHEN a `model.primary.provider` value is not one of `gemini`, `anthropic`, `openai`, `hf_endpoint`, THE Desk_Registry SHALL raise a `ValueError` identifying the file and the invalid provider value.
3. WHEN `model.primary.provider` is `hf_endpoint` and `model.primary.hf_endpoint_name` is absent or empty, THE Desk_Registry SHALL raise a `ValueError` identifying the file and the missing field.
4. WHEN a `model.fallback` block is present and its `provider` value is not one of the valid providers, THE Desk_Registry SHALL raise a `ValueError` identifying the file and the invalid fallback provider value.
5. WHEN a `tools` list entry names a tool not registered in the Orchestrator's MCP dispatcher, THE Desk_Registry SHALL log a warning identifying the desk and the unknown tool name, but SHALL NOT raise an error (the tool will simply never be called).
6. WHEN a `mcp_servers` list entry names a server not present in the MCPDispatcher config, THE Desk_Registry SHALL log a warning identifying the desk and the unknown server name, but SHALL NOT raise an error.
7. THE Desk_Registry SHALL log a startup summary listing each loaded desk, its primary provider/model, fallback provider/model (if configured), Qdrant collection, and tool count.
8. WHEN `config/desks/` contains no YAML files, THE Desk_Registry SHALL raise a `RuntimeError` at startup indicating that no desk configurations were found.

---

### Requirement 14: RSS Ingress — Direct Qdrant Ingest and Entity Extraction

**User Story:** As the OSIA system, I want the RSS ingress to write directly to Qdrant and extract noteworthy entities from each article, so that ingested articles are immediately available for context injection and automatically seed the research queue with background enrichment jobs.

#### Acceptance Criteria

1. THE RSS ingress (`src/gateways/rss_ingress.py`) SHALL replace all calls to `AnythingLLMDesk.ingest_raw_data` with calls to `Qdrant_Store.upsert` targeting the `collection_raw` collection.
2. WHEN upserting an RSS article, THE RSS ingress SHALL include the following metadata: `desk` ("collection-directorate"), `topic` (article title), `source` (feed URL), `reliability_tier` ("B"), `timestamp` (ISO-8601 collection time), `entity_tags` (populated after entity extraction, see criterion 4), `triggered_by` ("rss_ingress").
3. AFTER summarising each new article, THE RSS ingress SHALL call `Entity_Extractor.extract` on the article summary text, passing `"collection-directorate"` as the source desk.
4. THE RSS ingress SHALL call `Entity_Extractor.enqueue_research_jobs` with the extracted entities, passing the feed URL as `triggered_by`, so that each novel entity is queued for background research.
5. THE RSS ingress SHALL populate the `entity_tags` metadata field in the Qdrant upsert with the names of all extracted entities before writing the point.
6. WHEN entity extraction fails for an article (e.g. Gemini API error), THE RSS ingress SHALL log a warning and continue processing — the article SHALL still be upserted to Qdrant and staged in the daily digest without entity tags.
7. THE RSS ingress SHALL continue to stage article summaries in the Redis `osia:rss:daily_digest` key for SITREP consumption, unchanged.
8. THE RSS ingress SHALL continue to mark seen links in `osia:rss:seen_links`, unchanged.
9. THE RSS ingress SHALL NOT import or instantiate `AnythingLLMDesk` after this change.
