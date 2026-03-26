# Implementation Plan: OSIA Core Rebuild

## Overview

Replace AnythingLLM with a direct Python architecture: YAML-driven desk configs, native SDK model dispatch, full Qdrant ownership, and an entity extraction pipeline. Tasks are ordered so each step is immediately integrated — no orphaned code.

## Tasks

- [x] 1. Add new dependencies to pyproject.toml
  - Add `anthropic>=0.40.0`, `qdrant-client>=1.12.0`, `pyyaml>=6.0.2` to `[project.dependencies]`
  - Add `hypothesis`, `pytest`, `pytest-asyncio` to `[project.optional-dependencies]` dev group
  - Run `uv sync` to lock new deps
  - _Requirements: 2.3, 3.9, 11.1–11.6_

- [x] 2. Create global config and desk YAML config files
  - [x] 2.1 Create `config/osia.yaml` with `default_desk: the-watch-floor`
    - _Requirements: 12.4_
  - [x] 2.2 Create `config/desks/` directory and all seven desk YAML files
    - Write one `.yaml` per desk slug using the schema from the design document
    - Include all required top-level keys: `slug`, `name`, `prompt_file`, `model`, `qdrant`, `tools`, `mcp_servers`, `entity_research_target`
    - Use the desk-to-model mapping table from the design for provider/model assignments
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.10, 12.2_

- [x] 3. Implement `src/intelligence/qdrant_store.py`
  - [x] 3.1 Implement `QdrantStore` class with `ensure_collection`, `upsert`, `search`, `cross_desk_search`, `collection_stats`
    - Use `qdrant-client` async client; read `QDRANT_URL` and `QDRANT_API_KEY` from env
    - `ensure_collection`: 384-dim Cosine vectors, idempotent, log created vs already-present
    - `upsert`: embed via HF Inference API (`sentence-transformers/all-MiniLM-L6-v2`, `HF_TOKEN`), deterministic SHA-256 point ID, accept all required metadata fields
    - `search`: embed query, return `list[SearchResult]`, support optional payload filters
    - `cross_desk_search`: fan out across all registered desk collections + `osia:research_cache`, rank by score, deduplicate by point ID
    - `collection_stats`: return point count and vector count
    - On HF embedding failure: log error, return zero vectors (384-dim), continue
    - _Requirements: 3.1–3.10, 9.1–9.5_
  - [ ]* 3.2 Write property test for `ensure_collection` idempotency (Property 7)
    - **Property 7: ensure_collection is idempotent**
    - **Validates: Requirements 3.4, 9.2, 9.3**
  - [ ]* 3.3 Write property test for deterministic upsert point IDs (Property 6)
    - **Property 6: Upsert produces deterministic point IDs**
    - **Validates: Requirements 3.2**
  - [ ]* 3.4 Write property test for SearchResult field completeness (Property 8)
    - **Property 8: SearchResult fields are always complete**
    - **Validates: Requirements 3.10, 10.1, 10.2**

- [x] 4. Implement `src/intelligence/entity_extractor.py`
  - [x] 4.1 Implement `EntityExtractor` class with `extract` and `enqueue_research_jobs`
    - `extract`: call Gemini (`GEMINI_API_KEY`, `GEMINI_MODEL_ID`) to identify entities; parse JSON response into `list[Entity]`; on malformed/unparseable JSON log warning and return `[]`
    - `enqueue_research_jobs`: for each entity, check `osia:research:seen_topics` Redis set; if novel, push job payload to `osia:research_queue` with UUID4 `job_id`, entity-to-desk routing, `priority="normal"`, `directives_lens=True`; skip empty entity lists immediately
    - Use entity-to-desk routing table from design
    - _Requirements: 4.1–4.9_
  - [ ]* 4.2 Write property test for entity pipeline field completeness (Property 9)
    - **Property 9: Entity pipeline fields are complete**
    - **Validates: Requirements 4.3, 4.6**
  - [ ]* 4.3 Write property test for entity-to-desk routing exhaustiveness (Property 10)
    - **Property 10: Entity-to-desk routing is exhaustive**
    - **Validates: Requirements 4.7**

- [x] 5. Implement `src/desks/desk_registry.py`
  - [x] 5.1 Implement `DeskRegistry` class — config loading and validation
    - Discover and load all `config/desks/*.yaml` at instantiation; raise `RuntimeError` if directory is empty
    - Parse each file into `DeskConfig` / `ModelConfig` / `QdrantConfig` dataclasses
    - Validate all required keys; raise `ValueError` (file + key) for missing keys
    - Validate `provider` values; raise `ValueError` for invalid providers
    - Validate `hf_endpoint_name` present when provider is `hf_endpoint`
    - Load `prompt_text` from `prompt_file`; raise `FileNotFoundError` if missing
    - Load analytical mandate from `OSIA_DIRECTIVES_FILE` (default `DIRECTIVES.md`); raise `FileNotFoundError` if set and missing; skip with warning if set to empty string
    - Assemble full system prompt: desk prompt + `## ANALYTICAL MANDATE` (skip if already present) + citation protocol from `SourceTracker.build_citation_protocol()`
    - Log startup summary per desk; expose `get(slug)` and `list_slugs()`
    - _Requirements: 1.1–1.9, 1.11, 8.1–8.8, 13.1–13.8_
  - [ ]* 5.2 Write property test for desk registry round-trip (Property 1)
    - **Property 1: Desk registry round-trip**
    - **Validates: Requirements 1.1, 1.8, 1.9**
  - [ ]* 5.3 Write property test for schema validation rejects missing required keys (Property 2)
    - **Property 2: Schema validation rejects missing required keys**
    - **Validates: Requirements 1.2, 1.6, 13.2, 13.3**
  - [ ]* 5.4 Write property test for prompt file loading (Property 3)
    - **Property 3: Prompt file loading**
    - **Validates: Requirements 1.5**
  - [ ]* 5.5 Write property test for invalid provider rejection (Property 13)
    - **Property 13: Invalid provider values are rejected**
    - **Validates: Requirements 13.2**
  - [x] 5.6 Implement `DeskRegistry.invoke` — model dispatch with retry and fallback
    - Instantiate `HFEndpointManager` at startup; call `ensure_ready(slug)` before HF endpoint dispatch; raise `RuntimeError` if returns `False`
    - Prepend `## INTELLIGENCE CONTEXT` block to user message when `context_block` is provided
    - Dispatch to correct provider: `google-genai` SDK (gemini), `anthropic` SDK (anthropic), `httpx` POST (openai, hf_endpoint)
    - Apply `temperature`, `max_tokens`, 300s timeout per invocation
    - Retry on HTTP 503 up to 3× with 15s delay; on exhaustion attempt fallback if defined
    - On fallback: log warning (desk, failure reason, fallback model); apply fallback's own config
    - On fallback failure: log both failures, raise fallback exception
    - On no fallback: raise primary exception immediately
    - Return response text + metadata `{model_used, model_id}`
    - Expose `close()` to shut down HTTP clients
    - _Requirements: 2.1–2.16, 7.1–7.5_
  - [ ]* 5.7 Write property test for context block prepended under correct heading (Property 4)
    - **Property 4: Context block is prepended under correct heading**
    - **Validates: Requirements 2.7, 5.3**
  - [ ]* 5.8 Write property test for fallback invoked when primary fails (Property 5)
    - **Property 5: Fallback is invoked when primary fails**
    - **Validates: Requirements 2.12, 2.13, 2.16**
  - [ ]* 5.9 Write property test for prompt assembly includes mandate and citation protocol (Property 11)
    - **Property 11: Prompt assembly always includes mandate and citation protocol**
    - **Validates: Requirements 8.2, 8.3, 8.4, 8.7**

- [x] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Update `src/orchestrator.py` — wire in new components
  - [x] 7.1 Replace `AnythingLLMDesk` import/instantiation with `DeskRegistry`, `QdrantStore`, `EntityExtractor`
    - Instantiate all three at startup; call `QdrantStore.ensure_collection` for all nine collections during startup bootstrap
    - Populate `VALID_DESKS` from `DeskRegistry.list_slugs()` instead of a hardcoded set
    - Load `default_desk` from `config/osia.yaml` (fallback to `the-watch-floor`)
    - _Requirements: 6.1, 6.13, 9.1, 12.1, 12.4_
  - [x] 7.2 Add entity extraction and research job enqueuing to the task processing loop
    - After media interception and MCP research loop, call `EntityExtractor.extract` on task query/article text
    - Call `EntityExtractor.enqueue_research_jobs` with Signal recipient or RSS feed URL as `triggered_by`
    - _Requirements: 6.2, 6.3_
  - [x] 7.3 Add Qdrant context injection before desk invocation
    - Call `QdrantStore.search(desk_collection, query, top_k=5)` for the routed desk
    - If entity names extracted, call `QdrantStore.cross_desk_search(entity_names, top_k=3)`
    - Deduplicate combined results by point ID; format `## INTELLIGENCE CONTEXT` block
    - On Qdrant unavailability: log warning, proceed without context block
    - Pass context block (or `None`) to `DeskRegistry.invoke`
    - _Requirements: 5.1–5.5_
  - [x] 7.4 Replace `AnythingLLMDesk.send_task` calls with `DeskRegistry.invoke`
    - Route unknown slugs to `default_desk` with a warning log
    - Include `model_used`/`model_id` from returned metadata in source audit
    - _Requirements: 6.4, 6.5, 12.3_
  - [x] 7.5 Add `DeskRegistry.close()` to orchestrator shutdown sequence
    - _Requirements: 6.12_
  - [ ]* 7.6 Write property test for dynamic routing from registry (Property 12)
    - **Property 12: Dynamic routing from registry**
    - **Validates: Requirements 6.13, 12.1, 12.3**

- [x] 8. Update `src/gateways/rss_ingress.py` — direct Qdrant ingest and entity extraction
  - [x] 8.1 Replace `AnythingLLMDesk.ingest_raw_data` with `QdrantStore.upsert` to `collection_raw`
    - Instantiate `QdrantStore` and `EntityExtractor` in RSS ingress
    - After Gemini summarisation, call `EntityExtractor.extract(summary, "collection-directorate")`
    - Call `EntityExtractor.enqueue_research_jobs(entities, triggered_by=feed_url)`
    - Upsert to `collection_raw` with full metadata: `desk`, `topic`, `source`, `reliability_tier="B"`, `timestamp`, `entity_tags`, `triggered_by="rss_ingress"`
    - On entity extraction failure: log warning, upsert article without entity tags, continue
    - Remove `AnythingLLMDesk` import
    - Preserve Redis `osia:rss:daily_digest` staging and `osia:rss:seen_links` deduplication unchanged
    - _Requirements: 14.1–14.9_
  - [ ]* 8.2 Write property test for RSS upsert metadata completeness (Property 14)
    - **Property 14: RSS upsert metadata is complete**
    - **Validates: Requirements 14.2**

- [x] 9. Update `.env.example` and delete `src/desks/anythingllm_client.py`
  - Add `OSIA_DIRECTIVES_FILE=DIRECTIVES.md` as a documented optional variable to `.env.example`
  - Delete `src/desks/anythingllm_client.py`
  - _Requirements: 11.7, 6.1_

- [x] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Property tests use `hypothesis` with a minimum of 100 iterations per property; tag format: `# Feature: osia-core-rebuild, Property {N}: {property_text}`
- `AnythingLLM` service can be disabled (`systemd/osia-anythingllm.service`) after task 10 is validated — no data migration required as existing Qdrant collections are read-compatible
- The Research_Worker is not modified; all compatibility is achieved through `QdrantStore`'s read interface
