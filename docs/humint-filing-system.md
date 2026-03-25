# HUMINT Filing System — Design Document

## Current State

The `human-intelligence-and-profiling-desk` Qdrant collection has 1 point — a
failed task artifact. No structured intelligence on people, organisations, or
places is being filed. All incoming intel flows into `collection-directorate`
and desk-specific collections as unstructured research chunks, with no
deliberate subject-centric filing.

## Problem

A real intelligence agency maintains **subject files** — dossiers that
accumulate over time as new intelligence arrives. OSIA currently has no
equivalent. When a reel names three politicians, that context is ingested once
as a research blob and never cross-referenced against those individuals again.

The knowledge base grows in volume but not in depth on any specific subject.

---

## Design Goals

1. **Passive accumulation** — subject files grow automatically as intel flows
   through the system, without requiring explicit "file this" commands.
2. **Queryable** — "what do we know about X?" should return a coherent,
   accumulated picture, not a pile of unrelated chunks.
3. **Simple** — no new infrastructure. Uses Qdrant and AnythingLLM as-is.
4. **Non-blocking** — filing happens as a background side-effect of normal
   task processing, never delaying report delivery.

---

## Proposed Approach: Entity Extraction + Background Filing

### How it works

Every time a piece of intelligence is processed (media intercept, research
summary, RSS article), run a lightweight extraction pass to identify named
entities. For each entity, ingest a structured summary into the HUMINT desk
tagged with that entity's name.

Over time, the HUMINT desk accumulates a growing body of knowledge per subject.
When queried ("what do we know about Scott Morrison?"), AnythingLLM's RAG
retrieval pulls all relevant chunks together.

### Entity types to file

- **People** — politicians, business leaders, military figures, activists,
  journalists, influencers named in intel
- **Organisations** — companies, NGOs, government bodies, military units,
  media outlets
- **Places** — cities, regions, facilities, borders relevant to ongoing
  situations (lower priority, geopolitical desk already covers this)

### Filing format

Each filed chunk follows a consistent structure so RAG retrieval is clean:

```
SUBJECT: [Full Name]
TYPE: person | organisation | place
ROLE: [Their role/position at time of filing]
SOURCE: [URL or description of source]
DATE: [ISO date]
CONTEXT: [What was said or reported about them — 2-5 sentences]
TAGS: [comma-separated themes: politics, military, finance, etc.]
```

### Where it fits in the pipeline

```
Incoming intel (media intercept / research / RSS)
    │
    ▼
process_task() — existing flow unchanged
    │
    ├──► Desk analysis → Signal report (unchanged)
    │
    └──► _file_entities() [NEW, runs concurrently, non-blocking]
              │
              ▼
         Extract named entities via Gemini (cheap, fast prompt)
              │
              ▼
         For each entity → ingest structured chunk into
         human-intelligence-and-profiling-desk
```

The filing step runs with `asyncio.create_task()` so it never delays the
report. If it fails, it logs a warning and moves on.

---

## Entity Extraction Prompt Design

The extraction prompt needs to be cheap and structured. It should return JSON:

```json
{
  "entities": [
    {
      "name": "Scott Morrison",
      "type": "person",
      "role": "Former Australian Prime Minister",
      "context": "Appeared in reel defending fossil fuel subsidies, claimed transition costs would hurt workers.",
      "tags": ["politics", "australia", "energy", "labour"]
    }
  ]
}
```

Rules for the prompt:
- Only extract entities that are **meaningfully discussed** — not just mentioned in passing
- Cap at 5 entities per source to prevent runaway ingestion
- Skip generic entities (countries, common nouns)
- Include enough context to be useful standalone (2-5 sentences)

---

## Query Pattern

When the HUMINT desk is asked about a subject, AnythingLLM's RAG will retrieve
all chunks tagged with that name. The desk's Dolphin 3.0 model (uncensored,
running on Ollama) then synthesises a profile from the accumulated chunks.

Example query: `@agent Build a profile on Scott Morrison based on all available intelligence`

The desk will pull all filed chunks mentioning Morrison and produce a coherent
behavioural/political profile.

---

## Limitations of this approach

- **No deduplication** — the same fact may be filed multiple times from
  different sources. This is acceptable (more context = better RAG) but means
  point counts will grow fast.
- **No structured lookup** — you can't enumerate all subjects or list all
  people we have files on. It's search-first, not browse-first.
- **Embedding quality** — retrieval quality depends on the embedding model
  (`Xenova/all-MiniLM-L6-v2`, 384 dims). Unusual names may not retrieve well
  without exact text match.
- **No confidence scoring** — all filed intel is treated equally regardless
  of source quality.

These are acceptable tradeoffs for the current stage. A proper Qdrant-native
filing system with structured payloads and metadata filtering would address
them, but requires more infrastructure work.

---

## Future: Structured Subject Records (Phase 2)

When the passive accumulation approach has been running for a while and the
limitations become painful, the next step is to maintain a **canonical subject
record** in Qdrant with rich metadata:

```json
{
  "id": "uuid",
  "payload": {
    "subject_type": "person",
    "canonical_name": "Scott Morrison",
    "aliases": ["ScoMo"],
    "roles": ["Former PM of Australia", "Liberal Party leader"],
    "affiliations": ["Liberal Party of Australia", "Horizon Church"],
    "first_seen": "2024-01-15",
    "last_updated": "2026-03-25",
    "intel_count": 47,
    "tags": ["politics", "australia", "energy", "religion"]
  }
}
```

This enables:
- Listing all subjects we have files on
- Filtering by type, tag, affiliation
- Tracking when we first encountered a subject
- Network mapping (who appears together)

This is the right long-term architecture but requires a dedicated
`src/intelligence/subject_registry.py` module and careful schema design.

---

## Implementation Plan (Phase 1)

1. Add `_extract_entities(text: str) -> list[dict]` to orchestrator — calls
   Gemini with the extraction prompt, returns structured entity list
2. Add `_file_entities(entities: list[dict], source: str)` — formats each
   entity as a structured chunk and ingests into HUMINT desk via
   `AnythingLLMDesk.ingest_raw_data()`
3. Call `asyncio.create_task(_file_entities(...))` at the end of
   `process_task()` after media analysis and research are complete
4. Apply to: media intercepts, research summaries, RSS article summaries

Estimated scope: ~60 lines of new code in `orchestrator.py`.
