# OSIA Wiki.js Implementation Plan

## Overview
Wiki.js serves as OSIA's **living intelligence product layer** — human-browsable entity dossiers,
INTSUM archives, and cross-linked reports. It complements Qdrant (semantic retrieval) rather than
replacing it. An MCP server provides the AI interface so Hermes and the orchestrator can maintain
it autonomously.

---

## Session 1 — Docker Setup & Infrastructure ✅
**Goal:** `https://wiki.osia.dev` serving Wiki.js, fully TLS-terminated.

- [x] Add `wikijs` + `wikijs-db` (Postgres 15) services to `docker-compose.yml`
- [x] Add `WIKIJS_DB_PASSWORD` to `.env`
- [x] Create data directory `/home/ubuntu/osia-wikijs/db`
- [x] Create nginx config `wiki.osia.dev.conf`
- [x] Add `wiki.osia.dev` A record to Cloudflare (ZeroTier IP, DNS-only — private until locked down)
- [x] Expand Let's Encrypt cert (`osia.dev-0001`) to include `wiki.osia.dev` via DNS challenge
- [x] Run initial setup wizard (admin account, site title)
- [x] Verify GraphQL API endpoint at `https://wiki.osia.dev/graphql`
- [x] Add `WIKIJS_API_KEY` to `.env`

---

## Session 2 — OSIA Aesthetic Customisation & Base Structure ✅
**Goal:** Wiki matches OSIA visual identity and has complete page scaffolding.

- [x] Inject custom CSS (`assets/wiki/osia_wiki.css`) — dark terminal aesthetic, Ubuntu fonts, green primary
- [x] Upload `osia_logo_sm.png` as site logo
- [x] Set site title, description, robots noindex/nofollow (private)
- [x] Configure navigation sidebar (OSIA, Intelligence Desks, Entities, Knowledge Base)
- [x] Add `aesthetic:` block to all 9 desk YAMLs (logo variant, accent colour, bg style, wiki icon, css class, path)
- [x] Bootstrap 61 base pages via `scripts/wiki_bootstrap.py` (idempotent, re-runnable):
  ```
  /home                          Wiki home
  /desks/<slug>/                 9 desk landing pages
  /desks/<slug>/intsums          INTSUM archive per desk
  /desks/<slug>/standing-assessments
  /desks/<slug>/watchlist
  /entities/{persons,organisations,locations,networks}
  /intsums                       Cross-desk Watch Floor products
  /sitrep                        Daily SITREP archive
  /operations                    Named investigations scaffold
  /kb/declassified/<collection>  7 declassified KB collections
  /kb/{sources,thematic,methodology}
  /kb/methodology/{entity-dossier-template,intsum-format,wiki-conventions}
  ```
- [x] AI-maintainable sections fenced with `<!-- OSIA:AUTO:section-name -->` markers

---

## Session 3 — MCP Server ✅
**Goal:** Clean tool surface for AI to read/write wiki pages.

Repo: `git@github.com:osianet/wiki-js-mcp.git` (`/home/ubuntu/wiki-js-mcp`)

Tools:
| Tool | Description |
|------|-------------|
| `wiki_create_page` | Create new page at path with markdown content |
| `wiki_update_page` | Overwrite or patch AUTO-fenced section or update metadata only |
| `wiki_get_page` | Read page content by path |
| `wiki_search_pages` | Full-text search, returns title/path/description/tags |
| `wiki_list_pages` | List pages, optional path prefix filter |
| `wiki_move_page` | Rename/restructure a page path |

Error contract: all tools return `{"success": bool, "error": str | null, ...}` — never raise,
always return structured result so the calling AI can decide how to handle failures.

GraphQL client: `httpx` (async). API key stored as `WIKIJS_API_KEY` in `.env`.

---

## Session 4 — Orchestrator Wiring ✅
**Goal:** Automatic wiki maintenance as intelligence is produced.

- [x] **`src/intelligence/wiki_client.py`** — async Python GraphQL client (mirrors MCP server logic):
  - `WikiClient` async context manager with `get_page`, `create_page`, `update_page`, `upsert_page`, `patch_section`, `append_to_section`, `search_pages`
  - Path helpers: `intsum_wiki_path()`, `sitrep_wiki_path()`, `entity_wiki_path()`, `desk_wiki_section()` (reads `aesthetic.wiki_section` from desk YAMLs)
  - Page builders: `build_intsum_page()`, `build_entity_page()` — AUTO-fenced sections for targeted updates

- [x] **Orchestrator** (`src/orchestrator.py`):
  - Removed `generate_intsum_pdf` — replaced with `_wiki_publish_coro()` in the post-analysis gather
  - INTSUM published to `desks/<desk>/intsums/<date>-<topic-slug>` (or `sitrep/<date>` when `original_query` starts with "Generate a Daily SITREP")
  - `wiki_path` stored in Qdrant payload for Hermes back-reference
  - Entity pages created/updated: new entities get full dossier scaffold; existing pages get intel-log entry appended
  - SITREP auto-detected (query prefix) → routed to `/sitrep/<date>` instead of desk intsums
  - `WIKIJS_API_KEY` guard: wiki publish is a no-op when key is absent

- [x] **Research worker** (`src/workers/research_worker.py`):
  - After `store_research()`: searches wiki for entity page matching `job.topic`, appends research summary to `<!-- OSIA:AUTO:research-notes -->` section
  - Falls back to deterministic org-type path if search returns no entity pages

- [x] **Hermes worker** (`src/workers/hermes_worker.py`):
  - After verdict: if Qdrant payload carries `wiki_path`, patches `<!-- OSIA:AUTO:corroboration -->` section with verdict, confidence, reasoning, and sources

- [x] **`scripts/wiki_import_pdfs.py`** — PDF archive importer:
  - Parses `reports/*.pdf` filenames (`{date}_{time}_{desk}.pdf`) to derive wiki paths
  - Extracts text via PyMuPDF (falls back to pypdf)
  - Creates `desks/<desk>/intsums/<date>-<time>-imported` pages, idempotent
  - `--dry-run`, `--limit N`, `--desk SLUG` flags

---

## Cross-Linking Strategy

Pages link to each other through four mechanisms:

| Mechanism | How |
|-----------|-----|
| INTSUM → entities | Metadata table lists entity wiki links (`[Name](/entities/...)`) |
| Entity → INTSUMs | `intel-log` section gets a new entry per INTSUM that mentions the entity |
| Entity → research | `research-notes` section accumulates research worker findings |
| INTSUM → corroboration | `corroboration` section patched by Hermes after verification |

SITREP pages (`/sitrep/<date>`) are naturally linked to the desk INTSUMs of that day via shared entity pages — any entity mentioned in both a desk INTSUM and the SITREP will have both listed in its intel-log, creating a web of cross-references without explicit SITREP→INTSUM pointers.

**To kick off the archive import:**
```bash
# Preview first
uv run python scripts/wiki_import_pdfs.py --dry-run --limit 10

# Import all (recommended: run in screen/tmux — ~700 PDFs)
uv run python scripts/wiki_import_pdfs.py
```

---

## Key Config
| Item | Value |
|------|-------|
| Internal port | 3000 |
| Public URL | `https://wiki.osia.dev` |
| DB | Postgres 15, internal Docker network |
| Data path | `/home/ubuntu/osia-wikijs/` |
| Env vars | `WIKIJS_DB_PASSWORD`, `WIKIJS_API_KEY` |
