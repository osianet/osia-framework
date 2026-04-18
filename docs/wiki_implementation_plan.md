# OSIA Wiki.js Implementation Plan

## Overview
Wiki.js serves as OSIA's **living intelligence product layer** — human-browsable entity dossiers,
INTSUM archives, and cross-linked reports. It complements Qdrant (semantic retrieval) rather than
replacing it. An MCP server provides the AI interface so Hermes and the orchestrator can maintain
it autonomously.

---

## Session 1 — Docker Setup & Infrastructure
**Goal:** `https://wiki.osia.dev` serving Wiki.js, fully TLS-terminated.

- [ ] Add `wikijs` + `wikijs-db` (Postgres 15) services to `docker-compose.yml`
- [ ] Add `WIKIJS_DB_PASSWORD` to `.env`
- [ ] Create data directory `/home/ubuntu/osia-wikijs/db`
- [ ] Create nginx config `wiki.osia.dev.conf`
- [ ] Issue Let's Encrypt cert for `wiki.osia.dev`
- [ ] Run initial setup wizard (admin account, site title, Git storage backend optional)
- [ ] Verify GraphQL API endpoint at `https://wiki.osia.dev/graphql`

---

## Session 2 — OSIA Aesthetic Customisation
**Goal:** Wiki matches OSIA visual identity (dark intel-agency aesthetic).

- [ ] Inject custom CSS via Wiki.js Admin → Theme
- [ ] Upload OSIA logo as site logo
- [ ] Configure navigation sidebar (Entities, INTSUMs, Desks, Declassified KBs)
- [ ] Set up initial folder/namespace structure:
  ```
  /entities/persons/
  /entities/organisations/
  /entities/locations/
  /intsums/<desk-slug>/
  /desks/<desk-slug>/
  /kb/declassified/
  ```
- [ ] Create a template page for entity dossiers

---

## Session 3 — MCP Server
**Goal:** Clean tool surface for AI to read/write wiki pages.

File: `src/mcp/wiki_mcp.py`

Tools:
| Tool | Description |
|------|-------------|
| `wiki_create_page` | Create new page at path with markdown content |
| `wiki_update_page` | Overwrite or append to existing page |
| `wiki_get_page` | Read page content by path or ID |
| `wiki_search_pages` | Full-text search, returns title/path/snippet/tags |
| `wiki_list_pages` | List pages under a folder path |
| `wiki_move_page` | Rename/restructure a page path |

Error contract: all tools return `{"success": bool, "error": str | null, ...}` — never raise,
always return structured result so the calling AI can decide how to handle failures.

GraphQL client: `gql` + `aiohttp` transport. API key stored as `WIKIJS_API_KEY` in `.env`.

---

## Session 4 — Orchestrator Wiring
**Goal:** Automatic wiki maintenance as intelligence is produced.

- [ ] INTSUM write-back: after Watch Floor synthesises, write to `/intsums/<desk>/<date>-<slug>`
- [ ] Entity upsert: after entity extraction, create/update `/entities/persons/<name>` etc.
- [ ] Research worker: append research summaries to relevant entity pages
- [ ] Daily SITREP: write to `/intsums/sitrep/<date>`

---

## Key Config
| Item | Value |
|------|-------|
| Internal port | 3000 |
| Public URL | `https://wiki.osia.dev` |
| DB | Postgres 15, internal Docker network |
| Data path | `/home/ubuntu/osia-wikijs/` |
| Env vars | `WIKIJS_DB_PASSWORD`, `WIKIJS_API_KEY` |
