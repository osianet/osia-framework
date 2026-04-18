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
