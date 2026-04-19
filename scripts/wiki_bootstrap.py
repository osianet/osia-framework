#!/usr/bin/env python3
"""
OSIA Wiki.js Bootstrap Script
Creates the base page structure for all desks, entity namespaces,
SITREP archive, operations, and knowledge base sections.

AI-updateable sections are fenced with:
  <!-- OSIA:AUTO:section-name -->
  ...content...
  <!-- /OSIA:AUTO:section-name -->

The MCP server (Session 3) uses these markers to patch specific
blocks without rewriting the whole page.

Usage:
  uv run python scripts/wiki_bootstrap.py
  uv run python scripts/wiki_bootstrap.py --dry-run
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from textwrap import dedent

# ── Config ────────────────────────────────────────────────────────────────────

WIKI_URL = "http://localhost:3000/graphql"


def _load_api_key():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    with open(env_path) as f:
        for line in f:
            if line.startswith("WIKIJS_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("WIKIJS_API_KEY not found in .env")


# ── Desk definitions ───────────────────────────────────────────────────────────

DESKS = [
    {
        "slug": "geopolitical-and-security",
        "yaml_slug": "geopolitical-and-security-desk",
        "code": "ATLAS",
        "name": "Geopolitical & Security Desk",
        "director": "Director Marcus Hale",
        "accent": "#00dd55",
        "icon": "🌍",
        "mission": (
            "State conflicts, military operations, occupation, war crimes, "
            "sanctions, diplomacy, imperial power projection, sovereignty, "
            "Palestinian/Israeli conflict, regional alliances."
        ),
        "focus_areas": [
            "State conflicts & military operations",
            "Sanctions regimes & enforcement",
            "Imperial power projection",
            "Regional alliance structures",
            "War crimes & accountability",
            "Occupation & settler-colonialism",
        ],
    },
    {
        "slug": "cyber-intelligence-and-warfare",
        "yaml_slug": "cyber-intelligence-and-warfare-desk",
        "code": "PHANTOM",
        "name": "Cyber Intelligence & Warfare Desk",
        "director": "Commander Kaito Tatewaki",
        "accent": "#20c0e0",
        "icon": "🛡️",
        "mission": (
            "Cyber attacks, malware, nation-state intrusions, CVEs, threat actor TTPs, "
            "digital infrastructure threats, network reconnaissance, surveillance technology."
        ),
        "focus_areas": [
            "Nation-state intrusion campaigns",
            "Malware & ransomware operations",
            "CVE exploitation in the wild",
            "Threat actor TTP mapping",
            "Surveillance technology & spyware",
            "Critical infrastructure threats",
        ],
    },
    {
        "slug": "human-intelligence-and-profiling",
        "yaml_slug": "human-intelligence-and-profiling-desk",
        "code": "SPECTER",
        "name": "Human Intelligence & Profiling Desk",
        "director": "Agent Sarah Chen",
        "accent": "#cc3333",
        "icon": "👁️",
        "mission": (
            "Individual profiling, elite network mapping, trafficking networks, "
            "power structure exposure, activist protection, blackmail/coercion "
            "intelligence, comprador elites."
        ),
        "focus_areas": [
            "Elite network mapping",
            "Trafficking & exploitation networks",
            "Comprador elite profiling",
            "Activist protection intelligence",
            "Coercion & blackmail operations",
            "Financial-political nexus mapping",
        ],
    },
    {
        "slug": "finance-and-economics",
        "yaml_slug": "finance-and-economics-directorate",
        "code": "LEDGER",
        "name": "Finance & Economics Directorate",
        "director": "Director Priya Mehta",
        "accent": "#ffd060",
        "icon": "📊",
        "mission": (
            "Financial markets, economic policy, sanctions enforcement, offshore finance, "
            "corporate extraction, trade, debt as imperial leverage, money flows, OFAC, "
            "corporate accountability."
        ),
        "focus_areas": [
            "Offshore financial structures",
            "Sanctions enforcement & evasion",
            "Debt as imperial leverage",
            "Corporate extraction from the Global South",
            "ICIJ leaks & financial exposure",
            "Central bank & IMF policy analysis",
        ],
    },
    {
        "slug": "information-warfare",
        "yaml_slug": "information-warfare-desk",
        "code": "MIRAGE",
        "name": "Information & Psychological Warfare Desk",
        "director": "Director Alexei Volkov",
        "accent": "#cc3333",
        "icon": "📡",
        "mission": (
            "ADVERSARIAL operations ONLY — state/corporate propaganda, manufactured "
            "disinformation, coordinated inauthentic behaviour, manosphere radicalisation "
            "pipelines. Not aligned accountability content or grassroots social justice material."
        ),
        "focus_areas": [
            "State-sponsored disinformation campaigns",
            "Coordinated inauthentic behaviour (CIB)",
            "Manosphere radicalisation pipelines",
            "Corporate narrative manipulation",
            "Influence operation attribution",
            "Synthetic media & deepfake operations",
        ],
    },
    {
        "slug": "cultural-and-theological-intelligence",
        "yaml_slug": "cultural-and-theological-intelligence-desk",
        "code": "ORACLE",
        "name": "Cultural & Theological Intelligence Desk",
        "director": "Dr. Amara Osei",
        "accent": "#C8860A",
        "icon": "📖",
        "mission": (
            "Cultural movements, religious/ideological formations, feminist analysis, "
            "misogyny, gender dynamics, grassroots organising, survivor testimony, "
            "social justice content, diaspora networks, Indigenous knowledge."
        ),
        "focus_areas": [
            "Religious & ideological formation analysis",
            "Feminist & gender intelligence",
            "Diaspora network mapping",
            "Indigenous knowledge & land sovereignty",
            "Cultural weaponisation by colonial powers",
            "Grassroots organising & social movements",
        ],
    },
    {
        "slug": "science-technology-and-commercial",
        "yaml_slug": "science-technology-and-commercial-desk",
        "code": "FORGE",
        "name": "Science, Technology & Commercial Desk",
        "director": "Dr. Raj Patel",
        "accent": "#40c0ff",
        "icon": "🔬",
        "mission": (
            "AI, emerging technology, scientific research, patents, dual-use technology, "
            "biotech, space, commercial tech sector, academic research, tech policy."
        ),
        "focus_areas": [
            "AI capability & governance",
            "Dual-use technology proliferation",
            "Patent & IP weaponisation",
            "Biotech & biosecurity",
            "Space militarisation",
            "Tech sector corporate accountability",
        ],
    },
    {
        "slug": "environment-and-ecology",
        "yaml_slug": "environment-and-ecology-desk",
        "code": "TERRA",
        "name": "Environment & Ecological Intelligence Desk",
        "director": "Dr. Wren Nakamura",
        "accent": "#00aa44",
        "icon": "🌿",
        "mission": (
            "Climate, ecological disasters, extractivism, environmental racism, "
            "Indigenous land rights, pollution, resource conflicts, food security, "
            "environmental policy."
        ),
        "focus_areas": [
            "Extractivism & land dispossession",
            "Climate displacement & migration",
            "Environmental racism & sacrifice zones",
            "Indigenous land rights & defence",
            "Ecological disaster early warning",
            "Food & water security threats",
        ],
    },
    {
        "slug": "the-watch-floor",
        "yaml_slug": "the-watch-floor",
        "code": "SENTINEL",
        "name": "The Watch Floor",
        "director": "Director James Calloway",
        "accent": "#00dd55",
        "icon": "📡",
        "mission": (
            "Cross-desk synthesis, SITREP generation, situation overviews, "
            "multi-domain queries spanning several desks, and general intelligence "
            "questions that don't fit a single specialist desk."
        ),
        "focus_areas": [
            "Cross-desk intelligence synthesis",
            "Daily SITREP production",
            "Multi-domain threat correlation",
            "Strategic situation assessment",
            "Priority tasking & routing",
            "Watch Floor operational continuity",
        ],
    },
]

# ── Declassified KB collections ────────────────────────────────────────────────

DECLASSIFIED_COLLECTIONS = [
    {
        "slug": "frus-state-dept",
        "name": "FRUS — Foreign Relations of the United States",
        "description": (
            "Declassified US State Department diplomatic cables and foreign policy records "
            "from the *Foreign Relations of the United States* series. Covers Cold War-era "
            "diplomacy, covert operations, and US foreign policy decision-making. "
            "Source: `HistoryAtState/frus` GitHub repository (TEI XML)."
        ),
        "qdrant_collection": "frus-state-dept",
        "primary_desks": ["Geopolitical & Security", "Information Warfare", "Watch Floor"],
    },
    {
        "slug": "church-committee",
        "name": "Church Committee Reports",
        "description": (
            "US Senate Select Committee to Study Governmental Operations with Respect to "
            "Intelligence Activities (1975–76). Exposed CIA assassination plots, COINTELPRO, "
            "NSA mass surveillance, and illegal domestic operations. "
            "Source: govinfo.gov."
        ),
        "qdrant_collection": "church-committee",
        "primary_desks": ["Geopolitical & Security", "HUMINT", "Information Warfare", "Watch Floor"],
    },
    {
        "slug": "pentagon-papers",
        "name": "The Pentagon Papers",
        "description": (
            "US Department of Defense study of political and military involvement in Vietnam, "
            "leaked by Daniel Ellsberg in 1971. Five volumes (Gravel Edition). Reveals systematic "
            "government deception over the Vietnam War. "
            "Source: archive.org."
        ),
        "qdrant_collection": "pentagon-papers",
        "primary_desks": ["Geopolitical & Security", "Information Warfare"],
    },
    {
        "slug": "cia-crest",
        "name": "CIA CREST Archive",
        "description": (
            "CIA Records Search Tool — declassified CIA documents released under FOIA. "
            "Covers covert operations, intelligence assessments, and internal agency records. "
            "Source: archive.org `ciaindexed` collection."
        ),
        "qdrant_collection": "cia-crest",
        "primary_desks": ["Geopolitical & Security", "HUMINT", "Watch Floor"],
    },
    {
        "slug": "fbi-vault",
        "name": "FBI Vault — FOIA Releases",
        "description": (
            "FBI Freedom of Information Act releases covering domestic surveillance, "
            "COINTELPRO operations, organised crime, and political investigations. "
            "Source: archive.org FBI FOIA collections."
        ),
        "qdrant_collection": "fbi-vault",
        "primary_desks": ["HUMINT", "Information Warfare"],
    },
    {
        "slug": "wikileaks-cables",
        "name": "WikiLeaks — US Diplomatic Cables",
        "description": (
            "124,000+ US diplomatic cables released by WikiLeaks. Covers candid assessments "
            "of world leaders, covert operations, corporate lobbying, and diplomatic back-channels. "
            "Source: fn5/wikileaks-cables dataset."
        ),
        "qdrant_collection": "wikileaks-cables",
        "primary_desks": ["Geopolitical & Security", "HUMINT", "Finance", "Information Warfare", "Watch Floor"],
    },
    {
        "slug": "epstein-files",
        "name": "Epstein — Declassified Files",
        "description": (
            "Declassified court documents, FOIA releases, and investigative records "
            "relating to Jeffrey Epstein's trafficking network, associated elites, "
            "and institutional failures to prosecute. "
            "Primary use: elite network mapping and HUMINT."
        ),
        "qdrant_collection": "epstein-files",
        "primary_desks": ["HUMINT", "Finance", "Geopolitical & Security"],
    },
]

# ── GraphQL helpers ────────────────────────────────────────────────────────────


def gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    req = urllib.request.Request(
        WIKI_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body[:300]}") from e


def create_page(
    api_key: str,
    path: str,
    title: str,
    content: str,
    description: str = "",
    tags: list[str] | None = None,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        print(f"  [DRY-RUN] Would create: /{path}")
        return True

    query = """
    mutation CreatePage($path: String!, $title: String!, $content: String!,
                        $description: String!, $tags: [String]!) {
      pages {
        create(
          path: $path
          title: $title
          content: $content
          description: $description
          editor: "markdown"
          isPublished: true
          isPrivate: false
          locale: "en"
          tags: $tags
        ) {
          responseResult { succeeded errorCode message }
          page { id path }
        }
      }
    }
    """
    resp = gql(
        api_key,
        query,
        {
            "path": path,
            "title": title,
            "content": content,
            "description": description,
            "tags": tags or [],
        },
    )

    result = resp["data"]["pages"]["create"]["responseResult"]
    if result["succeeded"]:
        page_id = resp["data"]["pages"]["create"]["page"]["id"]
        print(f"  ✓ /{path} (id={page_id})")
        return True
    elif "already" in result.get("message", "").lower() or result.get("errorCode") == "PageDuplicateCreate":
        print(f"  ~ /{path} (already exists, skipping)")
        return True
    else:
        print(f"  ✗ /{path} — {result['message']} (code: {result['errorCode']})")
        return False


# ── Content templates ──────────────────────────────────────────────────────────


def home_page() -> str:
    desk_links = "\n".join(
        f"| {d['icon']} [{d['name']}](/desks/{d['slug']}) | `{d['code']}` | {d['director']} |" for d in DESKS
    )
    return dedent(f"""\
    > **OSIA INTERNAL — RESTRICTED ACCESS**
    > *Unauthorised access is a serious security violation. All activity is logged.*

    # OSIA Intelligence Wiki

    The **Open Source Intelligence Agency** intelligence wiki is the living knowledge layer
    for all OSIA intelligence products. It complements the Qdrant vector database (semantic
    retrieval) with human-browsable dossiers, INTSUM archives, and cross-linked entity records.

    This wiki is maintained jointly by human analysts and by Hermes, the OSIA AI orchestrator.
    AI-maintained sections are marked with `<!-- OSIA:AUTO -->` markers.

    ---

    ## Intelligence Desks

    | Desk | Code | Director |
    |------|------|----------|
    {desk_links}

    ---

    ## Key Sections

    | Section | Purpose |
    |---------|---------|
    | [Intelligence Desks](/desks) | Desk landing pages, watchlists, INTSUM archives |
    | [Entities](/entities) | Persons, organisations, locations, networks |
    | [SITREP Archive](/sitrep) | Daily situation reports |
    | [Cross-Desk INTSUMs](/intsums) | Watch Floor synthesis products |
    | [Operations](/operations) | Named multi-desk investigations |
    | [Knowledge Base](/kb) | Declassified documents, sources, thematic analysis |

    ---

    ## How This Wiki Is Maintained

    - **Desk landing pages** — updated by Hermes after each INTSUM cycle
    - **Entity dossiers** — created and expanded automatically as entities appear in intelligence
    - **SITREP pages** — auto-generated daily at 07:00 UTC
    - **Standing assessments** — human-curated analytical positions, AI-assisted updates
    - **Operations** — opened by analysts, maintained jointly

    *Wiki infrastructure: Wiki.js + PostgreSQL, served at `https://wiki.osia.dev`*
    *Vector retrieval: Qdrant at `https://qdrant.osia.dev`*
    """)


def desk_page(desk: dict) -> str:
    focus_list = "\n".join(f"- {f}" for f in desk["focus_areas"])
    return dedent(f"""\
    > **CLASSIFICATION: OSIA INTERNAL — {desk["code"]} DESK**

    # {desk["icon"]} {desk["name"]} | `{desk["code"]}`

    **Director:** {desk["director"]}
    **Mission:** {desk["mission"]}

    ---

    ## Focus Areas

    {focus_list}

    ---

    ## Active Watchlist

    <!-- OSIA:AUTO:watchlist -->
    *No active watchlist entries. Hermes will populate this section automatically.*
    <!-- /OSIA:AUTO:watchlist -->

    ---

    ## Current Analytical Focus

    <!-- OSIA:AUTO:focus -->
    *No current focus items.*
    <!-- /OSIA:AUTO:focus -->

    ---

    ## Recent Intelligence Products

    <!-- OSIA:AUTO:recent-intsums -->
    *No recent intelligence products. INTSUMs will appear here automatically after each cycle.*
    <!-- /OSIA:AUTO:recent-intsums -->

    ---

    ## Key Entities (Tracked by This Desk)

    <!-- OSIA:AUTO:key-entities -->
    *No entities currently flagged. Entity links will appear here as dossiers are created.*
    <!-- /OSIA:AUTO:key-entities -->

    ---

    ## Navigation

    - 📁 [INTSUM Archive](/desks/{desk["slug"]}/intsums) — All intelligence summaries from this desk
    - 📋 [Standing Assessments](/desks/{desk["slug"]}/standing-assessments) — Persistent analytical positions
    - 🎯 [Watchlist](/desks/{desk["slug"]}/watchlist) — Entities and topics under active monitoring
    """)


def desk_intsums_index(desk: dict) -> str:
    return dedent(f"""\
    > **CLASSIFICATION: OSIA INTERNAL — {desk["code"]} DESK**

    # {desk["icon"]} {desk["name"]} — INTSUM Archive

    Intelligence summaries produced by the **{desk["name"]}**. Each entry represents a
    completed analytical cycle. Pages are created automatically by Hermes at the path
    `/desks/{desk["slug"]}/intsums/YYYY-MM-DD-topic-slug`.

    ---

    ## Archive

    <!-- OSIA:AUTO:intsum-index -->
    *No intelligence products archived yet. Entries will appear here automatically.*
    <!-- /OSIA:AUTO:intsum-index -->

    ---

    *← [Back to {desk["name"]}](/desks/{desk["slug"]})*
    """)


def desk_standing_assessments(desk: dict) -> str:
    return dedent(f"""\
    > **CLASSIFICATION: OSIA INTERNAL — {desk["code"]} DESK**

    # {desk["icon"]} {desk["name"]} — Standing Assessments

    Standing assessments are persistent analytical positions maintained by the
    **{desk["name"]}**. Unlike INTSUMs (which are point-in-time products), standing
    assessments are living documents updated as the situation evolves.

    Each assessment carries a **reliability tier** and **last-reviewed date**.
    Hermes flags assessments for human review when new intelligence contradicts them.

    ---

    ## Active Assessments

    <!-- OSIA:AUTO:standing-assessments -->
    *No standing assessments yet. Create one below to begin.*
    <!-- /OSIA:AUTO:standing-assessments -->

    ---

    ## How to Create a Standing Assessment

    Create a sub-page at `/desks/{desk["slug"]}/standing-assessments/your-topic`.
    Use the following structure at minimum:

    ```
    **Assessment date:** YYYY-MM-DD
    **Last reviewed:** YYYY-MM-DD
    **Reliability tier:** A / B / C
    **Status:** Active / Superseded / Under Review

    ## Position
    [Your analytical position]

    ## Evidentiary Basis
    [Sources and intelligence products supporting this position]

    ## Dissenting Views
    [Alternative interpretations, if any]
    ```

    *← [Back to {desk["name"]}](/desks/{desk["slug"]})*
    """)


def desk_watchlist(desk: dict) -> str:
    return dedent(f"""\
    > **CLASSIFICATION: OSIA INTERNAL — {desk["code"]} DESK**

    # {desk["icon"]} {desk["name"]} — Active Watchlist

    Entities and topics under active monitoring by the **{desk["name"]}**.
    This page is maintained by Hermes and updated after each research and INTSUM cycle.

    **Priority levels:** 🔴 Critical · 🟡 Elevated · 🟢 Routine

    ---

    ## Persons Under Watch

    <!-- OSIA:AUTO:watchlist-persons -->
    *No persons currently on watchlist.*
    <!-- /OSIA:AUTO:watchlist-persons -->

    ## Organisations Under Watch

    <!-- OSIA:AUTO:watchlist-organisations -->
    *No organisations currently on watchlist.*
    <!-- /OSIA:AUTO:watchlist-organisations -->

    ## Topics & Events Under Watch

    <!-- OSIA:AUTO:watchlist-topics -->
    *No topics currently on watchlist.*
    <!-- /OSIA:AUTO:watchlist-topics -->

    ---

    *← [Back to {desk["name"]}](/desks/{desk["slug"]})*
    *↗ [INTSUM Archive](/desks/{desk["slug"]}/intsums)*
    """)


def entities_index() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # Entity Registry

    The OSIA entity registry contains dossiers on persons, organisations, locations,
    and networks of intelligence interest. Dossiers are created automatically by Hermes
    when entities appear in intelligence products, and expanded as new information arrives.

    **Entity pages are living documents** — they accumulate intelligence over time rather
    than representing a single point-in-time assessment.

    ---

    ## Categories

    | Category | Description |
    |----------|-------------|
    | 👤 [Persons](/entities/persons) | Individual dossiers — politicians, executives, operatives, persons of interest |
    | 🏛️ [Organisations](/entities/organisations) | Corporate, state, NGO, and criminal organisation profiles |
    | 📍 [Locations](/entities/locations) | Geographic locations, facilities, regions of operational significance |
    | 🕸️ [Networks](/entities/networks) | Mapped relationship networks (Epstein network, Five Eyes, etc.) |

    ---

    ## Recently Updated

    <!-- OSIA:AUTO:recently-updated-entities -->
    *No entities yet. Dossiers will appear here as they are created.*
    <!-- /OSIA:AUTO:recently-updated-entities -->
    """)


def entity_category_index(kind: str, icon: str, description: str, ai_section: str) -> str:
    return dedent(f"""\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # {icon} {kind}

    {description}

    Dossiers are created at `/{kind.lower()}/slug-name` when an entity is first
    identified in OSIA intelligence products. Pages accumulate intelligence across
    multiple desk cycles.

    ---

    ## Index

    <!-- OSIA:AUTO:{ai_section}-index -->
    *No {kind.lower()} dossiers yet. Entries will appear here automatically.*
    <!-- /OSIA:AUTO:{ai_section}-index -->
    """)


def entity_dossier_template() -> str:
    """Reference template — not a live page, stored in /kb/methodology."""
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — [DESK CODE]**

    # 👤 [Entity Name]

    | Field | Value |
    |-------|-------|
    | **Type** | Person / Organisation / Location / Network |
    | **Status** | Active / Inactive / Deceased / Unknown |
    | **Threat Level** | 🔴 Critical / 🟡 Elevated / 🟢 Routine / ⚪ Monitoring |
    | **Primary Desk** | [Desk name] |
    | **First Seen** | YYYY-MM-DD |
    | **Last Updated** | YYYY-MM-DD |
    | **Tags** | `tag1` `tag2` |

    ---

    ## Summary

    <!-- OSIA:AUTO:summary -->
    *Summary pending first intelligence cycle.*
    <!-- /OSIA:AUTO:summary -->

    ---

    ## Profile

    <!-- OSIA:AUTO:profile -->
    *Profile data pending.*
    <!-- /OSIA:AUTO:profile -->

    ---

    ## Known Associations

    <!-- OSIA:AUTO:associations -->
    *No associations mapped yet.*
    <!-- /OSIA:AUTO:associations -->

    ---

    ## Timeline

    <!-- OSIA:AUTO:timeline -->
    *No timeline entries yet.*
    <!-- /OSIA:AUTO:timeline -->

    ---

    ## Intelligence Products

    Intelligence summaries referencing this entity:

    <!-- OSIA:AUTO:intel-products -->
    *No intelligence products reference this entity yet.*
    <!-- /OSIA:AUTO:intel-products -->

    ---

    ## Source References

    <!-- OSIA:AUTO:sources -->
    *No sources cited yet.*
    <!-- /OSIA:AUTO:sources -->

    ---

    *← [Back to Entity Registry](/entities)*
    """)


def intsums_index() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — WATCH FLOOR PRODUCT**

    # Cross-Desk Intelligence Summaries

    This section contains intelligence summaries produced by the **Watch Floor**
    that synthesise intelligence from multiple desks. Desk-specific INTSUMs are
    archived under each desk's own section.

    | Path | Contents |
    |------|----------|
    | `/intsums/` | Watch Floor cross-desk synthesis products |
    | `/desks/<slug>/intsums/` | Desk-specific intelligence summaries |
    | `/sitrep/<year>/` | Daily situation reports |

    ---

    ## Archive

    <!-- OSIA:AUTO:intsums-index -->
    *No cross-desk intelligence products yet.*
    <!-- /OSIA:AUTO:intsums-index -->
    """)


def sitrep_index() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — WATCH FLOOR PRODUCT**

    # Daily SITREP Archive

    Situation reports are generated daily at **07:00 UTC** by the Watch Floor,
    synthesising the RSS digest, overnight research queue results, and standing
    Qdrant context across all desks.

    SITREPs are stored at `/sitrep/<year>/<YYYY-MM-DD>`.

    ---

    ## Archive

    <!-- OSIA:AUTO:sitrep-index -->
    *No SITREPs archived yet. The first will appear after the next 07:00 UTC cycle.*
    <!-- /OSIA:AUTO:sitrep-index -->
    """)


def operations_index() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # Operations

    Named OSIA investigations — multi-desk, long-running intelligence efforts
    focused on a specific target, network, or event. Operations are opened by
    analysts and maintained jointly with Hermes.

    Each operation lives at `/operations/<operation-slug>/` and contains:

    | Sub-page | Contents |
    |----------|----------|
    | `/` | Operation overview, status, tasking |
    | `/timeline` | Chronological event log |
    | `/entities` | Entities of interest to this operation |
    | `/intsums` | Intelligence products produced for this operation |

    ---

    ## Active Operations

    <!-- OSIA:AUTO:active-operations -->
    *No active operations. Open one by creating a sub-page under `/operations/`.*
    <!-- /OSIA:AUTO:active-operations -->

    ---

    ## Closed Operations

    <!-- OSIA:AUTO:closed-operations -->
    *No closed operations.*
    <!-- /OSIA:AUTO:closed-operations -->

    ---

    ## Opening a New Operation

    1. Create `/operations/operation-name` with an overview page
    2. Add sub-pages: `timeline`, `entities`, `intsums`
    3. Tag all related entity dossiers and INTSUMs with the operation name
    4. Notify Hermes via the research queue to begin tasking

    *Naming convention: `operation-<codename-lowercase>` — e.g. `operation-ironledger`*
    """)


def kb_index() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # Knowledge Base

    The OSIA knowledge base contains reference material, declassified historical
    documents, source registry, thematic analysis, and methodological frameworks.

    | Section | Contents |
    |---------|----------|
    | 🗄️ [Declassified](/kb/declassified) | Historical document collections (FRUS, Church Committee, WikiLeaks, etc.) |
    | 📚 [Source Registry](/kb/sources) | Intelligence source catalogue and reliability assessments |
    | 🧠 [Thematic Analysis](/kb/thematic) | Long-form analytical pieces on recurring themes |
    | 🔧 [Methodology](/kb/methodology) | OSIA analytical frameworks, desk templates, wiki conventions |
    """)


def kb_declassified_index() -> str:
    collection_rows = "\n".join(
        f"| [{c['name']}](/kb/declassified/{c['slug']}) | {', '.join(c['primary_desks'])} |"
        for c in DECLASSIFIED_COLLECTIONS
    )
    return dedent(f"""\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # Declassified Document Collections

    Historical document collections ingested into Qdrant and indexed here for
    analyst reference. Each collection page describes the source, coverage, and
    which desks draw on it as a boost collection.

    | Collection | Primary Desks |
    |------------|---------------|
    {collection_rows}

    ---

    > **Note:** These documents are indexed in Qdrant with temporal decay disabled
    > for historical records (`ingested_at_unix` set to document date, not ingest date).
    > Use the Qdrant dashboard at `https://qdrant.osia.dev` for direct collection queries.
    """)


def kb_collection_page(collection: dict) -> str:
    desk_list = "\n".join(f"- {d}" for d in collection["primary_desks"])
    return dedent(f"""\
    > **CLASSIFICATION: OSIA INTERNAL — KNOWLEDGE BASE**

    # {collection["name"]}

    {collection["description"]}

    ---

    ## Coverage & Usage

    **Qdrant collection:** `{collection["qdrant_collection"]}`
    **Used as boost collection by:**
    {desk_list}

    ---

    ## Notable Documents

    <!-- OSIA:AUTO:notable-docs -->
    *No notable documents flagged yet. Analysts can add key documents here for quick reference.*
    <!-- /OSIA:AUTO:notable-docs -->

    ---

    ## Related Entities

    <!-- OSIA:AUTO:related-entities -->
    *Entity links will appear here as dossiers reference documents from this collection.*
    <!-- /OSIA:AUTO:related-entities -->

    ---

    *← [Back to Declassified Collections](/kb/declassified)*
    """)


def kb_sources_index() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # Source Registry

    Registry of intelligence sources used by OSIA — RSS feeds, APIs, human sources,
    and document collections. Each source entry carries a reliability tier assessment.

    **Reliability tiers (Admiralty Scale):**

    | Tier | Source Reliability | Information Reliability |
    |------|--------------------|------------------------|
    | A1 | Completely reliable | Confirmed by other sources |
    | B2 | Usually reliable | Probably true |
    | C3 | Fairly reliable | Possibly true |
    | D4 | Not usually reliable | Doubtful |
    | E5 | Unreliable | Improbable |
    | F6 | Cannot be judged | Truth cannot be judged |

    ---

    ## Registered Sources

    <!-- OSIA:AUTO:source-registry -->
    *Source registry is populated manually by analysts. Add entries below.*
    <!-- /OSIA:AUTO:source-registry -->
    """)


def kb_thematic_index() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # Thematic Analysis

    Long-form analytical pieces on recurring intelligence themes. These are
    human-curated with AI assistance — they represent sustained analytical
    positions rather than point-in-time INTSUM products.

    Topics are cross-desk by nature: a thematic piece on *debt as imperial leverage*
    draws from Finance, Geopolitical, and Information Warfare simultaneously.

    ---

    ## Published Analyses

    <!-- OSIA:AUTO:thematic-index -->
    *No thematic analyses published yet.*
    <!-- /OSIA:AUTO:thematic-index -->
    """)


def kb_methodology_index() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # Methodology & Conventions

    OSIA analytical frameworks, wiki conventions, desk templates, and operational
    standards. This section is the reference guide for both human analysts and
    the Hermes AI system.

    | Document | Purpose |
    |----------|---------|
    | [Entity Dossier Template](/kb/methodology/entity-dossier-template) | Standard structure for all entity pages |
    | [INTSUM Format Guide](/kb/methodology/intsum-format) | How intelligence summaries are structured |
    | [Reliability Assessment Framework](/kb/sources) | Admiralty scale source & info grading |
    | [Wiki Conventions](/kb/methodology/wiki-conventions) | Path naming, tagging, AUTO markers |

    ---

    ## AI Maintenance Conventions

    ### AUTO Markers

    Sections maintained by Hermes are fenced with:

    ````
    <!-- OSIA:AUTO:section-name -->
    ...content managed by Hermes...
    <!-- /OSIA:AUTO:section-name -->
    ````

    The MCP server's `wiki_update_section` tool patches only the content between
    these markers, leaving surrounding human-curated content untouched.

    ### Path Conventions

    | Content type | Path pattern |
    |-------------|--------------|
    | Desk INTSUM | `/desks/<desk-slug>/intsums/YYYY-MM-DD-topic-slug` |
    | Person dossier | `/entities/persons/firstname-lastname` |
    | Organisation | `/entities/organisations/org-slug` |
    | Location | `/entities/locations/location-slug` |
    | Network | `/entities/networks/network-slug` |
    | Daily SITREP | `/sitrep/YYYY/YYYY-MM-DD` |
    | Cross-desk INTSUM | `/intsums/YYYY-MM-DD-topic-slug` |
    | Operation | `/operations/operation-slug` |

    ### Tagging Conventions

    All pages should carry relevant tags for cross-referencing:
    - Desk tag: `desk-atlas`, `desk-phantom`, etc.
    - Type tag: `intsum`, `dossier`, `sitrep`, `standing-assessment`, `operation`
    - Entity tags: entity slugs referenced in the page
    """)


def wiki_conventions() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — METHODOLOGY**

    # Wiki Conventions

    Reference guide for maintaining consistency across the OSIA wiki —
    for both human analysts and the Hermes AI system.

    ## Naming Conventions

    - All paths: lowercase, hyphen-separated, no special characters
    - Dates in paths: `YYYY-MM-DD` format always
    - Entity slugs: `firstname-lastname` for persons, `org-full-name` for organisations
    - Operation codenames: `operation-codename` (e.g. `operation-ironledger`)

    ## Page Titles

    - Desk pages: `[Icon] Desk Name | CODE`
    - Entity pages: `👤 Full Name` / `🏛️ Organisation Name`
    - INTSUMs: `INTSUM — Topic Description — YYYY-MM-DD`
    - SITREPs: `SITREP — YYYY-MM-DD`
    - Operations: `Operation CODENAME — Brief Description`

    ## Tags

    Every page must carry at minimum:
    1. A **type tag**: `intsum`, `dossier`, `sitrep`, `standing-assessment`, `operation`, `kb`
    2. A **desk tag**: `desk-atlas`, `desk-phantom`, `desk-specter`, etc.
    3. **Entity tags** for every named entity referenced

    ## Classification Headers

    Every page opens with a blockquote classification marker:

    ```
    > **CLASSIFICATION: OSIA INTERNAL — [DESK CODE]**
    ```

    Watch Floor products use:
    ```
    > **CLASSIFICATION: OSIA INTERNAL — WATCH FLOOR PRODUCT**
    ```

    *← [Back to Methodology](/kb/methodology)*
    """)


def intsum_format_guide() -> str:
    return dedent("""\
    > **CLASSIFICATION: OSIA INTERNAL — METHODOLOGY**

    # INTSUM Format Guide

    Standard structure for all OSIA intelligence summaries. Hermes produces INTSUMs
    in this format. Human analysts should follow the same structure for consistency.

    ---

    ## Standard INTSUM Structure

    ```markdown
    > **CLASSIFICATION: OSIA INTERNAL — [DESK CODE]**
    > **INTSUM-[DESK_CODE]-YYYY-MM-DD-NNN**

    # INTSUM — [Topic] — YYYY-MM-DD

    **Desk:** [Desk name]
    **Analyst:** Hermes / [Human analyst name]
    **Classification:** OSIA Internal
    **Reliability:** A1 / B2 / C3 (source / information)
    **Related entities:** [[Entity 1]], [[Entity 2]]
    **Tags:** `tag1` `tag2`

    ---

    ## Executive Summary

    [2-4 sentence summary of the key intelligence finding]

    ---

    ## Assessment

    [Detailed analytical assessment — what happened, what it means,
    what the implications are for OSIA's mandate]

    ---

    ## Key Entities

    [Named entities involved, with links to dossiers where they exist]

    ---

    ## Source Material

    [Sources consulted — with reliability tier where known]

    ---

    ## Implications & Tasking

    [What this intelligence implies for other desks, what follow-up
    research or monitoring is recommended]
    ```

    *← [Back to Methodology](/kb/methodology)*
    """)


# ── Bootstrap runner ───────────────────────────────────────────────────────────


def run(dry_run: bool = False):
    api_key = _load_api_key()
    ok = 0
    fail = 0

    def c(path, title, content, description="", tags=None):
        nonlocal ok, fail
        result = create_page(api_key, path, title, content, description, tags, dry_run)
        if result:
            ok += 1
        else:
            fail += 1
        if not dry_run:
            time.sleep(0.3)  # be gentle with the DB

    print("\n── Home ─────────────────────────────────────────────────────")
    c(
        "home",
        "OSIA Intelligence Wiki",
        home_page(),
        "OSIA wiki home — intelligence products, entity dossiers, SITREP archive",
        ["osia", "home"],
    )

    print("\n── Desks index ──────────────────────────────────────────────")
    desks_index_content = dedent(
        """\
    > **CLASSIFICATION: OSIA INTERNAL — RESTRICTED**

    # Intelligence Desks

    """
        + "\n".join(f"- {d['icon']} [{d['name']}](/desks/{d['slug']}) — `{d['code']}` — {d['director']}" for d in DESKS)
    )
    c("desks", "Intelligence Desks", desks_index_content, tags=["index", "desks"])

    print("\n── Desk pages ───────────────────────────────────────────────")
    for desk in DESKS:
        slug = desk["slug"]
        code = desk["code"].lower()
        print(f"\n  [{desk['code']}] {desk['name']}")
        c(
            f"desks/{slug}",
            f"{desk['icon']} {desk['name']} | {desk['code']}",
            desk_page(desk),
            desk["mission"],
            ["desk", f"desk-{code}"],
        )
        c(
            f"desks/{slug}/intsums",
            f"{desk['name']} — INTSUM Archive",
            desk_intsums_index(desk),
            tags=["intsum", "index", f"desk-{code}"],
        )
        c(
            f"desks/{slug}/standing-assessments",
            f"{desk['name']} — Standing Assessments",
            desk_standing_assessments(desk),
            tags=["standing-assessment", "index", f"desk-{code}"],
        )
        c(
            f"desks/{slug}/watchlist",
            f"{desk['name']} — Active Watchlist",
            desk_watchlist(desk),
            tags=["watchlist", f"desk-{code}"],
        )

    print("\n── Entities ─────────────────────────────────────────────────")
    c(
        "entities",
        "Entity Registry",
        entities_index(),
        "OSIA entity registry — persons, organisations, locations, networks",
        ["entity", "index"],
    )
    c(
        "entities/persons",
        "Persons",
        entity_category_index(
            "Persons",
            "👤",
            "Individual dossiers on politicians, executives, operatives, and persons of intelligence interest.",
            "persons",
        ),
        tags=["entity", "person", "index"],
    )
    c(
        "entities/organisations",
        "Organisations",
        entity_category_index(
            "Organisations",
            "🏛️",
            "Profiles on corporate, state, NGO, and criminal organisations of intelligence interest.",
            "organisations",
        ),
        tags=["entity", "organisation", "index"],
    )
    c(
        "entities/locations",
        "Locations",
        entity_category_index(
            "Locations",
            "📍",
            "Geographic locations, facilities, and regions of operational or intelligence significance.",
            "locations",
        ),
        tags=["entity", "location", "index"],
    )
    c(
        "entities/networks",
        "Networks",
        entity_category_index(
            "Networks",
            "🕸️",
            "Mapped relationship networks — elite networks, intelligence alliances, criminal organisations, and power structures.",
            "networks",
        ),
        tags=["entity", "network", "index"],
    )

    print("\n── INTSUMs & SITREPs ────────────────────────────────────────")
    c("intsums", "Cross-Desk Intelligence Summaries", intsums_index(), tags=["intsum", "index", "watch-floor"])
    c("sitrep", "Daily SITREP Archive", sitrep_index(), tags=["sitrep", "index", "watch-floor"])

    print("\n── Operations ───────────────────────────────────────────────")
    c("operations", "Operations", operations_index(), "Named OSIA multi-desk investigations", ["operation", "index"])

    print("\n── Knowledge Base ───────────────────────────────────────────")
    c("kb", "Knowledge Base", kb_index(), tags=["kb", "index"])
    c(
        "kb/declassified",
        "Declassified Document Collections",
        kb_declassified_index(),
        tags=["kb", "declassified", "index"],
    )

    print("\n  Declassified collections:")
    for col in DECLASSIFIED_COLLECTIONS:
        c(
            f"kb/declassified/{col['slug']}",
            col["name"],
            kb_collection_page(col),
            tags=["kb", "declassified", col["slug"]],
        )

    c("kb/sources", "Source Registry", kb_sources_index(), tags=["kb", "sources"])
    c("kb/thematic", "Thematic Analysis", kb_thematic_index(), tags=["kb", "thematic", "index"])
    c("kb/methodology", "Methodology & Conventions", kb_methodology_index(), tags=["kb", "methodology"])
    c(
        "kb/methodology/entity-dossier-template",
        "Entity Dossier Template",
        entity_dossier_template(),
        tags=["kb", "methodology", "template"],
    )
    c(
        "kb/methodology/intsum-format",
        "INTSUM Format Guide",
        intsum_format_guide(),
        tags=["kb", "methodology", "intsum"],
    )
    c("kb/methodology/wiki-conventions", "Wiki Conventions", wiki_conventions(), tags=["kb", "methodology"])

    print(f"\n── Done — {ok} created/skipped · {fail} failed ─────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap OSIA Wiki.js page structure")
    parser.add_argument("--dry-run", action="store_true", help="Print pages without creating")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
