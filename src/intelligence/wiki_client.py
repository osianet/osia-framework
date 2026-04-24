"""Wiki.js client for OSIA intelligence products.

Async Python client wrapping the Wiki.js GraphQL API. Used by the orchestrator,
research worker, and Hermes worker to write intelligence products to the wiki.
All operations are non-fatal — callers should log and continue on failure.

Environment variables:
  WIKIJS_URL      — GraphQL endpoint (default: http://localhost:3000/graphql)
  WIKIJS_API_KEY  — API key for write access
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger("osia.wiki_client")

WIKIJS_URL: str = os.getenv("WIKIJS_URL", "http://localhost:3000/graphql")
WIKIJS_API_KEY: str = os.getenv("WIKIJS_API_KEY", "")

_DESKS_DIR = Path("config/desks")
_desk_wiki_cache: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _load_desk_wiki_paths() -> dict[str, str]:
    global _desk_wiki_cache
    if _desk_wiki_cache:
        return _desk_wiki_cache
    mapping: dict[str, str] = {}
    for yaml_path in _DESKS_DIR.glob("*.yaml"):
        try:
            with open(yaml_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            slug = cfg.get("slug", yaml_path.stem)
            section = cfg.get("aesthetic", {}).get("wiki_section", "")
            if slug and section:
                mapping[slug] = section.strip("/")
        except Exception as exc:
            logger.debug("Skipping %s: %s", yaml_path.name, exc)
    _desk_wiki_cache = mapping
    return mapping


def desk_wiki_section(desk_slug: str) -> str:
    """Return the wiki section path for a desk slug (e.g. 'desks/geopolitical-and-security')."""
    paths = _load_desk_wiki_paths()
    if desk_slug in paths:
        return paths[desk_slug]
    base = desk_slug.removesuffix("-desk").removesuffix("-directorate")
    return f"desks/{base}"


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_len].rstrip("-")


def intsum_wiki_path(desk_slug: str, dt: datetime, topic: str) -> str:
    """Wiki path for an INTSUM: desks/<desk>/intsums/<date>-<topic-slug>"""
    section = desk_wiki_section(desk_slug)
    return f"{section}/intsums/{dt.strftime('%Y-%m-%d')}-{_slugify(topic or desk_slug)}"


def sitrep_wiki_path(dt: datetime) -> str:
    """Wiki path for a daily SITREP: sitrep/<date>"""
    return f"sitrep/{dt.strftime('%Y-%m-%d')}"


_ENTITY_FOLDER: dict[str, str] = {
    "Person": "persons",
    "Organisation": "organisations",
    "Location": "locations",
    "Network": "networks",
    "Event": "organisations",
    "Technology": "organisations",
}


def entity_wiki_path(entity_type: str, name: str) -> str:
    """Wiki path for an entity: entities/<type-folder>/<name-slug>"""
    folder = _ENTITY_FOLDER.get(entity_type, "organisations")
    return f"entities/{folder}/{_slugify(name)}"


def social_account_wiki_path(platform: str, handle: str) -> str:
    """Wiki path for a social media account dossier: entities/social-accounts/<platform>/<handle-slug>"""
    return f"entities/social-accounts/{platform.lower()}/{_slugify(handle)}"


# ---------------------------------------------------------------------------
# Page content builders
# ---------------------------------------------------------------------------


def build_intsum_page(
    analysis: str,
    desk_name: str,
    desk_section: str,
    ref_number: str,
    timestamp: str,
    source: str,
    entity_links: list[tuple[str, str]],
    source_account: tuple[str, str] | None = None,
) -> str:
    """Build INTSUM wiki page markdown with AUTO-fenced sections.

    source_account: optional (handle, wiki_path) for the content creator dossier link.
    """
    entity_str = ", ".join(f"[{name}](/{path})" for name, path in entity_links) or "*None identified*"
    creator_row = f"| **Content creator** | [@{source_account[0]}](/{source_account[1]}) |\n" if source_account else ""
    return (
        f"<!-- OSIA:AUTO:metadata -->\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| **Ref** | {ref_number} |\n"
        f"| **Date** | {timestamp} |\n"
        f"| **Desk** | [{desk_name}](/{desk_section}) |\n"
        f"| **Source** | {source} |\n"
        f"{creator_row}"
        f"| **Reliability** | A |\n"
        f"| **Entities** | {entity_str} |\n"
        f"<!-- /OSIA:AUTO:metadata -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:content -->\n"
        f"{analysis.strip()}\n"
        f"<!-- /OSIA:AUTO:content -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:corroboration -->\n"
        f"*Pending Hermes corroboration review.*\n"
        f"<!-- /OSIA:AUTO:corroboration -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:related-intel -->\n"
        f"*No related intelligence cross-referenced.*\n"
        f"<!-- /OSIA:AUTO:related-intel -->\n"
    )


def build_social_account_page(
    handle: str,
    platform: str,
    display_name: str,
    channel_url: str,
    first_seen: str,
    intsum_path: str,
    intsum_title: str,
) -> str:
    """Build a new social media account dossier wiki page."""
    profile_url = channel_url or f"https://www.{platform.lower()}.com/{handle}"
    dn = display_name or handle
    return (
        f"<!-- OSIA:AUTO:profile -->\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| **Handle** | @{handle} |\n"
        f"| **Platform** | {platform.title()} |\n"
        f"| **Display name** | {dn} |\n"
        f"| **Channel** | [{profile_url}]({profile_url}) |\n"
        f"| **First intel** | {first_seen} |\n"
        f"| **Reels processed** | 1 |\n"
        f"<!-- /OSIA:AUTO:profile -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:summary -->\n"
        f"*No summary compiled. Pending research worker analysis.*\n"
        f"<!-- /OSIA:AUTO:summary -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:intel-log -->\n"
        f"## Intelligence Log\n\n"
        f"- {first_seen} — [{intsum_title}](/{intsum_path})\n"
        f"<!-- /OSIA:AUTO:intel-log -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:research-notes -->\n"
        f"## Research Notes\n\n"
        f"*No research notes on file.*\n"
        f"<!-- /OSIA:AUTO:research-notes -->\n"
    )


def build_entity_page(
    entity_type: str,
    desk_name: str,
    desk_section: str,
    first_seen: str,
    intsum_path: str = "",
    intsum_title: str = "",
    summary: str = "",
) -> str:
    """Build a new entity wiki page with AUTO-fenced sections."""
    summary_body = summary.strip() if summary.strip() else "*No summary compiled. Pending research worker analysis.*"
    intel_log_entries = (
        f"- {first_seen} — [{intsum_title}](/{intsum_path})\n"
        if intsum_path and intsum_title
        else "*No intelligence products linked yet.*\n"
    )
    return (
        f"<!-- OSIA:AUTO:profile -->\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| **Type** | {entity_type} |\n"
        f"| **First seen** | {first_seen} |\n"
        f"| **Active desks** | [{desk_name}](/{desk_section}) |\n"
        f"<!-- /OSIA:AUTO:profile -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:summary -->\n"
        f"{summary_body}\n"
        f"<!-- /OSIA:AUTO:summary -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:intel-log -->\n"
        f"## Intelligence Log\n\n"
        f"{intel_log_entries}"
        f"<!-- /OSIA:AUTO:intel-log -->\n\n"
        f"---\n\n"
        f"<!-- OSIA:AUTO:research-notes -->\n"
        f"## Research Notes\n\n"
        f"*No research notes on file.*\n"
        f"<!-- /OSIA:AUTO:research-notes -->\n"
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class WikiClient:
    """Async client for Wiki.js GraphQL API.

    Can be used as an async context manager (creates its own httpx.AsyncClient)
    or with a shared client passed to the constructor.
    """

    def __init__(self, http: httpx.AsyncClient | None = None) -> None:
        self._external_http = http
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "WikiClient":
        self._client = self._external_http or httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *_) -> None:
        if self._external_http is None and self._client:
            await self._client.aclose()
        self._client = None

    async def _gql(self, query: str, variables: dict | None = None) -> dict:
        assert self._client is not None, "WikiClient used outside context manager"
        resp = await self._client.post(
            WIKIJS_URL,
            json={"query": query, "variables": variables or {}},
            headers={
                "Authorization": f"Bearer {WIKIJS_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def get_page(self, path: str) -> dict | None:
        """Return page dict with keys: id, path, title, description, tags (list[str]), content.
        Returns None if the page does not exist."""
        q = """
        query GetPage($path: String!) {
          pages {
            singleByPath(path: $path, locale: "en") {
              id path title description
              tags { tag }
              content
            }
          }
        }"""
        data = await self._gql(q, {"path": path})
        page = (data.get("data") or {}).get("pages", {}).get("singleByPath")
        if not page:
            return None
        page["tags"] = [t["tag"] for t in (page.get("tags") or [])]
        return page

    async def create_page(
        self,
        path: str,
        title: str,
        content: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> int | None:
        """Create a page. Returns the new page ID, or None on failure."""
        q = """
        mutation CreatePage(
          $path: String!, $title: String!, $content: String!,
          $description: String!, $tags: [String]!
        ) {
          pages {
            create(
              path: $path title: $title content: $content
              description: $description editor: "markdown"
              isPublished: true isPrivate: false locale: "en" tags: $tags
            ) {
              responseResult { succeeded errorCode message }
              page { id }
            }
          }
        }"""
        data = await self._gql(
            q,
            {
                "path": path,
                "title": title,
                "content": content,
                "description": description,
                "tags": tags or [],
            },
        )
        result = data["data"]["pages"]["create"]
        if result["responseResult"]["succeeded"]:
            return result["page"]["id"]
        logger.warning("Wiki create failed for '%s': %s", path, result["responseResult"]["message"])
        return None

    async def update_page(
        self,
        page_id: int,
        content: str,
        title: str,
        description: str,
        tags: list[str],
    ) -> bool:
        """Full page update. Returns True on success."""
        q = """
        mutation UpdatePage(
          $id: Int!, $content: String!, $title: String!,
          $description: String!, $tags: [String]!
        ) {
          pages {
            update(
              id: $id content: $content title: $title
              description: $description editor: "markdown"
              isPublished: true isPrivate: false locale: "en" tags: $tags
            ) {
              responseResult { succeeded errorCode message }
            }
          }
        }"""
        data = await self._gql(
            q,
            {
                "id": page_id,
                "content": content,
                "title": title,
                "description": description,
                "tags": tags,
            },
        )
        result = data["data"]["pages"]["update"]["responseResult"]
        if not result["succeeded"]:
            logger.warning("Wiki update failed for page %d: %s", page_id, result["message"])
        return result["succeeded"]

    async def upsert_page(
        self,
        path: str,
        title: str,
        content: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> bool:
        """Create or overwrite a page. Returns True on success."""
        existing = await self.get_page(path)
        if existing:
            return await self.update_page(
                existing["id"],
                content,
                title,
                description,
                tags if tags is not None else existing.get("tags", []),
            )
        return (await self.create_page(path, title, content, description, tags)) is not None

    async def patch_section(self, path: str, section: str, new_content: str) -> bool:
        """Replace the content of an OSIA:AUTO-fenced section."""
        existing = await self.get_page(path)
        if not existing:
            return False
        body = existing["content"]
        open_m = f"<!-- OSIA:AUTO:{section} -->"
        close_m = f"<!-- /OSIA:AUTO:{section} -->"
        if open_m not in body:
            logger.warning("Section %r not found in wiki page '%s'", section, path)
            return False
        body = re.sub(
            rf"{re.escape(open_m)}.*?{re.escape(close_m)}",
            f"{open_m}\n{new_content.strip()}\n{close_m}",
            body,
            flags=re.DOTALL,
        )
        return await self.update_page(
            existing["id"],
            body,
            existing["title"],
            existing.get("description", ""),
            existing.get("tags", []),
        )

    async def append_to_section(self, path: str, section: str, new_entry: str) -> bool:
        """Append a line/block to an OSIA:AUTO-fenced section."""
        existing = await self.get_page(path)
        if not existing:
            return False
        body = existing["content"]
        open_m = f"<!-- OSIA:AUTO:{section} -->"
        close_m = f"<!-- /OSIA:AUTO:{section} -->"
        if open_m not in body:
            logger.warning("Section %r not found in wiki page '%s'", section, path)
            return False

        def _append(m: re.Match) -> str:
            inner = m.group(1).rstrip()
            return f"{open_m}\n{inner}\n{new_entry.strip()}\n{close_m}"

        body = re.sub(
            rf"{re.escape(open_m)}(.*?){re.escape(close_m)}",
            _append,
            body,
            flags=re.DOTALL,
        )
        return await self.update_page(
            existing["id"],
            body,
            existing["title"],
            existing.get("description", ""),
            existing.get("tags", []),
        )

    async def search_pages(self, query_str: str) -> list[dict]:
        """Full-text search. Returns list of {id, title, path, description}."""
        q = """
        query SearchPages($query: String!) {
          pages {
            search(query: $query, locale: "en") {
              results { id title path description }
            }
          }
        }"""
        data = await self._gql(q, {"query": query_str})
        return (data.get("data") or {}).get("pages", {}).get("search", {}).get("results", [])
