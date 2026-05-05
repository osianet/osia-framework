"""
OSIA Profile Investigation Worker — actively investigate Instagram profiles in the wiki.

For each uninvestigated Instagram social-account page (up to MAX_PER_RUN per run):
  1. Fetches the actual Instagram profile via yt-dlp + account pool (bio, display
     name, follower count, recent post captions) — the starting point for everything
     that follows.
  2. Cross-desk Qdrant search across HUMINT-relevant collections — retrieves
     existing OSIA intelligence on this person from ingested reels and INTSUMs
     (with 70-day temporal decay).
  3. Targeted web searches — informed by real bio text, display name, and recent
     post themes rather than just a bare username.
  4. Calls the HUMINT desk (venice-uncensored) directly with all three context
     layers for synthesis.
  5. Writes the completed profile directly to the wiki summary section.

Processes MAX_PER_RUN profiles per invocation to respect Venice rate limits.

Environment variables:
  WIKIJS_URL                  — Wiki.js GraphQL endpoint
  WIKIJS_API_KEY              — Wiki.js API key
  REDIS_URL                   — Redis connection URL
  QDRANT_URL                  — Qdrant HTTP endpoint
  QDRANT_API_KEY              — Qdrant API key
  VENICE_API_KEY              — Venice API key (HUMINT desk primary)
  OPENROUTER_API_KEY          — Fallback if Venice is unavailable
  PROFILE_INV_MAX_PER_RUN     — Profiles investigated per run (default: 10)
  PROFILE_INV_COOLDOWN_DAYS   — Days before re-investigating a handle (default: 7)

Run:
  uv run python -m src.workers.profile_investigation_worker
"""

import asyncio
import hashlib
import logging
import os
import re
from datetime import UTC, datetime

import redis.asyncio as aioredis

from src.agents.instagram_account_manager import InstagramAccountManager
from src.desks.desk_registry import DeskRegistry
from src.intelligence.qdrant_store import QdrantStore
from src.intelligence.wiki_client import WikiClient

logger = logging.getLogger("osia.profile_investigation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEN_KEY_PREFIX = "osia:profile_inv:seen:"
MAX_PER_RUN = int(os.getenv("PROFILE_INV_MAX_PER_RUN", "10"))
COOLDOWN_DAYS = int(os.getenv("PROFILE_INV_COOLDOWN_DAYS", "7"))
SEEN_TTL = COOLDOWN_DAYS * 24 * 3600

HUMINT_DESK = "human-intelligence-and-profiling-desk"
QDRANT_DECAY_DAYS = 70.0
QDRANT_TOP_K = 12

# Only search collections relevant to HUMINT profiling, not cyber/CTI KBs.
# This avoids fanning out to 25+ collections and hitting Qdrant connection limits.
_HUMINT_COLLECTIONS = [
    "human-intelligence-and-profiling-desk",
    "the-watch-floor",
    "collection-directorate",
    "information-warfare-desk",
    "geopolitical-and-security-desk",
    "wikileaks-cables",
    "epstein-files",
]

_STUB_MARKER = "Pending research worker analysis"

_HANDLE_RE = re.compile(r"\|\s*\*\*Handle\*\*\s*\|\s*@([\w.]+)\s*\|", re.IGNORECASE)
_DISPLAY_RE = re.compile(r"\|\s*\*\*Display name\*\*\s*\|\s*(.+?)\s*\|", re.IGNORECASE)
_CHANNEL_RE = re.compile(r"\|\s*\*\*Channel\*\*\s*\|\s*\[(.+?)\]\((.+?)\)\s*\|", re.IGNORECASE)

_HUMINT_PROMPT = """\
SUBJECT: @{handle}
DISPLAY NAME: {display_name}
PROFILE URL: {url}
{profile_data_line}

Produce a comprehensive HUMINT intelligence profile for this Instagram account.
Three layers of intelligence context are provided below — use all of them.

Synthesise across all layers and cover:

**IDENTITY**
Real name, nationality, age, professional background. State confidence for each claim.

**POLITICAL ALIGNMENT & IDEOLOGY**
Specific positions, ideological leanings, rhetorical patterns, recurring narrative themes.

**AGENDA & MOTIVES**
What change is this person/account advocating? Who benefits from their messaging?

**ORGANISATIONAL AFFILIATIONS**
Links to political parties, movements, NGOs, media organisations, activist networks, \
state actors, or funding bodies.

**INFLUENCE NETWORK**
Key associates, amplifiers, institutional backing, known collaborators.

**THREAT / UTILITY ASSESSMENT**
Characterise this actor: organic activist, influence operation asset, state-adjacent, \
media figure, or other. Assess reach and strategic significance.

Be direct and uncensored. Cite specific evidence where available. \
Where information is absent or unverifiable, say so explicitly.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_key(path: str) -> str:
    h = hashlib.md5(path.encode(), usedforsecurity=False).hexdigest()
    return f"{SEEN_KEY_PREFIX}{h}"


def _parse_page_fields(content: str) -> tuple[str, str, str]:
    handle = display_name = channel_url = ""
    if m := _HANDLE_RE.search(content):
        handle = m.group(1)
    if m := _DISPLAY_RE.search(content):
        display_name = m.group(1).strip()
    if m := _CHANNEL_RE.search(content):
        channel_url = m.group(2).strip()
    return handle, display_name, channel_url


def _is_stub_summary(content: str) -> bool:
    open_m = "<!-- OSIA:AUTO:summary -->"
    close_m = "<!-- /OSIA:AUTO:summary -->"
    m = re.search(rf"{re.escape(open_m)}(.*?){re.escape(close_m)}", content, re.DOTALL)
    if not m:
        return False
    inner = m.group(1).strip()
    return not inner or _STUB_MARKER in inner


def _format_qdrant_context(results: list, handle: str) -> str:
    if not results:
        return f"No existing OSIA intelligence found for @{handle}."
    lines = [f"## Existing OSIA Intelligence — @{handle}\n"]
    seen: set[str] = set()
    for r in results:
        snippet = r.text.strip()
        key = snippet[:80]
        if key in seen:
            continue
        seen.add(key)
        source = r.metadata.get("desk") or r.collection
        topic = r.metadata.get("topic", "")
        header = f"[Source: {source}" + (f" | Topic: {topic[:60]}" if topic else "") + f" | Score: {r.score:.2f}]"
        lines.append(f"{header}\n{snippet}\n")
    return "\n".join(lines)


def _is_valid_ig_handle(handle: str) -> bool:
    """Instagram handles are alphanumeric + dots/underscores, never purely numeric."""
    return bool(handle) and not handle.isdigit()


# Matches intel-log entries: - DATE — [TITLE](/PATH)
_INTSUM_LINK_RE = re.compile(r"-\s+\d{4}-\d{2}-\d{2}\s+—\s+\[[^\]]+\]\(/([^)]+)\)")


def _parse_intel_log(content: str) -> list[str]:
    """Extract valid INTSUM wiki paths from the intel-log AUTO section."""
    open_m = "<!-- OSIA:AUTO:intel-log -->"
    close_m = "<!-- /OSIA:AUTO:intel-log -->"
    m = re.search(rf"{re.escape(open_m)}(.*?){re.escape(close_m)}", content, re.DOTALL)
    if not m:
        return []
    paths = []
    for match in _INTSUM_LINK_RE.finditer(m.group(1)):
        path = match.group(1).strip().lstrip("/")
        # Skip broken initial entries (path is "/" or empty)
        if path and path != "/" and path.startswith("desks/"):
            paths.append(path)
    return paths


def _extract_auto_section(content: str, section: str) -> str:
    """Return inner text of an OSIA:AUTO-fenced section, empty string if absent."""
    open_m = f"<!-- OSIA:AUTO:{section} -->"
    close_m = f"<!-- /OSIA:AUTO:{section} -->"
    m = re.search(
        rf"{re.escape(open_m)}(.*?){re.escape(close_m)}",
        content,
        re.DOTALL,
    )
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class ProfileInvestigationWorker:
    def __init__(self, redis_client: aioredis.Redis) -> None:
        self.redis = redis_client
        self._ig_pool = InstagramAccountManager(redis_client)
        self._desk = DeskRegistry()
        self._qdrant = QdrantStore()

    # ------------------------------------------------------------------
    # Step 1: Fetch the actual Instagram profile via instaloader
    # ------------------------------------------------------------------

    async def _fetch_profile(self, handle: str) -> dict | None:
        """
        Fetch profile metadata using instaloader with the Instagram account pool.

        Returns a dict with: display_name, bio, external_link, followers, posts,
        verified.

        The yt-dlp instagram:user extractor is currently broken; instaloader
        loads the same Netscape cookies from the pool via MozillaCookieJar.
        Post content comes from Qdrant (step 2) — get_posts() is skipped
        because it hits authenticated GraphQL which our pool cookies don't
        fully support.
        """
        cookie_result = await self._ig_pool.get_active_cookie_path()

        def _do_fetch() -> dict | None:
            from http.cookiejar import MozillaCookieJar

            import instaloader

            L = instaloader.Instaloader(quiet=True, request_timeout=30)

            if cookie_result:
                _, cookie_path = cookie_result
                jar = MozillaCookieJar(str(cookie_path))
                try:
                    jar.load(ignore_discard=True, ignore_expires=True)
                    L.context._session.cookies.update(jar)
                except Exception as exc:
                    logger.debug("[profile-inv] Cookie load warning for @%s: %s", handle, exc)

            try:
                profile = instaloader.Profile.from_username(L.context, handle)
            except Exception as exc:
                logger.warning("[profile-inv] instaloader profile fetch failed for @%s: %s", handle, exc)
                return None

            # Don't call get_posts() — it uses authenticated GraphQL which our
            # pool cookies don't fully support, causing 403/401 noise. Post
            # content is already available in Qdrant from reel ingestion (step 2).
            return {
                "display_name": profile.full_name or handle,
                "bio": profile.biography or "",
                "external_link": profile.external_url or "",
                "followers": str(profile.followers),
                "posts": str(profile.mediacount),
                "verified": profile.is_verified,
            }

        try:
            result = await asyncio.wait_for(asyncio.to_thread(_do_fetch), timeout=90)
            if result:
                logger.info(
                    "[profile-inv] instaloader fetched @%s — followers=%s bio_len=%d posts=%s",
                    handle,
                    result.get("followers"),
                    len(result.get("bio") or ""),
                    result.get("posts", "?"),
                )
            return result
        except TimeoutError:
            logger.warning("[profile-inv] instaloader timed out for @%s", handle)
            return None
        except Exception as exc:
            logger.warning("[profile-inv] instaloader error for @%s: %s", handle, exc)
            return None

    # ------------------------------------------------------------------
    # Step 2: Qdrant search across HUMINT-relevant collections
    # ------------------------------------------------------------------

    async def _fetch_intsum_intel(self, wiki: "WikiClient", intsum_paths: list[str]) -> str:
        """
        Fetch INTSUM wiki pages linked from the intel-log and extract their content.
        These contain the full OSIA analysis of specific reels that this account posted,
        making them the richest possible source of context about the account's content
        and narrative focus.
        """
        if not intsum_paths:
            return ""

        excerpts: list[str] = []
        for path in intsum_paths[:5]:  # cap at 5 most recent
            try:
                page = await wiki.get_page(path)
                if not page or not page.get("content"):
                    continue
                intel = _extract_auto_section(page["content"], "content")
                if intel:
                    # Trim very long INTSUMs but keep meaningful context
                    excerpts.append(f"**INTSUM: {page['title']}**\n\n{intel[:3000]}")
            except Exception as exc:
                logger.debug("[profile-inv] Failed to fetch INTSUM %s: %s", path, exc)

        if not excerpts:
            return ""
        logger.info("[profile-inv] Loaded %d INTSUM(s) from intel-log", len(excerpts))
        return "\n\n---\n\n".join(excerpts)

    async def _search_qdrant(self, handle: str, display_name: str, profile: dict | None) -> str:
        """
        Semantic search across HUMINT-relevant Qdrant collections.
        Query is enriched with bio text and display name so content from
        ingested reels surfaces via semantic similarity.
        """
        name = (profile.get("display_name") if profile else None) or display_name or handle
        bio_snippet = (profile.get("bio") or "")[:150] if profile else ""

        query_parts = [f"@{handle}"]
        if name and name.lower() != handle.lower():
            query_parts.append(name)
        if bio_snippet:
            query_parts.append(bio_snippet)
        query = " ".join(query_parts)

        try:
            results = await self._qdrant.cross_desk_search(
                query=query,
                top_k=QDRANT_TOP_K,
                decay_half_life_days=QDRANT_DECAY_DAYS,
                collections=_HUMINT_COLLECTIONS,
            )
            logger.info("[profile-inv] Qdrant returned %d results for @%s", len(results), handle)
            return _format_qdrant_context(results, handle)
        except Exception as exc:
            logger.warning("[profile-inv] Qdrant search error for @%s: %s", handle, exc)
            return f"Qdrant search unavailable: {exc}"

    # ------------------------------------------------------------------
    # Step 3: Targeted web searches (informed by profile data)
    # ------------------------------------------------------------------

    async def _ddg_search(self, query: str) -> str:
        try:
            from ddgs import DDGS

            results = await asyncio.to_thread(lambda: list(DDGS().text(query, max_results=5)))
            if not results:
                return "No results."
            return "\n\n".join(
                f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')[:400]}" for r in results
            )
        except Exception as exc:
            return f"Search error: {exc}"

    async def _gather_web_intel(self, handle: str, display_name: str, profile: dict | None) -> str:
        """
        Three targeted searches, informed by what yt-dlp told us about the profile.
        Bio text and display name produce far better results than a bare username.
        """
        name = (profile.get("display_name") if profile else None) or display_name or handle
        bio = (profile.get("bio") or "") if profile else ""
        external_link = (profile.get("external_link") or "") if profile else ""

        # Query 1: Identity — real name + handle
        q1 = f'"{name}" Instagram @{handle}'

        # Query 2: Political/ideological stance — use bio keywords if available
        bio_keywords = [w for w in bio.split() if len(w) > 3 and w.isalpha()][:5]
        q2 = f'"{name}" political views OR activism OR ideology OR beliefs'
        if bio_keywords:
            q2 = f'"{name}" {" ".join(bio_keywords)} political OR ideology OR movement'

        # Query 3: Affiliations — external link domain is a strong signal
        q3 = f'"{name}" organisation OR affiliation OR funding OR background'
        if external_link:
            domain = re.sub(r"https?://(?:www\.)?", "", external_link).split("/")[0]
            if domain:
                q3 = f'"{name}" "{domain}" OR affiliation OR organisation'

        sections = []
        for q in (q1, q2, q3):
            result = await self._ddg_search(q)
            sections.append(f"### Search: {q}\n\n{result}")
        return "\n\n---\n\n".join(sections)

    # ------------------------------------------------------------------
    # Step 4: HUMINT desk analysis
    # ------------------------------------------------------------------

    async def _analyse(
        self,
        handle: str,
        display_name: str,
        channel_url: str,
        profile: dict | None,
        intsum_intel: str,
        qdrant_context: str,
        web_intel: str,
    ) -> str:
        profile_lines: list[str] = []
        if profile:
            if profile.get("bio"):
                profile_lines.append(f"Bio: {profile['bio']}")
            if profile.get("external_link"):
                profile_lines.append(f"External link: {profile['external_link']}")
            metrics = [f"followers: {profile['followers']}"] if profile.get("followers") else []
            if profile.get("posts"):
                metrics.append(f"posts: {profile['posts']}")
            if profile.get("verified"):
                metrics.append("verified: yes")
            if metrics:
                profile_lines.append(" | ".join(metrics))

        name = (profile.get("display_name") if profile else None) or display_name or handle
        prompt = _HUMINT_PROMPT.format(
            handle=handle,
            display_name=name,
            url=channel_url or f"https://www.instagram.com/{handle}/",
            profile_data_line=("\n" + "\n".join(profile_lines)) if profile_lines else "",
        )

        layers: list[str] = []
        if intsum_intel:
            layers += ["# LAYER 1 — OSIA INTSUM ARCHIVE (previously ingested reels from this account)", intsum_intel]
        layers += [
            "# LAYER 2 — QDRANT SEMANTIC INTELLIGENCE",
            qdrant_context,
            "# LAYER 3 — WEB RESEARCH",
            web_intel,
        ]

        analysis, meta = await self._desk.invoke(
            HUMINT_DESK,
            user_message=prompt,
            context_block="\n\n".join(layers),
        )
        logger.info("[profile-inv] HUMINT analysis complete for @%s via %s", handle, meta.get("model_id"))
        return analysis

    # ------------------------------------------------------------------
    # Step 5: Write directly to wiki
    # ------------------------------------------------------------------

    async def _write_wiki(self, wiki: "WikiClient", path: str, handle: str, analysis: str, page_content: str) -> None:
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        ok = await wiki.patch_section(path, "summary", analysis)
        if not ok:
            logger.warning("[profile-inv] Wiki summary patch failed for @%s", handle)
            return

        note = f"- **{today}** — HUMINT desk profile investigation completed"
        existing_notes = _extract_auto_section(page_content, "research-notes")
        if "No research notes on file" in existing_notes:
            # Replace the stub entirely rather than appending below it
            await wiki.patch_section(path, "research-notes", f"## Research Notes\n\n{note}")
        else:
            await wiki.append_to_section(path, "research-notes", note)
        logger.info("[profile-inv] Wiki updated for @%s", handle)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("[profile-inv] Starting — max_per_run=%d cooldown=%dd", MAX_PER_RUN, COOLDOWN_DAYS)

        async with WikiClient() as wiki:
            pages = await wiki.list_pages(path_prefix="entities/social-accounts/instagram")

        if not pages:
            logger.info("[profile-inv] No Instagram profile pages found in wiki.")
            return

        logger.info("[profile-inv] %d total Instagram profile page(s) in wiki.", len(pages))
        investigated = skipped_seen = skipped_done = 0

        async with WikiClient() as wiki:
            for page_stub in pages:
                if investigated >= MAX_PER_RUN:
                    logger.info("[profile-inv] Reached MAX_PER_RUN=%d — stopping.", MAX_PER_RUN)
                    break

                path = page_stub.get("path", "")
                if not path:
                    continue

                # Skip numeric IDs without fetching the page — the slug is the last path segment
                slug = path.split("/")[-1]
                if not _is_valid_ig_handle(slug):
                    continue

                seen_key = _path_key(path)
                if await self.redis.exists(seen_key):
                    skipped_seen += 1
                    continue

                page = await wiki.get_page(path)
                if not page or not page.get("content"):
                    continue

                content = page["content"]
                if not _is_stub_summary(content):
                    skipped_done += 1
                    continue

                handle, display_name, channel_url = _parse_page_fields(content)
                if not handle or not _is_valid_ig_handle(handle):
                    logger.warning("[profile-inv] Could not extract valid handle from %s", path)
                    continue

                logger.info(
                    "[profile-inv] Investigating @%s (%s) [%d/%d]",
                    handle,
                    display_name or "—",
                    investigated + 1,
                    MAX_PER_RUN,
                )

                try:
                    # Step 1: Fetch the actual Instagram profile (instaloader)
                    profile = await self._fetch_profile(handle)

                    # Step 2: Fetch INTSUM content from the intel-log — the richest
                    # source of context, containing full analysis of their reels
                    intsum_paths = _parse_intel_log(content)
                    intsum_intel = await self._fetch_intsum_intel(wiki, intsum_paths)

                    # Step 3: Qdrant semantic search — existing OSIA intelligence
                    qdrant_context = await self._search_qdrant(handle, display_name, profile)

                    # Step 4: Targeted web searches (informed by bio + display name)
                    web_intel = await self._gather_web_intel(handle, display_name, profile)

                    # Step 5: HUMINT desk analysis — all layers
                    analysis = await self._analyse(
                        handle,
                        display_name,
                        channel_url,
                        profile,
                        intsum_intel,
                        qdrant_context,
                        web_intel,
                    )

                    # Step 6: Write directly to wiki
                    await self._write_wiki(wiki, path, handle, analysis, content)

                    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
                    await self.redis.setex(seen_key, SEEN_TTL, today)
                    investigated += 1

                except Exception as exc:
                    logger.error("[profile-inv] Failed on @%s: %s", handle, exc, exc_info=True)

        logger.info(
            "[profile-inv] Done. investigated=%d skipped_seen=%d skipped_done=%d",
            investigated,
            skipped_seen,
            skipped_done,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = aioredis.from_url(redis_url, decode_responses=False)
    try:
        worker = ProfileInvestigationWorker(redis_client)
        await worker.run()
    finally:
        await worker._desk.close()
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(_main())
