"""
One-time script: resolve numeric Instagram user IDs to real usernames and
move the corresponding wiki pages to the correct paths.

Background: the orchestrator was storing Instagram uploader_id (a numeric
internal user ID) as the handle when creating wiki pages, resulting in pages
at paths like entities/social-accounts/instagram/11648124508 instead of the
correct entities/social-accounts/instagram/grayzonenews.

This script:
  1. Lists all Instagram social-account wiki pages with numeric slugs.
  2. Resolves each numeric ID to a real username via instaloader.
  3. Checks the destination path doesn't already exist.
  4. Moves the wiki page to the correct path.
  5. Patches the page content to fix the handle field, channel URL, and title.

Usage:
  uv run python scripts/fix_numeric_ig_handles.py [--dry-run] [--limit N] [--delay S]

Flags:
  --dry-run    Print what would happen without making any changes.
  --limit N    Process at most N pages (default: all).
  --delay S    Seconds to sleep between instaloader lookups (default: 3).
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import redis.asyncio as aioredis  # noqa: E402

from src.agents.instagram_account_manager import InstagramAccountManager  # noqa: E402
from src.intelligence.wiki_client import WikiClient, social_account_wiki_path  # noqa: E402

logger = logging.getLogger("fix_numeric_handles")

_HANDLE_ROW_RE = re.compile(r"(\|\s*\*\*Handle\*\*\s*\|\s*)@[\w.]+(\s*\|)", re.IGNORECASE)
_CHANNEL_ROW_RE = re.compile(
    r"(\|\s*\*\*Channel\*\*\s*\|\s*\[)https://www\.instagram\.com/[\w.]+(\]\(https://www\.instagram\.com/)[\w.]+(\)\s*\|)",
    re.IGNORECASE,
)


def _patch_content(content: str, old_handle: str, new_handle: str) -> str:
    """Fix handle and channel URL in the profile table."""
    # | **Handle** | @old |
    content = _HANDLE_ROW_RE.sub(rf"\g<1>@{new_handle}\g<2>", content)
    # | **Channel** | [url](url) |
    new_url = f"https://www.instagram.com/{new_handle}"
    content = _CHANNEL_ROW_RE.sub(rf"\g<1>{new_url}\g<2>{new_handle}\g<3>", content)
    # Any plain @old_handle references in the body (e.g. summary placeholder or stubs)
    content = content.replace(f"@{old_handle}", f"@{new_handle}")
    return content


def _resolve_id(numeric_id: str, cookie_path: str | None, request_timeout: int = 30) -> dict | None:
    """
    Resolve an Instagram numeric user ID to a username using the private REST API.

    Instagram's GraphQL endpoint used by instaloader's Profile.from_id() is broken
    (returns 400 for query_hash lookups). The private API endpoint at
    i.instagram.com/api/v1/users/{id}/info/ still works with valid session cookies.
    """
    import requests
    from http.cookiejar import MozillaCookieJar

    session = requests.Session()
    session.headers.update(
        {
            # Mobile client UA required for the private API
            "User-Agent": (
                "Instagram 276.0.0.19.105 Android "
                "(29/10; 380dpi; 1080x2340; Google; sdk_gphone_x86; generic_x86; en_US)"
            ),
            "x-ig-app-id": "567067343352427",
            "Accept-Language": "en-US",
        }
    )

    if cookie_path:
        jar = MozillaCookieJar(cookie_path)
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(jar)
        except Exception as exc:
            logger.debug("Cookie load warning: %s", exc)

    try:
        resp = session.get(
            f"https://i.instagram.com/api/v1/users/{numeric_id}/info/",
            timeout=request_timeout,
        )
        if not resp.ok:
            logger.warning("Private API %d for ID %s: %s", resp.status_code, numeric_id, resp.text[:200])
            return None
        user = resp.json().get("user", {})
        username = user.get("username")
        if not username:
            logger.warning("No username in API response for ID %s", numeric_id)
            return None
        return {
            "username": username,
            "full_name": user.get("full_name") or username,
        }
    except Exception as exc:
        logger.warning("Could not resolve ID %s: %s", numeric_id, exc)
        return None


async def run(dry_run: bool, limit: int | None, delay: float) -> None:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = aioredis.from_url(redis_url, decode_responses=False)
    ig_pool = InstagramAccountManager(redis_client)

    # Get a cookie file from the pool (reuse same one throughout to avoid cycling)
    cookie_result = await ig_pool.get_active_cookie_path()
    cookie_path = str(cookie_result[1]) if cookie_result else None
    if cookie_path:
        logger.info("Using cookie from account %s", cookie_result[0])
    else:
        logger.warning("No active Instagram cookies — resolution may hit rate limits")

    async with WikiClient() as wiki:
        pages = await wiki.list_pages(path_prefix="entities/social-accounts/instagram")

    numeric_pages = [p for p in pages if (p.get("path") or "").split("/")[-1].isdigit()]

    logger.info("Found %d numeric-ID pages out of %d total", len(numeric_pages), len(pages))

    if limit:
        numeric_pages = numeric_pages[:limit]
        logger.info("Processing first %d", limit)

    resolved = failed = skipped_exists = 0

    for i, page_stub in enumerate(numeric_pages, 1):
        old_path = page_stub["path"]
        numeric_id = old_path.split("/")[-1]
        page_title = page_stub.get("title", f"@{numeric_id}")

        logger.info("[%d/%d] Resolving ID %s (%s)…", i, len(numeric_pages), numeric_id, page_title)

        # Resolve ID → username (synchronous instaloader call in thread)
        result = await asyncio.to_thread(_resolve_id, numeric_id, cookie_path)

        if not result:
            logger.warning("  ✗ Could not resolve %s — skipping", numeric_id)
            failed += 1
            await asyncio.sleep(delay)
            continue

        username = result["username"]
        full_name = result["full_name"]
        new_path = social_account_wiki_path("instagram", username)

        logger.info("  → @%s (%s)  [%s → %s]", username, full_name, old_path, new_path)

        if dry_run:
            logger.info("  [DRY RUN] Would move and patch page")
            resolved += 1
            await asyncio.sleep(delay)
            continue

        async with WikiClient() as wiki:
            # Check destination doesn't already exist
            existing_dest = await wiki.get_page(new_path)
            if existing_dest:
                logger.warning(
                    "  ✗ Destination %s already exists (id=%s) — skipping move",
                    new_path,
                    existing_dest["id"],
                )
                skipped_exists += 1
                await asyncio.sleep(delay)
                continue

            # Fetch full page content
            page = await wiki.get_page(old_path)
            if not page:
                logger.warning("  ✗ Could not fetch page at %s", old_path)
                failed += 1
                await asyncio.sleep(delay)
                continue

            page_id = page["id"]

            # Move the page
            moved = await wiki.move_page(page_id, new_path)
            if not moved:
                logger.warning("  ✗ Move failed for page %d", page_id)
                failed += 1
                await asyncio.sleep(delay)
                continue

            # Patch content: fix handle and channel URL
            new_content = _patch_content(page["content"], numeric_id, username)
            new_title = f"@{username}"
            new_desc = f"Instagram — @{username} — {page.get('description', '').split('—', 1)[-1].strip()}"

            updated = await wiki.update_page(
                page_id,
                new_content,
                new_title,
                new_desc,
                page.get("tags") or ["social-account", "instagram", "intel-source"],
            )
            if updated:
                logger.info("  ✓ Moved and patched → @%s", username)
                resolved += 1
            else:
                logger.warning("  ⚠ Moved but content patch failed for @%s", username)
                resolved += 1  # move succeeded, content patch is best-effort

        await asyncio.sleep(delay)

    await redis_client.aclose()

    logger.info(
        "\nDone. resolved=%d  failed=%d  skipped_already_exist=%d",
        resolved,
        failed,
        skipped_exists,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print actions without making changes")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to process")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between instaloader lookups")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        logger.info("DRY RUN — no changes will be made")

    asyncio.run(run(dry_run=args.dry_run, limit=args.limit, delay=args.delay))


if __name__ == "__main__":
    main()
