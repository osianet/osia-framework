"""
Instagram account warm-up via Camoufox browser automation.

Reuses the account's existing session cookies — no re-login, no 2FA.
Cookie content is read from and written back to Redis (no disk files needed).
Each session: browse feed → like 1-2 posts → follow intel sources → save fresh cookies.
Optionally uploads a generated avatar as profile picture on first session.
"""

import asyncio
import logging
import random
import tempfile
import time
from pathlib import Path

import httpx
import redis.asyncio as redis
from camoufox.async_api import AsyncCamoufox

from src.agents.instagram_account_manager import InstagramAccount, InstagramAccountManager

logger = logging.getLogger("osia.ig_warmup")

_INTEL_SOURCES_KEY = "osia:ig:intel_sources"

_COUNTRY_PROFILE: dict[str, tuple[str, str]] = {
    "AU": ("en-AU", "Australia/Sydney"),
    "US": ("en-US", "America/New_York"),
    "UK": ("en-GB", "Europe/London"),
    "GB": ("en-GB", "Europe/London"),
    "CA": ("en-CA", "America/Toronto"),
    "SG": ("en-SG", "Asia/Singapore"),
    "NZ": ("en-NZ", "Pacific/Auckland"),
}

# DiceBear avatar style — lorelei gives illustrated portrait-style avatars
_DICEBEAR_URL = "https://api.dicebear.com/7.x/lorelei/png"


async def _pause(lo: float, hi: float) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


def _parse_netscape_cookies(content: str) -> list[dict]:
    """Parse Netscape cookie string → list of Playwright-compatible cookie dicts."""
    cookies = []
    for line in content.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain, _flag, path, secure, expires_str, name, value = parts[:7]
        try:
            expires = int(expires_str)
        except ValueError:
            expires = -1
        cookies.append(
            {
                "domain": domain,
                "path": path,
                "secure": secure.upper() == "TRUE",
                "expires": expires if expires > 0 else -1,
                "name": name,
                "value": value,
            }
        )
    return cookies


async def _dump_cookies_netscape(page) -> str:
    """Export instagram.com session cookies from the browser as a Netscape format string."""
    cookies = await page.context.cookies(["https://www.instagram.com"])
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c.get("domain", ".instagram.com")
        if not domain.startswith("."):
            domain = f".{domain}"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure") else "FALSE"
        expiry = int(c["expires"]) if c.get("expires", -1) >= 0 else 0
        lines.append(f"{domain}\tTRUE\t{path}\t{secure}\t{expiry}\t{c.get('name', '')}\t{c.get('value', '')}")
    return "\n".join(lines) + "\n"


async def _dismiss_overlays(page) -> None:
    """Dismiss cookie banners, notification prompts, and login nudges."""
    for label in [
        "Allow all cookies",
        "Accept all",
        "Only allow essential cookies",
        "Not now",
        "Turn on Notifications",
        "Not Now",
    ]:
        try:
            btn = page.get_by_role("button", name=label, exact=True).first
            if await btn.is_visible(timeout=1_200):
                await btn.click()
                await _pause(0.4, 0.8)
        except Exception:  # noqa: BLE001 — best-effort: overlay may not exist, non-fatal
            pass


class InstagramWarmupSession:
    """
    Single Camoufox warm-up session for one Instagram account.

    Loads cookies from Redis, runs the session, saves fresh cookies back to Redis.
    No disk files required — the account manager materialises a temp file for yt-dlp
    separately when the orchestrator needs it.
    """

    def __init__(
        self,
        manager: InstagramAccountManager,
        redis_client: redis.Redis,
        headed: bool = False,
        upload_avatar: bool = True,
    ):
        self._manager = manager
        self._redis = redis_client
        self._headed = headed
        self._upload_avatar = upload_avatar

    async def run(self, account_id: str) -> bool:
        """Execute a complete warm-up session. Returns True on success."""
        account = await self._manager.get(account_id)
        if not account:
            raise ValueError(f"Account {account_id} not found")

        if account.state not in ("WARMING", "ACTIVE"):
            logger.warning("Account %s is %s — skipping", account_id, account.state)
            return False

        content = await self._manager.get_cookie_content(account_id)
        if not content:
            logger.warning("No cookie content in Redis for @%s — import cookies first", account.username)
            return False

        cookies = _parse_netscape_cookies(content)
        if not cookies:
            logger.warning("Cookie content unparseable for @%s", account.username)
            return False

        locale, _ = _COUNTRY_PROFILE.get(account.vpn_country.upper(), ("en-US", "America/New_York"))
        logger.info("Starting warm-up for @%s (id=%s…)", account.username, account_id[:8])

        async with AsyncCamoufox(
            headless=not self._headed,
            geoip=True,
            locale=locale,
            os="windows",
        ) as browser:
            context = await browser.new_context(viewport={"width": 1280, "height": 900})
            await context.add_cookies(cookies)
            page = await context.new_page()
            success = False
            try:
                success = await self._run_session(page, account)
            except Exception as exc:
                logger.error("Warm-up failed for @%s: %s", account.username, exc)
            finally:
                try:
                    fresh = await _dump_cookies_netscape(page)
                    await self._manager.set_cookie_content(account_id, fresh)
                    logger.info("Refreshed cookies in Redis for @%s", account.username)
                except Exception as exc:
                    logger.warning("Cookie save failed: %s", exc)
                await context.close()

        if success:
            await self._manager.record_warmup_session(account_id)
            logger.info("Session recorded for @%s (total=%d)", account.username, account.warmup_sessions + 1)
        return success

    async def _run_session(self, page, account: InstagramAccount) -> bool:
        logged_in, suspended = await self._check_session(page, account.username)
        if suspended:
            await self._manager.flag(account.id, reason="suspended")
            logger.error("@%s is suspended — flagged", account.username)
            return False
        if not logged_in:
            logger.warning("Session expired for @%s — cookies invalid", account.username)
            return False

        if self._upload_avatar and not account.has_profile_pic:
            await self._upload_profile_pic(page, account)

        await self._browse_feed(page, duration_secs=random.randint(180, 360))

        likes = await self._like_posts(page, count=random.randint(1, 2))
        logger.info("Liked %d posts for @%s", likes, account.username)

        handles = await self._pick_intel_handles(count=random.randint(2, 3))
        if handles:
            followed = await self._follow_accounts(page, handles)
            logger.info("Followed %d accounts for @%s", followed, account.username)

        return True

    async def _check_session(self, page, username: str) -> tuple[bool, bool]:
        """Navigate to Instagram. Returns (logged_in, suspended)."""
        await page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:  # noqa: BLE001 — networkidle timeout is non-fatal; domcontentloaded already awaited
            pass
        await _dismiss_overlays(page)
        await _pause(1.5, 3.0)

        url = page.url
        if "accounts/suspended" in url:
            return False, True
        if "accounts/login" in url:
            return False, False

        logger.info("Session active for @%s", username)
        return True, False

    async def _browse_feed(self, page, duration_secs: int) -> None:
        """Scroll the Instagram home feed for ~duration_secs seconds."""
        await page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
        await _pause(2.0, 4.0)
        await _dismiss_overlays(page)

        deadline = time.time() + duration_secs
        while time.time() < deadline:
            await page.mouse.wheel(0, random.randint(300, 700))
            pause = random.uniform(2.0, 6.0)
            await asyncio.sleep(min(pause, max(0.1, deadline - time.time())))

    async def _like_posts(self, page, count: int = 2) -> int:
        """Like `count` posts visible in the feed."""
        liked = 0
        await page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
        await _pause(2.0, 4.0)
        await _dismiss_overlays(page)

        for _ in range(random.randint(3, 5)):
            await page.mouse.wheel(0, random.randint(400, 700))
            await _pause(1.5, 3.0)

        try:
            like_btns = await page.query_selector_all('article button[aria-label="Like"]')
            if not like_btns:
                like_btns = await page.query_selector_all('svg[aria-label="Like"]')

            skip = random.randint(1, 3)
            candidates = like_btns[skip : skip + count + 4]
            random.shuffle(candidates)

            for btn in candidates[:count]:
                try:
                    await btn.scroll_into_view_if_needed()
                    await _pause(0.8, 2.5)
                    await btn.click()
                    await _pause(1.0, 2.5)
                    liked += 1
                except Exception as exc:
                    logger.debug("Like click skipped: %s", exc)
        except Exception as exc:
            logger.warning("_like_posts error: %s", exc)

        return liked

    async def _pick_intel_handles(self, count: int = 3) -> list[str]:
        """Pick random handles from the osia:ig:intel_sources Redis set."""
        try:
            members = await self._redis.srandmember(_INTEL_SOURCES_KEY, count)
            return [m.decode() if isinstance(m, bytes) else str(m) for m in (members or [])]
        except Exception as exc:
            logger.warning("Could not fetch intel handles: %s", exc)
            return []

    async def _follow_accounts(self, page, handles: list[str]) -> int:
        """Navigate to each profile and click Follow."""
        followed = 0
        for handle in handles:
            try:
                await page.goto(
                    f"https://www.instagram.com/{handle}/",
                    wait_until="domcontentloaded",
                    timeout=20_000,
                )
                await _pause(2.0, 4.0)
                await _dismiss_overlays(page)

                follow_btn = page.get_by_role("button", name="Follow", exact=True).first
                if await follow_btn.is_visible(timeout=4_000):
                    await follow_btn.click()
                    await _pause(1.0, 2.5)
                    followed += 1
                    logger.info("Followed @%s", handle)
                else:
                    logger.debug("@%s — already following or Follow not visible", handle)
            except Exception as exc:
                logger.warning("Follow @%s failed: %s", handle, exc)
        return followed

    async def _upload_profile_pic(self, page, account: InstagramAccount) -> bool:
        """
        Fetch a DiceBear avatar seeded from the account ID and upload it
        as the Instagram profile picture. Avatar is stored in tmp/ and discarded after.
        """
        avatar_path = Path(tempfile.gettempdir()) / f"osia_ig_avatar_{account.id}.png"

        try:
            async with httpx.AsyncClient(timeout=20) as http:
                r = await http.get(
                    _DICEBEAR_URL,
                    params={"seed": account.id, "size": "400", "backgroundColor": "b6e3f4,c0aede,d1d4f9"},
                )
                r.raise_for_status()
                avatar_path.write_bytes(r.content)
            logger.info("Avatar downloaded for @%s", account.username)
        except Exception as exc:
            logger.warning("Avatar download failed for @%s: %s", account.username, exc)
            return False

        try:
            await page.goto(
                "https://www.instagram.com/accounts/edit/",
                wait_until="domcontentloaded",
                timeout=20_000,
            )
            await _pause(2.0, 3.5)
            await _dismiss_overlays(page)

            try:
                change_link = page.get_by_text("Change profile photo", exact=True).first
                if await change_link.is_visible(timeout=4_000):
                    await change_link.click()
                    await _pause(0.8, 1.5)
            except Exception:  # noqa: BLE001 — best-effort: link absent on some IG UI variants
                pass

            file_input = page.locator('input[type="file"]').first
            await file_input.set_input_files(str(avatar_path))
            await _pause(2.5, 4.5)

            for label in ["Done", "Apply", "Save", "Next", "Confirm"]:
                try:
                    btn = page.get_by_role("button", name=label, exact=True).first
                    if await btn.is_visible(timeout=3_000):
                        await btn.click()
                        await _pause(1.0, 2.0)
                        break
                except Exception:  # noqa: BLE001 — best-effort: confirm button varies by IG UI version
                    pass

            await self._manager.mark_has_profile_pic(account.id)
            logger.info("Profile picture uploaded for @%s", account.username)
            return True

        except Exception as exc:
            logger.warning("Profile pic upload failed for @%s: %s", account.username, exc)
            return False
        finally:
            try:
                avatar_path.unlink(missing_ok=True)
            except Exception:
                pass
