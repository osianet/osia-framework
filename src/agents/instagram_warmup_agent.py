"""
Instagram account warm-up and re-login via Camoufox browser automation.

Warmup: reuses existing session cookies — browse feed, like, follow, refresh cookies.
Relogin: opens headed browser, fills stored credentials, user clears any checkpoint,
         exports fresh sessionid cookies back to Redis.

Cookie content is stored in and written back to Redis (no disk files required).
"""

import asyncio
import logging
import random
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import redis.asyncio as redis
from camoufox.async_api import AsyncCamoufox

from src.agents.instagram_account_manager import InstagramAccount, InstagramAccountManager

logger = logging.getLogger("osia.ig_warmup")

_INTEL_SOURCES_KEY = "osia:ig:intel_sources"

_WG_CONF = Path("/etc/wireguard/wg0.conf")
_COUNTRIES_DIR = Path("/etc/wireguard/countries")

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
                "httpOnly": False,
                "sameSite": "None" if secure.upper() == "TRUE" else "Lax",
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
        wg_conf: Path = _WG_CONF,
        countries_dir: Path = _COUNTRIES_DIR,
        debug: bool = False,
    ):
        self._manager = manager
        self._redis = redis_client
        self._headed = headed
        self._upload_avatar = upload_avatar
        self._wg_conf = wg_conf
        self._countries_dir = countries_dir
        self._debug = debug
        self._shot_counter = 0
        self._debug_dir: Path | None = None

    async def _shot(self, page, account_id: str, step: str) -> None:
        """Save a numbered screenshot when debug mode is on."""
        if not self._debug or self._debug_dir is None:
            return
        self._shot_counter += 1
        path = self._debug_dir / f"{self._shot_counter:02d}_{step}.png"
        try:
            await page.screenshot(path=str(path), full_page=False)
            logger.info("[DEBUG] screenshot → %s", path)
        except Exception as exc:
            logger.debug("Screenshot failed (%s): %s", step, exc)

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

        if self._debug:
            self._shot_counter = 0
            self._debug_dir = self._manager._base_dir / "logs" / "ig_debug" / account_id[:8]
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[DEBUG] screenshots → %s", self._debug_dir)

        original_slug = self._current_vpn_slug()
        target_slug: str | None = None
        try:
            target_slug = self._resolve_vpn_slug(account.vpn_country)
        except ValueError as exc:
            logger.warning("VPN switch skipped: %s", exc)

        if target_slug and target_slug != original_slug:
            logger.info("Switching VPN to %s (%s)", account.vpn_country, target_slug)
            await self._switch_vpn(target_slug)
        else:
            target_slug = None  # no switch happened; nothing to restore

        try:
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
                    success = await self._run_session(page, account, account_id)
                except Exception as exc:
                    logger.error("Warm-up failed for @%s: %s", account.username, exc)
                    await self._shot(page, account_id, "error")
                finally:
                    try:
                        fresh = await _dump_cookies_netscape(page)
                        await self._manager.set_cookie_content(account_id, fresh)
                        logger.info("Refreshed cookies in Redis for @%s", account.username)
                    except Exception as exc:
                        logger.warning("Cookie save failed: %s", exc)
                    await context.close()
        finally:
            if target_slug and original_slug:
                logger.info("Restoring VPN to %s", original_slug)
                try:
                    await self._switch_vpn(original_slug)
                except Exception as exc:
                    logger.warning("VPN restore failed: %s", exc)

        if success:
            await self._manager.record_warmup_session(account_id)
            logger.info("Session recorded for @%s (total=%d)", account.username, account.warmup_sessions + 1)
        return success

    # ------------------------------------------------------------------ relogin

    async def relogin(self, account_id: str) -> bool:
        """
        Open a headed Camoufox browser, fill stored credentials, wait for the
        user to clear any checkpoint/2FA, then export fresh cookies to Redis.
        Always runs headed — manual intervention may be required.
        Returns True if a valid sessionid was captured.
        """
        account = await self._manager.get(account_id)
        if not account:
            raise ValueError(f"Account {account_id} not found")

        locale, _ = _COUNTRY_PROFILE.get(account.vpn_country.upper(), ("en-US", "America/New_York"))
        logger.info("Starting re-login for @%s", account.username)

        if self._debug:
            self._shot_counter = 0
            self._debug_dir = self._manager._base_dir / "logs" / "ig_debug" / account_id[:8]
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[DEBUG] screenshots → %s", self._debug_dir)

        original_slug = self._current_vpn_slug()
        target_slug: str | None = None
        try:
            target_slug = self._resolve_vpn_slug(account.vpn_country)
        except ValueError as exc:
            logger.warning("VPN switch skipped: %s", exc)

        if target_slug and target_slug != original_slug:
            logger.info("Switching VPN to %s (%s)", account.vpn_country, target_slug)
            await self._switch_vpn(target_slug)
        else:
            target_slug = None

        success = False
        try:
            async with AsyncCamoufox(
                headless=False,  # always headed — user must handle checkpoints
                geoip=True,
                locale=locale,
                os="windows",
            ) as browser:
                context = await browser.new_context(viewport={"width": 1280, "height": 900})
                page = await context.new_page()
                try:
                    success = await self._run_login(page, account, account_id)
                except Exception as exc:
                    logger.error("Re-login failed for @%s: %s", account.username, exc)
                    await self._shot(page, account_id, "relogin_error")
                finally:
                    await context.close()
        finally:
            if target_slug and original_slug:
                logger.info("Restoring VPN to %s", original_slug)
                try:
                    await self._switch_vpn(original_slug)
                except Exception as exc:
                    logger.warning("VPN restore failed: %s", exc)

        return success

    async def _run_login(self, page, account: InstagramAccount, account_id: str) -> bool:
        """Fill login form, wait for user to clear any checkpoint, save cookies."""
        await page.goto(
            "https://www.instagram.com/",
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        try:
            await page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:  # noqa: BLE001
            pass
        await _pause(2.0, 3.0)
        await _dismiss_overlays(page)
        await self._shot(page, account_id, "relogin_landing_page")

        await self._shot(page, account_id, "relogin_login_page")

        # Instagram's desktop login form uses name="email" / name="pass".
        # Fall back to broader selectors in case of future changes.
        username_field = None
        for sel in [
            'input[name="email"]',
            'input[name="username"]',
            'input[autocomplete*="username"]',
            'input[type="text"]',
        ]:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=5_000):
                    username_field = el
                    break
            except Exception:  # noqa: BLE001
                pass

        if not username_field:
            await self._shot(page, account_id, "relogin_no_form")
            raise RuntimeError("Could not locate username input — check relogin_login_page.png")

        await username_field.click()
        for ch in account.username:
            await username_field.type(ch, delay=random.randint(40, 120))
        await _pause(0.4, 0.8)

        # Password field
        password_field = None
        for sel in ['input[name="pass"]', 'input[name="password"]', 'input[type="password"]']:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=3_000):
                    password_field = el
                    break
            except Exception:  # noqa: BLE001
                pass

        if not password_field:
            raise RuntimeError("Could not locate password input")

        await password_field.click()
        for ch in account.password:
            await password_field.type(ch, delay=random.randint(40, 120))
        await _pause(0.5, 1.0)
        await self._shot(page, account_id, "relogin_filled")

        # Submit — try button, then input[type=submit], then Enter key
        submitted = False
        for label in ["Log in", "Log In"]:
            try:
                btn = page.get_by_role("button", name=label, exact=True).first
                if await btn.is_visible(timeout=2_000):
                    await btn.click()
                    submitted = True
                    break
            except Exception:  # noqa: BLE001
                pass
        if not submitted:
            try:
                submit = page.locator('input[type="submit"], button[type="submit"]').first
                if await submit.is_visible(timeout=2_000):
                    await submit.click()
                    submitted = True
            except Exception:  # noqa: BLE001
                pass
        if not submitted:
            await password_field.press("Enter")
        await _pause(2.0, 4.0)
        await self._shot(page, account_id, "relogin_after_submit")

        # Wait for the user to resolve any checkpoint / 2FA — poll until we land
        # on the feed or a recognised post-login page (up to 5 minutes).
        print(
            f"\n>>> Browser open for @{account.username} — resolve any checkpoint/2FA then wait. <<<\n"
            ">>> Script will continue automatically once you reach the Instagram feed. <<<\n"
        )
        logger.info("Waiting for @%s to reach feed (up to 5 min)…", account.username)

        landed = False
        for _ in range(60):  # 60 × 5s = 5 minutes
            await asyncio.sleep(5)
            url = page.url
            # Dismiss post-login prompts on each poll
            await _dismiss_overlays(page)
            for label in ["Save Info", "Not Now", "Not now", "Skip", "Cancel"]:
                try:
                    btn = page.get_by_role("button", name=label, exact=True).first
                    if await btn.is_visible(timeout=500):
                        await btn.click()
                        await _pause(0.5, 1.0)
                except Exception:  # noqa: BLE001
                    pass
            # Check we've reached the feed or a profile page (i.e. no longer on login/challenge)
            if not any(kw in url for kw in ("login", "challenge", "checkpoint", "two_factor")):
                try:
                    on_feed = await page.locator('svg[aria-label="Home"]').first.is_visible(timeout=2_000)
                except Exception:  # noqa: BLE001
                    on_feed = False
                if on_feed:
                    landed = True
                    break

        await self._shot(page, account_id, "relogin_post_login")

        if not landed:
            logger.error("@%s never reached the feed — re-login timed out", account.username)
            return False

        # Export cookies and check sessionid is present
        fresh = await _dump_cookies_netscape(page)
        cookie_names = [
            line.split("\t")[5]
            for line in fresh.splitlines()
            if line and not line.startswith("#") and len(line.split("\t")) >= 7
        ]
        if "sessionid" not in cookie_names:
            logger.error("Re-login appeared to succeed but sessionid missing — not saving")
            return False

        await self._manager.set_cookie_content(account_id, fresh)
        logger.info(
            "Fresh cookies saved for @%s (%d cookies, sessionid present)",
            account.username,
            len(cookie_names),
        )

        # Unflag if it was flagged, ensure it's in ACTIVE state
        account = await self._manager.get(account_id)
        if account and account.state == "FLAGGED":
            await self._manager.unflag(account_id)
            await self._manager.promote(account_id)
        elif account and account.state in ("CREATED", "WARMING"):
            await self._manager.promote(account_id)

        return True

    async def _run_session(self, page, account: InstagramAccount, account_id: str) -> bool:
        logged_in, suspended = await self._check_session(page, account.username)
        await self._shot(page, account_id, "session_check")
        if suspended:
            await self._manager.flag(account.id, reason="suspended")
            logger.error("@%s is suspended — flagged", account.username)
            return False
        if not logged_in:
            logger.warning("Session expired for @%s — cookies invalid", account.username)
            return False

        if self._upload_avatar and not account.has_profile_pic:
            await self._upload_profile_pic(page, account, account_id)

        await self._browse_feed(page, account_id, duration_secs=random.randint(180, 360))
        await self._shot(page, account_id, "feed_browse_done")

        likes = await self._like_posts(page, account_id, count=random.randint(1, 2))
        logger.info("Liked %d posts for @%s", likes, account.username)
        await self._shot(page, account_id, "likes_done")

        handles = await self._pick_intel_handles(count=random.randint(2, 3))
        if handles:
            followed = await self._follow_accounts(page, account_id, handles)
            logger.info("Followed %d accounts for @%s", followed, account.username)
        await self._shot(page, account_id, "follows_done")

        return True

    def _sudo_read(self, path: Path) -> str:
        """Read a root-owned file via sudo cat."""
        result = subprocess.run(["sudo", "cat", str(path)], capture_output=True, text=True, check=True)
        return result.stdout

    def _sudo_list_confs(self) -> list[Path]:
        """List *.conf files in the countries dir via sudo ls."""
        result = subprocess.run(["sudo", "ls", str(self._countries_dir)], capture_output=True, text=True, check=True)
        return [
            self._countries_dir / name.strip() for name in result.stdout.splitlines() if name.strip().endswith(".conf")
        ]

    def _current_vpn_slug(self) -> str | None:
        """Identify current VPN slug by matching wg0.conf endpoint against country configs."""
        import re

        try:
            text = self._sudo_read(self._wg_conf)
            m = re.search(r"Endpoint\s*=\s*(\S+)", text)
            if not m:
                return None
            current_ep = m.group(1).strip()
            for conf in self._sudo_list_confs():
                conf_text = self._sudo_read(conf)
                ep_m = re.search(r"Endpoint\s*=\s*(.+)", conf_text)
                if ep_m and ep_m.group(1).strip() == current_ep:
                    return conf.stem
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
        return None

    def _resolve_vpn_slug(self, country: str) -> str:
        """Pick a random WireGuard slug for the given country code."""
        pool: dict[str, list[str]] = {}
        for conf in sorted(self._sudo_list_confs()):
            cc = conf.stem.split("-")[0].upper()
            pool.setdefault(cc, []).append(conf.stem)
        cc = country.upper()
        if cc not in pool:
            raise ValueError(f"No WireGuard config available for country '{country}'")
        return random.choice(pool[cc])

    async def _switch_vpn(self, slug: str) -> None:
        """Rewrite wg0.conf peer block via sudo tee and restart wg0."""
        import re

        conf = self._countries_dir / f"{slug}.conf"
        conf_text = self._sudo_read(conf)

        pub_m = re.search(r"PublicKey\s*=\s*(.+)", conf_text)
        ep_m = re.search(r"Endpoint\s*=\s*(.+)", conf_text)
        if not pub_m or not ep_m:
            raise ValueError(f"Cannot parse [Peer] block from {conf}")
        pubkey = pub_m.group(1).strip()
        endpoint = ep_m.group(1).strip()

        wg_text = self._sudo_read(self._wg_conf)
        idx = wg_text.find("[Peer]")
        interface_block = wg_text[:idx].rstrip() if idx != -1 else wg_text.rstrip()

        new_conf = f"{interface_block}\n\n[Peer]\nPublicKey = {pubkey}\nAllowedIPs = 0.0.0.0/0\nEndpoint = {endpoint}\n"
        proc = await asyncio.to_thread(
            subprocess.run,
            ["sudo", "tee", str(self._wg_conf)],
            input=new_conf,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to write wg0.conf: {proc.stderr}")

        await asyncio.to_thread(subprocess.run, ["sudo", "wg-quick", "down", "wg0"], check=True, capture_output=True)
        await asyncio.sleep(1)
        await asyncio.to_thread(subprocess.run, ["sudo", "wg-quick", "up", "wg0"], check=True, capture_output=True)
        await asyncio.sleep(2)
        logger.info("VPN → %s (%s)", slug, endpoint)

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

        # Instagram sometimes serves the logged-out marketing page at / without
        # redirecting to /accounts/login/. Confirm login by checking for feed
        # elements that only appear when authenticated.
        try:
            logged_in = await page.locator('svg[aria-label="Home"]').first.is_visible(timeout=5_000)
            if not logged_in:
                # Also accept the nav search icon as a logged-in indicator
                logged_in = await page.locator('svg[aria-label="Search"]').first.is_visible(timeout=3_000)
        except Exception:  # noqa: BLE001
            logged_in = False

        if not logged_in:
            logger.warning("Instagram served marketing page for @%s — treating as logged out", username)
            return False, False

        logger.info("Session active for @%s", username)
        return True, False

    async def _browse_feed(self, page, account_id: str, duration_secs: int) -> None:
        """Scroll the Instagram home feed for ~duration_secs seconds."""
        await page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
        await _pause(2.0, 4.0)
        await _dismiss_overlays(page)
        await self._shot(page, account_id, "feed_start")

        deadline = time.time() + duration_secs
        while time.time() < deadline:
            await page.mouse.wheel(0, random.randint(300, 700))
            pause = random.uniform(2.0, 6.0)
            await asyncio.sleep(min(pause, max(0.1, deadline - time.time())))

    async def _like_posts(self, page, account_id: str, count: int = 2) -> int:
        """Like `count` posts visible in the feed."""
        liked = 0
        await page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
        await _pause(2.0, 4.0)
        await _dismiss_overlays(page)

        for _ in range(random.randint(3, 5)):
            await page.mouse.wheel(0, random.randint(400, 700))
            await _pause(1.5, 3.0)

        await self._shot(page, account_id, "like_candidates")
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

    async def _follow_accounts(self, page, account_id: str, handles: list[str]) -> int:
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
                await self._shot(page, account_id, f"follow_{handle}")

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

    async def _upload_profile_pic(self, page, account: InstagramAccount, account_id: str) -> bool:
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
            # Navigate to own profile and click the avatar — current IG UI exposes
            # the file input via the profile picture button, not /accounts/edit/.
            await page.goto(
                f"https://www.instagram.com/{account.username}/",
                wait_until="domcontentloaded",
                timeout=20_000,
            )
            await _pause(2.0, 3.5)
            await _dismiss_overlays(page)
            await self._shot(page, account_id, "avatar_profile_page")

            # Click the profile picture to open the upload dialog.
            # Fresh accounts have a grey placeholder with no meaningful alt text,
            # so we try broad structural selectors before falling back to alt-text ones.
            clicked = False
            for sel in [
                f'img[alt="{account.username}\'s profile picture"]',
                'img[alt*="profile picture"]',
                'button:has(img[alt*="profile picture"])',
                'span:has(img[alt*="profile picture"])',
                # Desktop IG wraps the avatar in a <button> with an SVG or <img> inside
                "header button img",
                "header button svg",
                "header section button",
                "header button",
            ]:
                try:
                    el = page.locator(sel).first
                    if await el.is_visible(timeout=3_000):
                        await el.click()
                        await _pause(1.0, 2.0)
                        clicked = True
                        break
                except Exception:  # noqa: BLE001
                    pass

            await self._shot(page, account_id, "avatar_after_click")

            if not clicked:
                logger.warning("Could not click profile picture for @%s — skipping avatar upload", account.username)
                return False

            # "Upload photo" option appears in the dialog
            for label in ["Upload photo", "Upload Photo", "Change profile photo"]:
                try:
                    btn = page.get_by_text(label, exact=True).first
                    if await btn.is_visible(timeout=3_000):
                        await btn.click()
                        await _pause(0.8, 1.5)
                        break
                except Exception:  # noqa: BLE001
                    pass

            await self._shot(page, account_id, "avatar_upload_dialog")

            # File input should now be visible; use short timeout so failure is fast
            file_input = page.locator('input[type="file"]').first
            await file_input.set_input_files(str(avatar_path), timeout=8_000)
            await _pause(2.5, 4.5)
            await self._shot(page, account_id, "avatar_after_file_set")

            for label in ["Done", "Apply", "Save", "Next", "Confirm"]:
                try:
                    btn = page.get_by_role("button", name=label, exact=True).first
                    if await btn.is_visible(timeout=3_000):
                        await btn.click()
                        await _pause(1.0, 2.0)
                        break
                except Exception:  # noqa: BLE001
                    pass

            await self._shot(page, account_id, "avatar_confirmed")
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
