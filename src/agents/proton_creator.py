"""
Proton Mail account creation via Camoufox browser automation.

Flow:
  1. Navigate to account.proton.me/signup
  2. Free plan is pre-selected — no interaction needed
  3. Fill username + password
  4. Click "Start using Proton Mail now"
  5. Handle Proton CAPTCHA (human in headed mode, automatic otherwise)
  6. Skip optional recovery steps
  7. Verify we landed on the inbox/welcome page
  8. Register account in pool as AVAILABLE
"""

import asyncio
import logging
import random
import secrets
import string
from pathlib import Path

from camoufox.async_api import AsyncCamoufox

from src.agents.proton_account_manager import ProtonAccount, ProtonAccountManager

logger = logging.getLogger("osia.proton_creator")

_SIGNUP_URL = "https://account.proton.me/signup"


class ProtonCreator:
    """Creates Proton Mail free accounts via browser automation."""

    def __init__(self, manager: ProtonAccountManager, headless: bool = True):
        self._manager = manager
        self._headless = headless

    async def create_new(self) -> ProtonAccount:
        """
        Full creation flow. Returns the registered ProtonAccount (state=AVAILABLE).
        Raises on failure.
        """
        username = _gen_username()
        password = _gen_password()

        logger.info("Creating Proton account: %s@proton.me", username)
        await self._run_browser_signup(username, password)

        account = await self._manager.register(username, password)
        logger.info("Proton account registered: %s (%s)", account.id[:8], account.email)
        return account

    async def test_login(self, account: ProtonAccount) -> bool:
        """
        Open a headed browser, log in with stored credentials, confirm inbox is reachable.
        Returns True on success.
        """
        debug_dir = self._manager._base_dir / "logs" / "proton_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        async with AsyncCamoufox(headless=False, geoip=True, locale="en-AU", os="windows") as browser:
            context = await browser.new_context(viewport={"width": 1280, "height": 900})
            page = await context.new_page()
            try:
                await page.goto("https://account.proton.me/login", wait_until="domcontentloaded", timeout=30_000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10_000)
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)
                await _pause(1.5, 2.5)

                # Fill email/username
                for sel in [
                    'input[id="username"]',
                    'input[name="username"]',
                    'input[type="email"]',
                    'input[autocomplete="username"]',
                ]:
                    try:
                        el = page.locator(sel).first
                        if await el.is_visible(timeout=3_000):
                            await el.click()
                            await el.fill(account.email)
                            break
                    except Exception as exc:
                        logger.debug("Suppressed: %s", exc)

                await _pause(0.5, 1.0)

                # Fill password
                for sel in ['input[id="password"]', 'input[name="password"]', 'input[type="password"]']:
                    try:
                        el = page.locator(sel).first
                        if await el.is_visible(timeout=3_000):
                            await el.click()
                            await el.fill(account.password)
                            break
                    except Exception as exc:
                        logger.debug("Suppressed: %s", exc)

                await _pause(0.5, 1.0)

                # Submit
                for name in ["Sign in", "Log in", "Continue"]:
                    try:
                        btn = page.get_by_role("button", name=name, exact=True).first
                        if await btn.is_visible(timeout=2_000):
                            await btn.click()
                            break
                    except Exception as exc:
                        logger.debug("Suppressed: %s", exc)

                # Wait up to 30s to land on inbox
                for _ in range(30):
                    await asyncio.sleep(1)
                    url = page.url
                    if any(kw in url for kw in ("mail.proton.me", "/inbox", "account.proton.me/mail")):
                        await _save_shot(page, debug_dir, account.username, "test_login_ok")
                        logger.info("Login confirmed for %s — URL: %s", account.email, url)
                        return True
                    if "account.proton.me" in url and "login" not in url and "signup" not in url:
                        await _save_shot(page, debug_dir, account.username, "test_login_ok")
                        logger.info("Login confirmed for %s — URL: %s", account.email, url)
                        return True

                await _save_shot(page, debug_dir, account.username, "test_login_fail")
                logger.error("Login did not reach inbox for %s — URL: %s", account.email, page.url)
                return False
            finally:
                await context.close()

    async def _run_browser_signup(self, username: str, password: str) -> None:
        debug_dir = self._manager._base_dir / "logs" / "proton_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        async with AsyncCamoufox(
            headless=self._headless,
            geoip=True,
            locale="en-AU",
            os="windows",
        ) as browser:
            context = await browser.new_context(viewport={"width": 1280, "height": 900})
            page = await context.new_page()

            try:
                await self._signup_flow(page, username, password, debug_dir)
                await _verify_logged_in(page, username, debug_dir)
            except Exception:
                try:
                    shot = debug_dir / f"{username}_failure.png"
                    await page.screenshot(path=str(shot), full_page=False)
                    logger.error("Failure screenshot: %s", shot)
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)
                raise
            finally:
                await context.close()

    async def _signup_flow(self, page, username: str, password: str, debug_dir: Path) -> None:
        logger.info("Navigating to Proton signup page…")
        await page.goto(_SIGNUP_URL, wait_until="domcontentloaded", timeout=45_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
        await _pause(2.0, 3.0)

        await _save_shot(page, debug_dir, username, "loaded")

        # The Free plan card should already be selected. Click it to be sure.
        await _ensure_free_plan(page)

        # --- Fill username ---
        # The username input is inside an iframe on Proton's signup page.
        # Proton embeds the form fields in a sandboxed iframe for security.
        await _fill_username(page, username)
        await _pause(0.5, 1.0)

        # --- Fill password + confirm ---
        await _fill_password(page, password)
        # Proton reveals a "Re-enter password" field after the first password is typed
        await _pause(0.8, 1.2)
        await _fill_confirm_password(page, password)

        await _save_shot(page, debug_dir, username, "filled")

        # --- Submit ---
        if not await _click_submit(page):
            raise RuntimeError("Cannot locate 'Start using Proton Mail now' button")

        await _pause(2.0, 3.0)
        await _save_shot(page, debug_dir, username, "post_submit")

        # --- CAPTCHA ---
        await self._handle_captcha(page, username, debug_dir)

        # --- Optional recovery steps (skip them all) ---
        await _skip_optional_steps(page)

        await _save_shot(page, debug_dir, username, "post_optional")

    async def _handle_captcha(self, page, username: str, debug_dir: Path) -> None:
        """
        Proton uses their own CAPTCHA (not Google reCAPTCHA).
        In headed mode: wait for the human to solve it (up to 10 minutes).
        In headless mode: attempt to click through; raise if it fails.
        """
        # Wait up to 15s for a CAPTCHA iframe or modal to appear
        captcha_present = False
        for sel in [
            'iframe[title*="captcha" i]',
            'iframe[src*="captcha" i]',
            '[data-testid="captcha"]',
            'iframe[title*="CAPTCHA"]',
        ]:
            try:
                if await page.locator(sel).first.is_visible(timeout=3_000):
                    captcha_present = True
                    break
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

        # Also check if we're still on the signup page (not yet navigated to inbox)
        if not captcha_present:
            url = page.url
            still_on_signup = "signup" in url or "account.proton.me" in url
            if not still_on_signup:
                logger.info("No CAPTCHA detected — already past signup")
                return

        if not self._headless:
            print(
                "\n"
                ">>> CAPTCHA: Please solve the Proton CAPTCHA in the browser window. <<<\n"
                ">>> The script will continue automatically once you're past it.      <<<\n",
                flush=True,
            )
            logger.info("Headed mode — waiting for human to solve CAPTCHA (up to 10 min)…")
            for _ in range(120):  # 120 × 5s = 10 minutes
                await asyncio.sleep(5)
                url = page.url
                # We've moved past signup when the URL is mail.proton.me or shows a welcome/inbox page
                if any(kw in url for kw in ("mail.proton.me", "proton.me/mail", "/inbox", "account.proton.me/mail")):
                    logger.info("CAPTCHA solved — URL: %s", url)
                    return
                # Also accept if we're on account.proton.me but not /signup anymore
                if "account.proton.me" in url and "signup" not in url and "captcha" not in url:
                    logger.info("CAPTCHA solved — past signup page: %s", url)
                    return
            raise RuntimeError("CAPTCHA not solved within 10 minutes — timed out")

        # Headless: try clicking the CAPTCHA checkbox (Proton's widget is similar to hCaptcha)
        logger.info("Headless mode — attempting CAPTCHA checkbox click…")
        for frame in page.frames:
            url = frame.url or ""
            if "captcha" not in url.lower():
                continue
            for sel in ("#checkbox", ".checkbox", '[role="checkbox"]', "input[type='checkbox']"):
                try:
                    el = frame.locator(sel).first
                    if await el.is_visible(timeout=2_000):
                        await el.click()
                        logger.info("Clicked CAPTCHA checkbox in frame %s", url[:60])
                        await _pause(3.0, 5.0)
                        break
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)

        # Check if we moved past signup
        await _pause(2.0, 3.0)
        url = page.url
        if "signup" in url and "captcha" in url.lower():
            raise RuntimeError("CAPTCHA still blocking after headless attempt — run with --headed to solve manually")


# ------------------------------------------------------------------ helpers


async def _ensure_free_plan(page) -> None:
    """Click the Free plan card if it's not already selected."""
    try:
        free_card = page.get_by_text("Free", exact=True).first
        if await free_card.is_visible(timeout=3_000):
            await free_card.click()
            await _pause(0.5, 1.0)
            logger.debug("Free plan selected")
    except Exception as exc:
        logger.debug("Free plan click suppressed: %s", exc)


async def _fill_username(page, username: str) -> None:
    """
    Proton embeds the username input in a sandboxed iframe.
    Try the iframe first, then fall back to direct page selectors.
    """
    # Attempt 1: find username input inside any iframe
    for frame in page.frames:
        for sel in ['input[id*="username" i]', 'input[name*="username" i]', 'input[autocomplete="username"]']:
            try:
                el = frame.locator(sel).first
                if await el.is_visible(timeout=2_000):
                    await el.click()
                    await el.fill("")
                    await _type(el, username)
                    logger.debug("Filled username in frame")
                    return
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

    # Attempt 2: direct page selector
    for sel in [
        'input[id*="username" i]',
        'input[name*="username" i]',
        'input[placeholder*="username" i]',
        'input[autocomplete="username"]',
    ]:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=2_000):
                await el.click()
                await el.fill("")
                await _type(el, username)
                logger.debug("Filled username on page")
                return
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

    raise RuntimeError("Cannot locate Proton username input — check loaded screenshot")


async def _fill_password(page, password: str) -> None:
    """Fill the first password field (may be inside an iframe)."""
    # Proton's iframe may have two password inputs; target only the first one.
    for frame in page.frames:
        els = await frame.locator('input[type="password"]').all()
        if els:
            el = els[0]
            try:
                if await el.is_visible(timeout=2_000):
                    await el.click()
                    await el.fill("")
                    await _type(el, password)
                    logger.debug("Filled password in frame")
                    return
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

    for sel in ['input[type="password"]', 'input[id*="password" i]', 'input[name*="password" i]']:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=2_000):
                await el.click()
                await el.fill("")
                await _type(el, password)
                logger.debug("Filled password on page")
                return
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

    raise RuntimeError("Cannot locate Proton password input")


async def _fill_confirm_password(page, password: str) -> None:
    """
    Fill the 'Re-enter password' field that Proton reveals after the first password is typed.
    It's the second password-type input in the same iframe.
    """
    # Try iframe first — pick the second password input
    for frame in page.frames:
        els = await frame.locator('input[type="password"]').all()
        if len(els) >= 2:
            el = els[1]
            try:
                if await el.is_visible(timeout=3_000):
                    await el.click()
                    await el.fill("")
                    await _type(el, password)
                    logger.debug("Filled confirm password in frame")
                    return
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

    # Fallback: look by label/placeholder text on the main page
    for sel in [
        'input[id*="confirm" i]',
        'input[name*="confirm" i]',
        'input[placeholder*="confirm" i]',
        'input[placeholder*="re-enter" i]',
        'input[placeholder*="repeat" i]',
    ]:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=2_000):
                await el.click()
                await el.fill("")
                await _type(el, password)
                logger.debug("Filled confirm password on page")
                return
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

    # If we couldn't find it, log a warning but don't raise — it may not have appeared yet
    # or may be optional on some Proton flows.
    logger.warning("Could not locate confirm-password field — continuing without it")


async def _click_submit(page) -> bool:
    """Click the signup submit button."""
    for name in ["Start using Proton Mail now", "Create account", "Get Proton for free", "Continue"]:
        try:
            btn = page.get_by_role("button", name=name, exact=True).first
            if await btn.is_visible(timeout=3_000):
                await btn.click()
                return True
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
    # Fallback: any submit button
    try:
        btn = page.locator('button[type="submit"]').first
        if await btn.is_visible(timeout=2_000):
            await btn.click()
            return True
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)
    return False


async def _skip_optional_steps(page) -> None:
    """
    Proton shows optional recovery email/phone steps after signup.
    Skip all of them.
    """
    skip_labels = [
        "Maybe later",
        "Skip",
        "No thanks",
        "Not now",
        "Not Now",
        "Continue",
        "Start using Proton Mail",
    ]
    for _ in range(8):
        clicked = False
        for label in skip_labels:
            try:
                btn = page.get_by_role("button", name=label, exact=True).first
                if await btn.is_visible(timeout=1_500):
                    await btn.click()
                    await _pause(1.0, 2.0)
                    clicked = True
                    break
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
        if not clicked:
            break


async def _verify_logged_in(page, username: str, debug_dir: Path) -> None:
    """
    Verify we successfully landed on the Proton Mail inbox or welcome page.
    Raises if not reached within 15 seconds.
    """
    await _pause(2.0, 3.0)
    url = page.url
    logger.info("Post-signup URL: %s", url)

    await _save_shot(page, debug_dir, username, "post_signup")

    # Success indicators
    success_urls = ("mail.proton.me", "/inbox", "account.proton.me/mail")
    if any(kw in url for kw in success_urls):
        logger.info("Proton signup confirmed — on inbox/welcome page")
        return

    # Wait up to 10s for navigation to complete
    for _ in range(10):
        await asyncio.sleep(1)
        url = page.url
        if any(kw in url for kw in success_urls):
            logger.info("Proton signup confirmed — URL: %s", url)
            return
        # Also accept if past signup (no longer on /signup)
        if "account.proton.me" in url and "signup" not in url and "captcha" not in url:
            logger.info("Proton signup confirmed — past signup: %s", url)
            return

    # Check for inbox elements as fallback
    for sel in ['[data-testid="sidebar"]', 'a[href*="inbox"]', '[aria-label="Inbox"]']:
        try:
            if await page.locator(sel).first.is_visible(timeout=3_000):
                logger.info("Proton signup confirmed — inbox element visible")
                return
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

    raise RuntimeError(f"Proton signup did not complete — still on: {url}. Check logs/proton_debug/ for screenshots.")


async def _save_shot(page, debug_dir: Path, username: str, label: str) -> None:
    try:
        shot = debug_dir / f"{username}_{label}.png"
        await page.screenshot(path=str(shot), full_page=False)
        logger.info("Screenshot: %s", shot)
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)


async def _type(field, text: str) -> None:
    for ch in text:
        await field.type(ch, delay=random.randint(40, 120))


async def _pause(lo: float, hi: float) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


# ------------------------------------------------------------------ generators

_ADJECTIVES = [
    "dark",
    "swift",
    "quiet",
    "still",
    "grey",
    "cold",
    "deep",
    "clear",
    "sharp",
    "low",
    "bright",
    "calm",
    "dry",
    "flat",
    "hard",
    "lean",
    "mild",
    "narrow",
    "open",
    "plain",
    "bold",
    "free",
    "wise",
    "pure",
]
_NOUNS = [
    "river",
    "stone",
    "creek",
    "ridge",
    "field",
    "peak",
    "moss",
    "vale",
    "ford",
    "glen",
    "brook",
    "cliff",
    "cove",
    "dale",
    "dune",
    "grove",
    "heath",
    "knoll",
    "marsh",
    "moor",
    "lake",
    "pine",
    "frost",
    "cloud",
]


def _gen_username() -> str:
    return f"{random.choice(_ADJECTIVES)}.{random.choice(_NOUNS)}.{random.randint(10, 9999)}"


def _gen_password() -> str:
    chars = string.ascii_letters + string.digits + "!@#$"
    return "".join(secrets.choice(chars) for _ in range(16))
