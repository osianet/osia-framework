"""
Proton Mail inbox reader via Camoufox browser automation.

Logs in to a stored Proton account, polls the inbox for a matching email,
opens it (triggering client-side decryption), and extracts a 6-digit
verification code from the body.

Primary use: retrieve Instagram signup OTPs sent to Proton email addresses.
"""

import asyncio
import logging
import re
from pathlib import Path

from camoufox.async_api import AsyncCamoufox

from src.agents.proton_account_manager import ProtonAccount, ProtonAccountManager

logger = logging.getLogger("osia.proton_reader")

_INBOX_URL = "https://mail.proton.me/inbox"
_LOGIN_URL = "https://account.proton.me/login"

# Matches any standalone 6-digit sequence (Instagram, Twitter, etc.)
_CODE_RE = re.compile(r"\b(\d{6})\b")


class ProtonMailReader:
    """
    Read emails from a Proton Mail inbox via browser automation.

    Proton decrypts message content client-side in JavaScript so we must use
    a real browser — there is no plaintext API to call directly.
    """

    def __init__(self, manager: ProtonAccountManager, headless: bool = True):
        self._manager = manager
        self._headless = headless

    async def get_verification_code(
        self,
        account: ProtonAccount,
        *,
        sender_contains: str | None = None,
        subject_contains: str | None = None,
        timeout: int = 120,
        poll_interval: int = 5,
    ) -> str | None:
        """
        Log in to Proton Mail, poll the inbox until a matching email arrives,
        and return the first 6-digit code found in the body.

        Args:
            account:          The Proton account to check.
            sender_contains:  Case-insensitive substring the sender must contain (e.g. "instagram").
            subject_contains: Case-insensitive substring the subject must contain.
            timeout:          Total seconds to wait before giving up.
            poll_interval:    Seconds between inbox refresh attempts.

        Returns:
            The 6-digit code string, or None if timeout was reached.
        """
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
                if not await self._login(page, account, debug_dir):
                    raise RuntimeError(f"Could not log in to {account.email}")

                deadline = asyncio.get_event_loop().time() + timeout
                attempt = 0
                while asyncio.get_event_loop().time() < deadline:
                    attempt += 1
                    remaining = int(deadline - asyncio.get_event_loop().time())
                    logger.info(
                        "Inbox check #%d for %s (%ds remaining)…",
                        attempt,
                        account.email,
                        remaining,
                    )

                    await self._refresh_inbox(page)
                    code = await self._scan_inbox(page, sender_contains, subject_contains, debug_dir, account.username)
                    if code:
                        logger.info("Verification code retrieved for %s: %s", account.email, code)
                        return code

                    await asyncio.sleep(poll_interval)

                logger.warning("Timed out waiting for verification email for %s", account.email)
                return None
            finally:
                await context.close()

    # ------------------------------------------------------------------ internals

    async def _login(self, page, account: ProtonAccount, debug_dir: Path) -> bool:
        """Navigate to Proton login and authenticate. Returns True when inbox is reached."""
        await page.goto(_LOGIN_URL, wait_until="domcontentloaded", timeout=30_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
        await _pause(1.5, 2.5)

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

        for name in ["Sign in", "Log in", "Continue"]:
            try:
                btn = page.get_by_role("button", name=name, exact=True).first
                if await btn.is_visible(timeout=2_000):
                    await btn.click()
                    break
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

        for _ in range(30):
            await asyncio.sleep(1)
            url = page.url
            if any(kw in url for kw in ("mail.proton.me", "/inbox")):
                logger.info("Logged in to %s", account.email)
                await _pause(2.0, 3.0)
                return True
            if "account.proton.me" in url and "login" not in url and "signup" not in url:
                # Landed on an account page — navigate to inbox directly
                await page.goto(_INBOX_URL, wait_until="domcontentloaded", timeout=30_000)
                await _pause(2.0, 3.0)
                return True

        await _save_shot(page, debug_dir, account.username, "login_fail")
        logger.error("Login failed for %s — still on: %s", account.email, page.url)
        return False

    async def _refresh_inbox(self, page) -> None:
        """Reload the inbox to surface new messages."""
        try:
            inbox_link = page.get_by_role("link", name="Inbox").first
            if await inbox_link.is_visible(timeout=2_000):
                await inbox_link.click()
                await _pause(1.5, 2.5)
                return
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

        if "mail.proton.me" in page.url:
            try:
                await page.reload(wait_until="domcontentloaded", timeout=20_000)
                await _pause(1.5, 2.5)
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

    async def _scan_inbox(
        self,
        page,
        sender_contains: str | None,
        subject_contains: str | None,
        debug_dir: Path,
        username: str,
    ) -> str | None:
        """
        Scan the visible inbox rows. For each matching row click to open the
        email and extract a 6-digit code from the decrypted body.
        """
        await _pause(1.0, 1.5)

        # Proton Mail renders email rows with varying test IDs across versions.
        row_selectors = [
            '[data-testid="message-item:body"]',
            '[data-testid="message-row"]',
            '[data-testid^="message-item"]',
            '[class*="message-item"]',
            '[role="row"]',
        ]

        rows = []
        for sel in row_selectors:
            try:
                found = await page.locator(sel).all()
                if found:
                    rows = found
                    break
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

        if not rows:
            logger.debug("No email rows found in inbox view")
            return None

        logger.debug("Found %d email rows", len(rows))

        for row in rows[:10]:
            try:
                row_text = (await row.inner_text()).lower()

                if sender_contains and sender_contains.lower() not in row_text:
                    continue
                if subject_contains and subject_contains.lower() not in row_text:
                    continue

                logger.info("Matching email row found — opening…")
                await row.click()
                await _pause(2.5, 4.0)

                await _save_shot(page, debug_dir, username, "email_open")

                code = await self._extract_code_from_page(page)
                if code:
                    return code

                # Go back to inbox for next row
                try:
                    back = page.get_by_role("button", name="Back").first
                    if await back.is_visible(timeout=1_500):
                        await back.click()
                        await _pause(1.0, 1.5)
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)

            except Exception as exc:
                logger.debug("Suppressed while processing row: %s", exc)

        return None

    async def _extract_code_from_page(self, page) -> str | None:
        """
        Extract a 6-digit verification code from the currently open email.
        Proton renders the decrypted email body in a sandboxed iframe.
        """
        # Try sandboxed content iframe first (Proton's standard rendering)
        for frame in page.frames:
            if frame is page.main_frame:
                continue
            try:
                body_text = await frame.inner_text("body", timeout=5_000)
                if body_text and len(body_text) > 5:
                    code = _first_code(body_text)
                    if code:
                        logger.debug("Code found in email iframe")
                        return code
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

        # Try known message-body selectors on the main page
        for sel in [
            '[data-testid="message-content"]',
            '[data-testid="message-body"]',
            ".message-content",
            ".proton-message-body",
            "[class*='message-body']",
        ]:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=2_000):
                    text = await el.inner_text()
                    code = _first_code(text)
                    if code:
                        return code
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

        # Last resort: full page text (noisier but catches edge cases)
        try:
            full_text = await page.inner_text("body")
            return _first_code(full_text)
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

        return None


# ------------------------------------------------------------------ helpers


def _first_code(text: str) -> str | None:
    """Return the first 6-digit code found in text, or None."""
    m = _CODE_RE.search(text)
    return m.group(1) if m else None


async def _save_shot(page, debug_dir: Path, username: str, label: str) -> None:
    try:
        shot = debug_dir / f"{username}_{label}.png"
        await page.screenshot(path=str(shot), full_page=False)
        logger.info("Screenshot: %s", shot)
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)


async def _pause(lo: float, hi: float) -> None:
    import random

    await asyncio.sleep(random.uniform(lo, hi))
