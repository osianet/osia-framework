"""
Instagram account creation via Camoufox browser automation.

Flow:
  1. Purchase SMSPool phone number for target country
  2. Switch WireGuard VPN to a country-matched exit (requires root)
  3. Launch Camoufox with matching locale and geoip spoofing
  4. Navigate Instagram phone-number signup form
  5. Poll SMSPool API for 6-digit OTP and enter it
  6. Export session cookies to Netscape format for yt-dlp
  7. Register account in pool as WARMING
  8. Restore original VPN endpoint

Requires sudo/root for wg-quick VPN switching.
"""

import asyncio
import logging
import os
import random
import re
import secrets
import string
import subprocess
from pathlib import Path

from camoufox.async_api import AsyncCamoufox

from src.agents.instagram_account_manager import InstagramAccount, InstagramAccountManager

logger = logging.getLogger("osia.ig_creator")

_SIGNUP_URL = "https://www.instagram.com/accounts/emailsignup/"

# Country → (locale, timezone)
_COUNTRY_PROFILE: dict[str, tuple[str, str]] = {
    "AU": ("en-AU", "Australia/Sydney"),
    "US": ("en-US", "America/New_York"),
    "UK": ("en-GB", "Europe/London"),
    "GB": ("en-GB", "Europe/London"),
    "CA": ("en-CA", "America/Toronto"),
    "SG": ("en-SG", "Asia/Singapore"),
    "NZ": ("en-NZ", "Pacific/Auckland"),
}

_BIRTHDAY_YEAR_RANGE = (1987, 2000)

_WG_CONF = Path("/etc/wireguard/wg0.conf")
_COUNTRIES_DIR = Path("/etc/wireguard/countries")


class InstagramCreator:
    """
    Orchestrates the full account creation pipeline:
    number purchase → VPN switch → browser signup → OTP → cookie export → pool registration.
    """

    def __init__(
        self,
        manager: InstagramAccountManager,
        wg_conf: Path = _WG_CONF,
        countries_dir: Path = _COUNTRIES_DIR,
        headless: bool = True,
    ):
        self._manager = manager
        self._wg_conf = wg_conf
        self._countries_dir = countries_dir
        self._headless = headless

    async def create_new(
        self, country: str = "AU", skip_vpn: bool = False, sms_country: str | None = None
    ) -> InstagramAccount:
        """
        Full automated creation flow. Returns the registered account (state=WARMING).
        Raises on any failure; cleans up partial state and cancels SMSPool order.

        skip_vpn=True: omit VPN switching entirely — use when the caller (e.g. ig_create_account.sh)
        handles VPN switching externally with appropriate privileges.

        sms_country: SMSPool country code to use for purchasing the OTP number.
        Defaults to `country`. Override when the target country only has virtual numbers
        (e.g. AU only has AU_V) — use a country with real SIM numbers such as US or GB.
        The phone number is only used for signup OTP, not for account identity.
        """
        if not self._manager.smspool:
            raise RuntimeError("SMSPool not configured — set SMSPOOL_API_KEY in .env")

        sms_country = sms_country or country
        order_data: dict = {}
        account: InstagramAccount | None = None
        original_slug: str | None = None
        target_slug: str | None = None

        try:
            # --- Purchase phone number ---
            logger.info("Purchasing SMSPool number for sms_country=%s (account country=%s)", sms_country, country)
            order_data = await self._manager.smspool.purchase_number(sms_country)
            # `number` = full number with country code (e.g. 61468245956)
            # `phonenumber` = subscriber digits only (e.g. 468245956) — do not use alone
            phone_raw = str(order_data.get("number") or order_data.get("phonenumber", ""))
            phone = phone_raw if phone_raw.startswith("+") else f"+{phone_raw}"
            order_id = str(order_data.get("order_id") or order_data.get("orderid", ""))
            if not phone or not order_id:
                raise RuntimeError(f"Unexpected SMSPool response: {order_data}")
            logger.info("SMSPool order purchased: %s sms_country=%s", order_id, sms_country)

            # --- Register bare account record (CREATED state) ---
            account = await self._manager.register(
                username=_gen_username(),
                password=_gen_password(),
                email="",
                phone=phone,
                phone_country=sms_country,
                vpn_country=country,
                smspool_order_id=order_id,
            )

            if not skip_vpn:
                original_slug = self._current_vpn_slug()
                target_slug = self._resolve_vpn_slug(country)
                if target_slug != original_slug:
                    logger.info("Switching VPN to %s (%s)", country, target_slug)
                    await self._switch_vpn(target_slug)
                else:
                    logger.info("VPN already on %s — no switch needed", target_slug)
                    original_slug = None  # no switch happened; nothing to restore

            # --- Browser signup ---
            locale, _tz = _COUNTRY_PROFILE.get(country.upper(), ("en-US", "America/New_York"))
            await self._run_browser_signup(account, locale)

            # Verify cookies contain sessionid — a missing sessionid means the browser
            # session was captured before authentication completed (common if signup was
            # interrupted by a CAPTCHA or checkpoint that was not resolved).
            content = await self._manager.get_cookie_content(account.id)
            if not content:
                raise RuntimeError("Cookie content missing after signup — creation incomplete")
            cookie_names = [
                line.split("\t")[5]
                for line in content.splitlines()
                if line and not line.startswith("#") and len(line.split("\t")) >= 7
            ]
            if "sessionid" not in cookie_names:
                raise RuntimeError(
                    f"sessionid missing from signup cookies (got: {cookie_names}) — "
                    "signup may have stalled at a CAPTCHA or checkpoint"
                )

            # --- Promote to WARMING ---
            await self._manager.start_warming(account.id)
            logger.info("Account %s (%s) → WARMING", account.id, account.username)
            return account

        except Exception as exc:
            logger.error("Account creation failed: %s", exc)
            if account:
                await self._manager.retire(account.id)
            if order_id := (str(order_data.get("order_id") or order_data.get("orderid", ""))):
                try:
                    await self._manager.smspool.cancel_order(order_id)
                    logger.info("Cancelled SMSPool order %s", order_id)
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)
            raise

        finally:
            if not skip_vpn and original_slug:
                logger.info("Restoring VPN to %s", original_slug)
                try:
                    await self._switch_vpn(original_slug)
                except Exception as e:
                    logger.warning("VPN restore failed: %s", e)

    # ---------------------------------------------------------------- browser

    async def _run_browser_signup(self, account: InstagramAccount, locale: str) -> None:
        import tempfile

        # Use a temp file as staging; cookies are stored in Redis, not on disk permanently.
        cookie_path = Path(tempfile.gettempdir()) / f"osia_ig_signup_{account.id}.txt"
        debug_dir = self._manager._base_dir / "logs" / "ig_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        async with AsyncCamoufox(
            headless=self._headless,
            geoip=True,
            locale=locale,
            os="windows",
        ) as browser:
            context = await browser.new_context(viewport={"width": 1280, "height": 900})
            page = await context.new_page()

            # Intercept Google reCAPTCHA requests to capture the sitekey before it's needed.
            # The sitekey appears in the ?k= parameter of the reCAPTCHA API URL.
            captured_sitekey: list[str] = []

            def _on_request(request) -> None:
                url = request.url
                if "google.com/recaptcha" in url or "recaptcha.net/recaptcha" in url:
                    m = re.search(r"[?&]k=([^&]+)", url)
                    if m and not captured_sitekey:
                        captured_sitekey.append(m.group(1))
                        logger.info("reCAPTCHA sitekey captured: %s", captured_sitekey[0][:20])

            page.on("request", _on_request)

            try:
                await self._signup_flow(page, account, captured_sitekey)

                # Verify we're actually logged in before exporting cookies.
                # Navigate to the home page and look for the authenticated feed icon.
                # This catches cases where OTP was entered but Instagram rejected it,
                # showed a challenge, or the session was never created.
                await _verify_logged_in(page, account.id, debug_dir)

                await _export_cookies_netscape(page, cookie_path)
                content = cookie_path.read_text(encoding="utf-8")
                await self._manager.set_cookie_content(account.id, content)
                logger.info("Cookies stored in Redis for account %s", account.id)
            except Exception:
                try:
                    shot = debug_dir / f"{account.id}_failure.png"
                    html = debug_dir / f"{account.id}_failure.html"
                    await page.screenshot(path=str(shot), full_page=True)
                    raw_html = await page.content()
                    # Mask password values before writing — logs may be readable by other users
                    safe_html = re.sub(
                        r'(<input[^>]+type=["\']password["\'][^>]+value=["\'])[^"\']*(["\'])',
                        r"\1***\2",
                        raw_html,
                        flags=re.IGNORECASE,
                    )
                    html.write_text(safe_html)
                    logger.error("Debug screenshot: %s", shot)
                    logger.error("Debug HTML: %s", html)
                except Exception as dump_err:
                    logger.warning("Could not save debug dump: %s", dump_err)
                raise
            finally:
                await context.close()
                cookie_path.unlink(missing_ok=True)

    async def _signup_flow(self, page, account: InstagramAccount, captured_sitekey: list[str] | None = None) -> None:
        logger.info("Opening signup page...")
        await page.goto(_SIGNUP_URL, wait_until="load", timeout=45_000)
        # Wait for React to mount — Instagram is a SPA; DOM-ready fires before JS renders.
        try:
            await page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)  # networkidle can time out on slow pages — proceed anyway
        await _pause(1.5, 3.0)
        logger.info("Page loaded — URL: %s  title: %s", page.url, await page.title())

        # Early diagnostic dump — captures what the page actually looks like after load.
        # Saved before any interaction so we have a clean baseline regardless of later failure.
        debug_dir = self._manager._base_dir / "logs" / "ig_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            early_shot = debug_dir / f"{account.id}_loaded.png"
            await page.screenshot(path=str(early_shot), full_page=True)
            logger.info("Early screenshot: %s", early_shot)
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

        await _dismiss_overlays(page)

        # Instagram uses <label for="..."> associations with no name/placeholder on inputs.
        # get_by_label() is the only reliable selector strategy.
        # Wait up to 10s for any signup-form label to appear (React may still be mounting).
        await _wait_for_any_label(page, ["Mobile number or email", "Phone number"], timeout=10_000)

        # Mobile number or email
        phone_field = await _by_label(page, ["Mobile number or email", "Phone number, username, or email"])
        if not phone_field:
            raise RuntimeError("Cannot locate phone/email input on Instagram signup form")
        await phone_field.click()
        await _type(phone_field, account.phone)
        await _pause(0.4, 0.9)

        # Full name
        name_field = await _by_label(page, ["Full name"])
        if name_field:
            await name_field.click()
            await _type(name_field, _gen_full_name())
            await _pause(0.3, 0.7)

        # Username — Instagram may auto-suggest one; clear it first
        user_field = await _by_label(page, ["Username"])
        if user_field:
            await _pause(0.5, 1.0)
            await user_field.click(click_count=3)
            await _type(user_field, account.username)
            await _pause(0.3, 0.7)

        # Password
        pass_field = await _by_label(page, ["Password"])
        if not pass_field:
            raise RuntimeError("Cannot locate password field on signup form")
        await pass_field.click()
        await _type(pass_field, account.password)
        await _pause(0.5, 1.0)

        # Birthday — part of the same signup form, must be filled before Submit
        await _fill_birthday(page)

        # Submit signup form — Instagram uses div[role="button"] not button[type="submit"]
        if not await _click_action_button(page, ["Submit", "Next", "Sign up"]):
            raise RuntimeError("Cannot locate signup submit button")
        # Wait for the SPA to finish its post-submit API call and render the next step.
        # networkidle alone isn't sufficient — the spinner can be client-side only, and
        # Instagram's CAPTCHA dialog renders a few seconds after the spinner appears.
        try:
            await page.wait_for_load_state("networkidle", timeout=20_000)
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
        await _pause(3.0, 5.0)
        await _check_for_block(page)

        # Instagram shows a "Help us confirm it's you" dialog with a reCAPTCHA before the OTP.
        # Solve it if TWOCAPTCHA_API_KEY is set; otherwise raise a clear error.
        await self._handle_recaptcha_dialog(page, captured_sitekey or [])

        # Instagram may ask HOW to receive the code (SMS / email choice) before showing the input.
        await _handle_delivery_choice(page)

        # OTP
        await self._handle_otp(page, account)

        # Optional post-signup steps
        await _skip_optional_steps(page)

    async def _handle_recaptcha_dialog(self, page, captured_sitekey: list[str]) -> None:
        """
        Handle the 'Help us confirm it's you' dialog that Instagram shows after signup submit.

        Strategy:
          1. If no dialog is visible, return immediately.
          2. Click 'Next' — Instagram's reCAPTCHA is often invisible/auto-solved, so Next
             alone is sufficient in many sessions. Wait for the dialog to disappear.
          3. If the dialog is STILL visible after Next, fall back to 2captcha solving:
             extract the sitekey (from captured network requests or DOM), solve via API,
             inject the token, then click Next again.
        """
        try:
            dialog = page.get_by_role("dialog").first
            # Wait up to 15s for Instagram to render the security/CAPTCHA dialog.
            # It appears several seconds after the signup spinner, not immediately.
            try:
                await dialog.wait_for(state="visible", timeout=15_000)
            except Exception:
                return  # No dialog appeared — proceed directly to OTP/delivery choice
            dialog_text = await dialog.inner_text()
            logger.info("Security dialog: %s", dialog_text[:80].replace("\n", " "))
        except Exception:
            return  # no dialog

        # In headed mode: let the human solve it, then continue automatically.
        if not self._headless:
            print(
                "\n\n"
                ">>> CAPTCHA: Please solve the reCAPTCHA in the browser window. <<<\n"
                ">>> The script will continue automatically once the dialog closes. <<<\n",
                flush=True,
            )
            logger.info("Headed mode — waiting for human to solve reCAPTCHA (up to 5 min)...")
            for _ in range(60):
                await asyncio.sleep(5)
                try:
                    if not await page.get_by_role("dialog").first.is_visible():
                        logger.info("reCAPTCHA dialog closed — resuming")
                        return
                except Exception:
                    return
            raise RuntimeError("reCAPTCHA dialog still open after 5 minutes — timed out")

        # Step 1: try clicking the reCAPTCHA checkbox in the nested iframe.
        # The dialog embeds Instagram's proxy iframe (#captcha-recaptcha), which in turn
        # contains Google's reCAPTCHA anchor iframe.  Clicking the checkbox with Camoufox's
        # realistic fingerprint often results in instant auto-verification (green tick).
        logger.info("Attempting reCAPTCHA checkbox click via nested frame traversal...")
        checkbox_clicked = await _click_recaptcha_checkbox(page)
        if checkbox_clicked:
            logger.info("Checkbox clicked — waiting for reCAPTCHA auto-verification...")
            # Give reCAPTCHA 4s to auto-verify (green tick); if it opens image grid we'll need
            # the 2captcha path instead.
            await _pause(3.5, 4.5)

        # Step 2: click Next and see if dialog closes
        logger.info("Clicking Next in security dialog...")
        await _click_action_button(page, ["Next", "Continue"])
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
        await _pause(1.5, 2.5)

        try:
            still_visible = await page.get_by_role("dialog").first.is_visible()
        except Exception:
            still_visible = False

        if not still_visible:
            logger.info("Security dialog dismissed")
            return

        # Step 3: still stuck — try 2captcha token injection
        logger.info("Dialog still visible — attempting 2captcha solve")

        sitekey = captured_sitekey[0] if captured_sitekey else None
        if not sitekey:
            sitekey = await _extract_sitekey(page)

        api_key = os.getenv("TWOCAPTCHA_API_KEY", "")
        if not sitekey or not api_key:
            missing = []
            if not sitekey:
                missing.append("sitekey not extractable (check logs/ig_debug)")
            if not api_key:
                missing.append("TWOCAPTCHA_API_KEY not set in .env")
            raise RuntimeError(f"reCAPTCHA dialog could not be auto-passed: {'; '.join(missing)}")

        logger.info("Solving reCAPTCHA Enterprise via 2captcha (sitekey=%s)...", sitekey[:20])
        from twocaptcha import TwoCaptcha  # noqa: PLC0415

        solver = TwoCaptcha(api_key)
        result = await asyncio.to_thread(
            solver.recaptcha,
            sitekey=sitekey,
            url=page.url,
            enterprise=1,
        )
        token = result["code"]
        logger.info("reCAPTCHA solved (token_len=%d)", len(token))

        await page.evaluate(
            """(token) => {
                for (const sel of ['#g-recaptcha-response', 'textarea[name="g-recaptcha-response"]']) {
                    const el = document.querySelector(sel);
                    if (el) { el.value = token; el.dispatchEvent(new Event('change', {bubbles: true})); }
                }
            }""",
            token,
        )

        await _click_action_button(page, ["Next", "Continue"])
        await page.wait_for_load_state("networkidle", timeout=15_000)
        await _pause(1.5, 2.5)

    async def _handle_otp(self, page, account: InstagramAccount) -> None:
        # Log current page title/URL to help diagnose where we are
        try:
            logger.info("OTP step — URL: %s  title: %s", page.url, await page.title())
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

        # Wait for OTP page to render — it appears after CAPTCHA/delivery choice resolves.
        # Poll with a combined wait so we don't fail if the page needs a few more seconds.
        otp_field = None
        for _ in range(20):  # up to 20s
            otp_field = await _by_label(
                page,
                [
                    "Confirmation code",
                    "Security code",
                    "Verification code",
                    "Enter the code",
                    "Enter confirmation code",
                    "6-digit code",
                    "Code",
                ],
            ) or await _find(
                page,
                [
                    'input[maxlength="6"]',
                    'input[type="number"]',
                    'input[autocomplete="one-time-code"]',
                    'input[name="verificationCode"]',
                ],
            )
            if otp_field:
                break
            await asyncio.sleep(1)

        if not otp_field:
            await _check_for_block(page)
            raise RuntimeError("Cannot locate OTP input field — Instagram may have blocked signup")

        logger.info("Polling SMSPool for OTP (order %s, up to 5 min)...", account.smspool_order_id)
        otp = await self._manager.smspool.poll_otp(account.smspool_order_id, timeout=300, interval=5)
        if not otp:
            raise RuntimeError(f"OTP not received within 5 minutes for order {account.smspool_order_id}")

        otp = otp.replace(" ", "")
        logger.info("OTP received: %s", otp)
        await otp_field.click()
        await _type(otp_field, otp)
        await _pause(0.5, 1.0)

        if await _click_action_button(page, ["Next", "Submit", "Confirm", "Continue"]):
            # Wait for the OTP field to disappear — that's the clearest signal that
            # Instagram has accepted the code and is transitioning to the next page.
            # networkidle alone fires too early (the spinner is often client-side).
            try:
                await otp_field.wait_for(state="hidden", timeout=20_000)
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
            await _pause(3.0, 5.0)

        # Save a screenshot so we can see whether Instagram accepted the OTP or
        # is showing a rejection / additional challenge page.
        debug_dir = self._manager._base_dir / "logs" / "ig_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            shot = debug_dir / f"{account.smspool_order_id}_post_otp.png"
            await page.screenshot(path=str(shot), full_page=False)
            logger.info("Post-OTP screenshot: %s", shot)
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)

    # ---------------------------------------------------------------- VPN

    def _current_vpn_slug(self) -> str | None:
        """Identify current slug by matching wg0.conf endpoint against country configs."""
        try:
            text = self._wg_conf.read_text()
            m = re.search(r"Endpoint\s*=\s*(\S+)", text)
            if not m:
                return None
            current_ep = m.group(1).strip()
            for conf in self._countries_dir.glob("*.conf"):
                _, ep = _parse_peer(conf)
                if ep == current_ep:
                    return conf.stem
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
        return None

    def _resolve_vpn_slug(self, country: str) -> str:
        pool: dict[str, list[str]] = {}
        for conf in sorted(self._countries_dir.glob("*.conf")):
            cc = conf.stem.split("-")[0].upper()
            pool.setdefault(cc, []).append(conf.stem)
        cc = country.upper()
        if cc not in pool:
            raise ValueError(f"No WireGuard config available for country '{country}'")
        return random.choice(pool[cc])

    async def _switch_vpn(self, slug: str) -> None:
        conf = self._countries_dir / f"{slug}.conf"
        pubkey, endpoint = _parse_peer(conf)
        interface_block = _parse_interface_block(self._wg_conf)
        new_conf = f"{interface_block}\n\n[Peer]\nPublicKey = {pubkey}\nAllowedIPs = 0.0.0.0/0\nEndpoint = {endpoint}\n"
        self._wg_conf.write_text(new_conf)
        await asyncio.to_thread(subprocess.run, ["wg-quick", "down", "wg0"], check=True, capture_output=True)
        await asyncio.sleep(1)
        await asyncio.to_thread(subprocess.run, ["wg-quick", "up", "wg0"], check=True, capture_output=True)
        await asyncio.sleep(2)
        logger.info("VPN → %s (%s)", slug, endpoint)


# ------------------------------------------------------------------ helpers


def _parse_peer(conf: Path) -> tuple[str, str]:
    text = conf.read_text()
    pub = re.search(r"PublicKey\s*=\s*(.+)", text)
    ep = re.search(r"Endpoint\s*=\s*(.+)", text)
    if not pub or not ep:
        raise ValueError(f"Cannot parse [Peer] from {conf}")
    return pub.group(1).strip(), ep.group(1).strip()


def _parse_interface_block(conf: Path) -> str:
    text = conf.read_text()
    idx = text.find("[Peer]")
    return text[:idx].rstrip() if idx != -1 else text.rstrip()


async def _find(page, selectors: list[str]):
    """Return the first visible locator matching any CSS selector, or None."""
    for sel in selectors:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=2_000):
                return el
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
    return None


async def _wait_for_any_label(page, labels: list[str], timeout: int = 10_000) -> None:
    """
    Wait until at least one of the given label texts is visible in the page.
    Silently returns after timeout if none appear (caller's error handling takes over).
    """
    try:
        # Build a CSS selector that matches any <label> containing any of the strings.
        # Playwright's locator.wait_for() is the right primitive here.
        for label in labels:
            found = False
            try:
                await page.get_by_label(label, exact=False).first.wait_for(state="visible", timeout=timeout)
                found = True
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
            if found:
                return
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)


async def _by_label(page, labels: list[str]):
    """Return the first visible input matched by any label text (exact=False), or None."""
    for label in labels:
        try:
            el = page.get_by_label(label, exact=False).first
            if await el.is_visible(timeout=2_000):
                return el
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
    return None


async def _click_action_button(page, names: list[str], timeout: int = 5_000) -> bool:
    """
    Click the first visible button (role=button or button element) matching any name.
    Instagram uses div[role="button"] — get_by_role covers both native and ARIA buttons.
    Returns True if clicked, False if none found.
    """
    for name in names:
        try:
            el = page.get_by_role("button", name=name, exact=True).first
            if await el.is_visible(timeout=timeout):
                await el.click()
                return True
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
    return False


async def _type(field, text: str) -> None:
    """Type text with randomised inter-key delays to mimic human input."""
    for ch in text:
        await field.type(ch, delay=random.randint(40, 130))


async def _pause(lo: float, hi: float) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


async def _dismiss_overlays(page) -> None:
    """Click away cookie banners and login-nudge dialogs."""
    await _click_action_button(
        page,
        ["Allow all cookies", "Accept all", "Only allow essential cookies", "Not now"],
        timeout=1_500,
    )


async def _check_for_block(page) -> None:
    """Raise if Instagram has shown a suspicious-activity or CAPTCHA wall."""
    try:
        content = await page.content()
        lowered = content.lower()
        if any(kw in lowered for kw in ("suspicious", "unusual activity", "we detected", "verify it's you")):
            raise RuntimeError("Instagram flagged signup as suspicious — manual intervention required")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)


_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


async def _fill_birthday(page) -> None:
    """
    Fill the birthday comboboxes on the Instagram signup form.
    Instagram uses role=combobox / role=listbox / role=option — not <select>.
    If the birthday section isn't present, returns silently.
    """
    year = random.randint(*_BIRTHDAY_YEAR_RANGE)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    month_combo = page.get_by_role("combobox", name="Select Month")
    if not await month_combo.is_visible(timeout=3_000):
        return  # Birthday not on this page

    await _select_combobox_option(page, month_combo, _MONTH_NAMES[month - 1])
    await _pause(0.3, 0.6)

    day_combo = page.get_by_role("combobox", name="Select Day")
    await _select_combobox_option(page, day_combo, str(day))
    await _pause(0.3, 0.6)

    year_combo = page.get_by_role("combobox", name="Select Year")
    await _select_combobox_option(page, year_combo, str(year))
    await _pause(0.4, 0.9)


async def _select_combobox_option(page, combo, value: str) -> None:
    """Open an ARIA combobox and click the matching role=option."""
    await combo.click()
    await _pause(0.3, 0.6)
    # Options appear in a listbox; click the one matching value
    option = page.get_by_role("option", name=value, exact=True).first
    try:
        await option.click(timeout=5_000)
    except Exception:
        # Fallback: type value and press Enter (works for year which has many options)
        await combo.fill(value)
        await page.keyboard.press("Enter")
    await _pause(0.2, 0.4)


async def _skip_optional_steps(page) -> None:
    """Dismiss up to 6 rounds of optional post-signup prompts."""
    for _ in range(6):
        if not await _click_action_button(page, ["Not now", "Not Now", "Skip", "Cancel", "Maybe later", "Maybe Later"]):
            # Also try aria-label Close (icon buttons)
            try:
                el = page.get_by_role("button", name="Close")
                if await el.is_visible(timeout=1_500):
                    await el.click()
                    await _pause(0.7, 1.5)
                    continue
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
            break
        await _pause(0.7, 1.5)


async def _click_recaptcha_checkbox(page) -> bool:
    """
    Find and click the 'I'm not a robot' checkbox inside the nested reCAPTCHA iframes.
    Instagram embeds: page → #captcha-recaptcha (Instagram proxy) → Google anchor iframe.
    Returns True if the checkbox was clicked, False if not found.
    """
    # Iterate all frames — Playwright exposes every frame regardless of nesting depth.
    for frame in page.frames:
        url = frame.url or ""
        # Google's reCAPTCHA anchor frame contains the checkbox
        if "recaptcha" not in url.lower() and "captcha" not in url.lower():
            continue
        for sel in ("#recaptcha-anchor", ".recaptcha-checkbox-border", "[role='checkbox']"):
            try:
                el = frame.locator(sel).first
                if await el.is_visible(timeout=2_000):
                    await el.click()
                    logger.info("Clicked reCAPTCHA checkbox in frame %s", url[:60])
                    return True
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
    # Fallback: try Playwright's frame_locator nesting
    try:
        checkbox = page.frame_locator("#captcha-recaptcha").frame_locator("iframe").locator("#recaptcha-anchor").first
        if await checkbox.is_visible(timeout=3_000):
            await checkbox.click()
            logger.info("Clicked reCAPTCHA checkbox via frame_locator nesting")
            return True
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)
    logger.info("reCAPTCHA checkbox not found in any frame")
    return False


async def _extract_sitekey(page) -> str | None:
    """Try every known location for the reCAPTCHA sitekey."""
    # Main page DOM
    try:
        sk = await page.evaluate("() => document.querySelector('[data-sitekey]')?.dataset.sitekey ?? null")
        if sk:
            return sk
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)
    # All frames
    for frame in page.frames:
        try:
            sk = await frame.evaluate("() => document.querySelector('[data-sitekey]')?.dataset.sitekey ?? null")
            if sk:
                logger.info("Sitekey found in frame %s", (frame.url or "")[:60])
                return sk
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
        try:
            # Some reCAPTCHA builds expose it on window.___grecaptcha_cfg
            sk = await frame.evaluate("""
                () => {
                    const cfg = window.___grecaptcha_cfg;
                    if (!cfg || !cfg.clients) return null;
                    for (const k of Object.keys(cfg.clients)) {
                        const c = cfg.clients[k];
                        for (const k2 of Object.keys(c || {})) {
                            if (c[k2]?.sitekey) return c[k2].sitekey;
                        }
                    }
                    return null;
                }
            """)
            if sk:
                return sk
        except Exception as exc:
            logger.debug("Suppressed: %s", exc)
    return None


async def _dismiss_security_dialog(page) -> None:
    """
    Handle the 'Help us confirm it's you' interstitial that appears after signup submit.
    The dialog has no inputs — just a Next button. Click it up to 3 times in case
    Instagram chains multiple confirmation steps.
    """
    for _ in range(3):
        try:
            # Use .first to avoid strict-mode violation when multiple dialogs exist
            dialog = page.get_by_role("dialog").first
            if not await dialog.is_visible():
                break
            text = await dialog.inner_text()
            logger.info("Dialog detected: %s", text[:120].replace("\n", " "))
            clicked = await _click_action_button(page, ["Next", "Continue", "OK", "Got it"])
            if not clicked:
                # Dialog visible but no recognised button — don't loop forever
                break
            await page.wait_for_load_state("networkidle", timeout=12_000)
            await _pause(1.0, 2.0)
        except Exception:
            break


async def _handle_delivery_choice(page) -> None:
    """
    Instagram sometimes shows a 'How would you like to receive your confirmation code?'
    step with SMS / Email radio buttons. Select SMS (phone) if present, then click Next.
    """
    try:
        # Wait briefly for delivery choice page — it appears after CAPTCHA resolves
        sms_opt = page.get_by_role("radio", name="SMS").first
        try:
            await sms_opt.wait_for(state="visible", timeout=5_000)
        except Exception:
            return  # No delivery choice page
        if await sms_opt.is_visible():
            logger.info("Delivery choice page detected — selecting SMS")
            await sms_opt.click()
            await _pause(0.5, 1.0)
            await _click_action_button(page, ["Next", "Send code", "Send"])
            await page.wait_for_load_state("networkidle", timeout=12_000)
            await _pause(1.0, 2.0)
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)


async def _save_shot(page, debug_dir: Path, account_id: str, label: str) -> None:
    try:
        shot = debug_dir / f"{account_id}_{label}.png"
        await page.screenshot(path=str(shot), full_page=False)
        logger.info("Screenshot: %s", shot)
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)


async def _verify_logged_in(page, account_id: str, debug_dir: Path) -> None:
    """
    Confirm we're authenticated before exporting cookies.

    Checks the current page first (without navigating) to avoid interrupting an
    in-progress post-OTP session establishment. Only navigates to the home page
    if the current URL is neutral (not a login/challenge page and not already the feed).
    Raises RuntimeError if no authenticated feed indicator is found.
    """

    def _has_bad_url(url: str) -> bool:
        return any(kw in url for kw in ("login", "emailsignup", "challenge", "checkpoint", "two_factor", "verify"))

    async def _check_feed_visible() -> bool:
        for label in ("Home", "Search"):
            try:
                if await page.locator(f'svg[aria-label="{label}"]').first.is_visible(timeout=3_000):
                    return True
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
        return False

    # --- Step 1: check current page (don't navigate yet) ---
    url = page.url
    logger.info("Post-signup current URL: %s", url)

    if _has_bad_url(url):
        await _save_shot(page, debug_dir, account_id, "post_signup")
        raise RuntimeError(f"Signup did not complete — browser is on: {url}")

    # Give the current page a moment to settle if Instagram is mid-transition
    await asyncio.sleep(3)
    if await _check_feed_visible():
        logger.info("Login confirmed on current page for account %s", account_id[:8])
        await _save_shot(page, debug_dir, account_id, "post_signup")
        return

    # --- Step 2: navigate home only if current page is ambiguous ---
    # (e.g. still on a post-OTP transition page that isn't the feed or a challenge)
    logger.info("Feed not visible on current page — navigating to home to confirm")
    await page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=30_000)
    try:
        await page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception as exc:
        logger.debug("Suppressed: %s", exc)
    await asyncio.sleep(2)

    url = page.url
    logger.info("Post-navigation URL: %s", url)
    await _save_shot(page, debug_dir, account_id, "post_signup")

    if _has_bad_url(url):
        raise RuntimeError(f"Signup did not complete — browser is on: {url}")

    logged_in = await _check_feed_visible()
    if not logged_in:
        raise RuntimeError(
            f"Signup appeared to complete but browser shows no authenticated feed — URL: {url}. "
            "Check logs/ig_debug/ for post_signup screenshot."
        )

    logger.info("Login confirmed for account %s", account_id[:8])


async def _export_cookies_netscape(page, cookie_path: Path) -> None:
    """Write instagram.com session cookies in Netscape format for yt-dlp."""
    cookies = await page.context.cookies(["https://www.instagram.com"])
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c.get("domain", ".instagram.com")
        if not domain.startswith("."):
            domain = f".{domain}"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure") else "FALSE"
        expiry = int(c["expires"]) if c.get("expires", -1) >= 0 else 0
        name = c.get("name", "")
        value = c.get("value", "")
        lines.append(f"{domain}\tTRUE\t{path}\t{secure}\t{expiry}\t{name}\t{value}")
    cookie_path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ generators

_FIRST_NAMES = [
    "James",
    "Sarah",
    "Michael",
    "Emma",
    "Daniel",
    "Laura",
    "Thomas",
    "Anna",
    "Chris",
    "Megan",
    "Ryan",
    "Claire",
    "Luke",
    "Amy",
    "Sean",
    "Kate",
]
_LAST_NAMES = [
    "Walker",
    "Mills",
    "Clarke",
    "Reid",
    "Burns",
    "Shaw",
    "Grant",
    "Dean",
    "Holt",
    "Cross",
    "West",
    "Stone",
    "Lane",
    "Ford",
    "Blake",
    "Nash",
]
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
]


def _gen_username() -> str:
    return f"{random.choice(_ADJECTIVES)}_{random.choice(_NOUNS)}{random.randint(10, 999)}"


def _gen_full_name() -> str:
    return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"


def _gen_password() -> str:
    chars = string.ascii_letters + string.digits + "!@#$"
    return "".join(secrets.choice(chars) for _ in range(16))
