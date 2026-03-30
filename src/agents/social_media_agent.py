"""
Vision-driven social media agent that controls a physical Android device via ADB.

Instead of hardcoded tap coordinates, every action goes through a
screenshot → Gemini Vision → execute loop, making it resilient to
UI changes across app versions and screen sizes.
"""

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from google import genai

from src.gateways.adb_device import ADBDevice

logger = logging.getLogger("osia.social_agent")

# How long to wait for a screen transition after a tap/swipe
_TRANSITION_WAIT = (1.5, 3.0)  # random range in seconds — looks more human

# Maximum vision-action iterations per high-level command
_MAX_STEPS = 15

# Android launcher/home screen package names
_HOME_SCREEN_PACKAGES = {
    "com.android.launcher",
    "com.android.launcher2",
    "com.android.launcher3",
    "com.google.android.apps.nexuslauncher",
    "com.sec.android.app.launcher",
    "com.miui.home",
    "com.huawei.android.launcher",
    "com.oneplus.launcher",
    "com.motorola.launcher3",
}


class HomeScreenError(Exception):
    """Raised when the agent detects it has landed on the Android home screen."""


@dataclass
class ScreenState:
    """Snapshot of what the agent currently sees."""

    screenshot_path: str
    description: str = ""
    elements: list[dict] = field(default_factory=list)


@dataclass
class ActionResult:
    """Outcome of a high-level agent action."""

    success: bool
    data: str = ""
    error: str = ""


class SocialMediaAgent:
    """
    Autonomous agent that drives a physical Android phone through social media
    apps using Gemini Vision to interpret the screen and decide actions.

    Usage:
        agent = SocialMediaAgent(adb, gemini_client)
        comments = await agent.read_comments("https://www.instagram.com/p/ABC123/")
        await agent.post_comment("https://www.instagram.com/p/ABC123/", "Great post!")
    """

    def __init__(
        self,
        adb: ADBDevice,
        gemini_client: genai.Client,
        model_id: str = "gemini-2.5-flash",
        base_dir: str | Path | None = None,
    ):
        self.adb = adb
        self.gemini = gemini_client
        self.model_id = model_id
        self.base_dir = Path(base_dir or os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
        self._scratch_dir = self.base_dir / "tmp" / "social_agent"
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        # Cached screen dimensions — fetched lazily on first use
        self._screen_width: int | None = None
        self._screen_height: int | None = None

    # ------------------------------------------------------------------
    # Core vision-action loop
    # ------------------------------------------------------------------

    async def _screenshot(self) -> str:
        """Take a screenshot and return the local file path."""
        path = str(self._scratch_dir / "screen.png")
        await self.adb.take_screenshot(path)
        # Lazily cache screen dimensions on first screenshot
        if self._screen_width is None:
            self._screen_width, self._screen_height = await self.adb.get_screen_size()
            logger.info("Screen resolution cached: %dx%d", self._screen_width, self._screen_height)
        return path

    async def _human_delay(self):
        """Random pause to mimic human interaction cadence."""
        delay = random.uniform(*_TRANSITION_WAIT)
        await asyncio.sleep(delay)

    async def _analyze_screen(self, screenshot_path: str, goal: str) -> dict:
        """
        Send a screenshot to Gemini Vision and ask it to interpret the screen
        relative to the current goal.

        Returns a dict with:
            - description: what's on screen
            - action: next action to take (tap/swipe/type/scroll_down/scroll_up/back/done/fail)
            - coordinates: {x, y} if action is tap
            - text: string if action is type
            - data: any extracted data (e.g. comment text)
            - reasoning: why this action was chosen
        """
        screen_file = self.gemini.files.upload(file=screenshot_path)

        # Build resolution context string if available
        if self._screen_width and self._screen_height:
            resolution_context = (
                f"The screen resolution is exactly {self._screen_width}x{self._screen_height} pixels "
                f"(width x height). All coordinates you provide MUST be within these bounds: "
                f"x between 0 and {self._screen_width}, y between 0 and {self._screen_height}. "
                f"The coordinate origin (0,0) is the top-left corner."
            )
        else:
            resolution_context = "Coordinates use pixels from the top-left origin (0,0)."

        # Build a layout guide based on known resolution
        if self._screen_width and self._screen_height:
            safe_y_start = int(self._screen_height * 0.75)
            right_bar_x = int(self._screen_width * 0.85)
            danger_zone_note = (
                f"SCREEN LAYOUT GUIDE ({self._screen_width}x{self._screen_height}):\n\n"
                f"FEED POSTS (static image, text post, standard scrolling feed):\n"
                f"  - Content area occupies y=0 to y={safe_y_start} — DO NOT tap here, it expands full-screen.\n"
                f"  - Interaction bar (like, comment, share buttons): y={safe_y_start} to y={self._screen_height} (bottom 25%).\n\n"
                f"REELS / SHORTS / FULL-SCREEN VERTICAL VIDEO:\n"
                f"  - Identified by a full-screen vertical video and a vertical column of icons on the right edge.\n"
                f"  - Like (heart), comment (speech bubble), share (arrow), audio icons are stacked on the\n"
                f"    RIGHT SIDE RAIL at x >= {right_bar_x}. These ARE safe to tap at any y value.\n"
                f"  - Tapping the CENTER of the video (x < {right_bar_x}) pauses/unpauses — avoid unless intentional.\n"
                f"  - The comment text input bar (when comments are open) is at the BOTTOM: y >= {safe_y_start}.\n\n"
                f"Identify the layout type first, then choose coordinates accordingly."
            )
        else:
            danger_zone_note = (
                "SCREEN LAYOUT:\n"
                "Feed posts: interaction bar (like/comment/share) is at the BOTTOM of the screen.\n"
                "Reels/Shorts: like/comment/share buttons are on the RIGHT SIDE RAIL (rightmost ~15% of width) "
                "and are safe to tap at any height. Do not tap the center of a Reel as it plays/pauses the video."
            )

        prompt = f"""You are an autonomous agent controlling a physical Android phone.
Your current goal: {goal}

Screen information: {resolution_context}

{danger_zone_note}

Analyze this screenshot and decide the SINGLE next action to take.

Respond with ONLY valid JSON (no markdown, no code fences):
{{
    "description": "Brief description of what's currently on screen",
    "action": "tap|swipe_down|swipe_up|type|tap_and_type|type_and_submit|submit|back|done|fail",
    "coordinates": {{"x": 540, "y": 1200}},
    "swipe": {{"x1": 540, "y1": 1500, "x2": 540, "y2": 500}},
    "text": "text to type if action is type or tap_and_type",
    "data": "any extracted data relevant to the goal (e.g. comments text, post content)",
    "reasoning": "why you chose this action"
}}

Rules:
- "coordinates" is required when action is "tap" or "tap_and_type" — tap the exact center of the UI element.
- "swipe" is required when action is "swipe_down" or "swipe_up".
- "text" is required when action is "type" or "tap_and_type".
- "tap_and_type" should be used when you need to select an empty input field and type into it (WITHOUT submitting).
- "type_and_submit" taps a field, types the text, AND presses Enter to submit — use this for posting comments. It is a single atomic action; never use tap_and_type + submit separately for a comment.
- "submit" presses the Enter/Return key — use it if text is already in the field and just needs sending.
- Use "done" when the goal has been achieved. Put the final result in "data".
- Use "fail" if the goal cannot be achieved from the current state.
- If you need to scroll to see more content, use "swipe_down" or "swipe_up".
- Be precise with coordinates — they must land on the intended UI element.
- For FEED posts: only tap UI chrome in the bottom interaction bar. Do not tap post images or video thumbnails.
- For REELS/SHORTS: the action buttons (heart/comment/share) are on the RIGHT SIDE RAIL — tap those directly.
- The comment input box is a rounded rectangle near the BOTTOM of the screen. Tap its CENTER.
- CRITICAL — COMMENT FLOW: (1) Use tap_and_type ONCE to focus the comment field and type the text. (2) Use "submit" to send it via Enter. Never tap the input field again or retype — if the text is already in the field, use "submit" immediately.
- CRITICAL — GBOARD CLIPBOARD: If a Gboard clipboard strip appears at the top of the keyboard (showing clipboard snippets), tap the ✕ close button on that strip to dismiss it. Do NOT press back — that closes the keyboard entirely.
- If you see a full-screen image or video with no UI controls visible, use "back" to return to the post view.
- CRITICAL: If you can see the Android home screen (wallpaper, app icons, clock/date, no social media app visible), use "fail" immediately — do not try to navigate back.
"""

        response = self.gemini.models.generate_content(
            model=self.model_id,
            contents=[screen_file, prompt],
        )

        text = response.text.strip()
        # Strip markdown code fences if the model wraps them
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Gemini sometimes wraps JSON in prose reasoning — try to extract it
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.warning("Gemini returned non-JSON response: %s", text[:200])
            return {
                "description": text[:200],
                "action": "fail",
                "reasoning": "Could not parse vision response as JSON",
                "_parse_error": True,
            }

    async def _execute_action(self, action: dict):
        """Execute a single action returned by the vision model."""
        act = action.get("action", "fail")

        if act == "tap":
            coords = action.get("coordinates", {})
            x, y = coords.get("x", 0), coords.get("y", 0)
            logger.info("Tapping (%d, %d) — %s", x, y, action.get("reasoning", ""))
            await self.adb.tap(x, y)

        elif act in ("swipe_down", "swipe_up"):
            swipe = action.get("swipe")
            if swipe:
                logger.info("Swiping — %s", action.get("reasoning", ""))
                await self.adb.swipe(
                    swipe["x1"],
                    swipe["y1"],
                    swipe["x2"],
                    swipe["y2"],
                )
            else:
                # Fallback: generic scroll in the center of the screen
                w, h = await self.adb.get_screen_size()
                cx = w // 2
                if act == "swipe_down":
                    await self.adb.swipe(cx, h * 3 // 4, cx, h // 4)
                else:
                    await self.adb.swipe(cx, h // 4, cx, h * 3 // 4)

        elif act == "type":
            text = action.get("text", "")
            logger.info("Typing: %s", text[:50])
            await self.adb.type_text(text)

        elif act == "tap_and_type":
            coords = action.get("coordinates", {})
            x, y = coords.get("x", 0), coords.get("y", 0)
            text = action.get("text", "")
            logger.info("Tapping (%d, %d) and typing: %s", x, y, text[:50])
            await self.adb.tap(x, y)
            await asyncio.sleep(1.0)  # Wait for keyboard/focus
            await self.adb.type_text(text)

        elif act == "type_and_submit":
            # Atomic: tap the input field, type the text, then press Enter to submit.
            # Avoids the vision-loop re-entry that causes comment duplication.
            coords = action.get("coordinates", {})
            x, y = coords.get("x", 0), coords.get("y", 0)
            text = action.get("text", "")
            logger.info("Typing and submitting at (%d, %d): %s", x, y, text[:50])
            await self.adb.tap(x, y)
            await asyncio.sleep(1.0)
            await self.adb.type_text(text)
            await asyncio.sleep(0.5)
            await self.adb.press_send()

        elif act == "submit":
            logger.info("Submitting via Send key — %s", action.get("reasoning", ""))
            await self.adb.press_send()

        elif act == "back":
            logger.info("Pressing back — %s", action.get("reasoning", ""))
            await self.adb.press_back()

        await self._human_delay()

    async def _run_goal(self, goal: str, pre_url: str | None = None) -> ActionResult:
        """
        Execute a vision-action loop until the goal is achieved or we hit the step limit.
        Optionally opens a URL first to navigate to the right content.

        Stuck detection: if the screen description repeats for 2+ consecutive steps, or
        the model returns 'fail', press the Android back button to escape (e.g. full-screen
        image/video overlay). Back-press recovery is capped at 3 attempts to avoid loops.
        """
        if pre_url:
            await self.adb.open_url(pre_url)
            await asyncio.sleep(4)  # wait for app to load the deep link

        collected_data: list[str] = []
        last_description: str = ""
        stuck_count: int = 0
        back_presses: int = 0
        parse_retries: int = 0
        _MAX_BACK_PRESSES = 3
        _MAX_PARSE_RETRIES = 1
        _STUCK_THRESHOLD = 2

        for step in range(_MAX_STEPS):
            screenshot_path = await self._screenshot()

            # Detect home screen — agent has been ejected from the app
            foreground = await self.adb.get_foreground_app()
            if any(foreground.startswith(pkg) for pkg in _HOME_SCREEN_PACKAGES):
                logger.warning("Home screen detected (foreground=%s) — aborting task.", foreground)
                raise HomeScreenError(f"Landed on home screen ({foreground}), task aborted.")

            analysis = await self._analyze_screen(screenshot_path, goal)

            description = analysis.get("description", "")
            action = analysis.get("action", "")

            logger.info(
                "Step %d/%d — screen: %s | action: %s",
                step + 1,
                _MAX_STEPS,
                description[:80],
                action,
            )

            # Vision-based home screen fallback: foreground app check can miss some launchers.
            # If the model itself reports seeing the home screen, fail immediately — Back presses
            # from the home screen do nothing useful and just burn steps.
            _desc_lower = description.lower()
            if "home screen" in _desc_lower or "android home" in _desc_lower:
                logger.warning("Vision confirmed home screen — failing fast (foreground=%s).", foreground)
                return ActionResult(
                    success=False,
                    error="Tool encountered android home screen error — app was closed during task.",
                )

            # Accumulate any data the model extracts along the way
            if analysis.get("data"):
                data_val = analysis["data"]
                collected_data.append(data_val if isinstance(data_val, str) else json.dumps(data_val, indent=2))

            if action == "done":
                return ActionResult(
                    success=True,
                    data="\n\n".join(collected_data) if collected_data else analysis.get("data", ""),
                )

            # Stuck detection: same screen description repeating or explicit fail
            if description and description == last_description:
                stuck_count += 1
            else:
                stuck_count = 0
            last_description = description

            if action == "fail" or stuck_count >= _STUCK_THRESHOLD:
                # Parse errors are model output failures, not UI navigation failures.
                # Retrying the screenshot is safer than pressing back, which can walk
                # the user out of the app (feed → app drawer → home screen).
                if analysis.get("_parse_error") and parse_retries < _MAX_PARSE_RETRIES:
                    logger.info(
                        "Step %d: vision model returned non-JSON — retrying screenshot (attempt %d/%d)",
                        step + 1,
                        parse_retries + 1,
                        _MAX_PARSE_RETRIES,
                    )
                    parse_retries += 1
                    stuck_count = 0
                    last_description = ""
                    await asyncio.sleep(1.5)
                    continue

                if back_presses < _MAX_BACK_PRESSES:
                    # Guard: don't press back if we're already on the home screen —
                    # that just opens the app drawer and burns steps.
                    fg = await self.adb.get_foreground_app()
                    if any(fg.startswith(pkg) for pkg in _HOME_SCREEN_PACKAGES):
                        return ActionResult(
                            success=False,
                            error="Landed on home screen — aborting back-press recovery.",
                        )
                    logger.warning(
                        "Agent stuck (action=%s, stuck_count=%d) — pressing back button (attempt %d/%d)",
                        action,
                        stuck_count,
                        back_presses + 1,
                        _MAX_BACK_PRESSES,
                    )
                    await self.adb.press_back()
                    await asyncio.sleep(1)
                    stuck_count = 0
                    last_description = ""
                    back_presses += 1
                    continue
                else:
                    return ActionResult(
                        success=False,
                        error=analysis.get("reasoning", "Agent could not achieve the goal after back-button recovery."),
                    )

            await self._execute_action(analysis)

        return ActionResult(
            success=False,
            error=f"Reached maximum steps ({_MAX_STEPS}) without completing the goal.",
            data="\n\n".join(collected_data),
        )

    # ------------------------------------------------------------------
    # High-level social media actions
    # ------------------------------------------------------------------

    async def read_comments(self, post_url: str, max_scrolls: int = 3) -> ActionResult:
        """
        Navigate to a social media post and extract comments.

        Args:
            post_url: Direct URL to the post (Instagram, TikTok, Facebook, YouTube).
            max_scrolls: Hint for how many scroll-loads of comments to capture.

        Returns:
            ActionResult with extracted comments in .data
        """
        goal = (
            f"Navigate to the comments section of this post and extract all visible comments "
            f"(author + text). "
            f"IMPORTANT — for Instagram Reels: tap the speech-bubble/comment icon on the RIGHT SIDE of "
            f"the reel (the chat-bubble icon in the vertical row of action buttons). Do NOT tap the "
            f"'Add a comment...' text bar at the bottom — that opens a keyboard, not the comment thread. "
            f"If a keyboard appears at any point, use the 'back' action to dismiss it. "
            f"Once the comments panel is open, scroll down up to {max_scrolls} times to load more. "
            f"When you have collected the comments, use 'done' and put ALL the extracted "
            f"comments in the 'data' field as structured text."
        )
        return await self._run_goal(goal, pre_url=post_url)

    async def post_comment(self, post_url: str, comment_text: str) -> ActionResult:
        """
        Navigate to a post and leave a comment.

        Args:
            post_url: Direct URL to the post.
            comment_text: The comment to post.

        Returns:
            ActionResult indicating success/failure.
        """
        goal = (
            f"Post a comment on this social media post. Follow these steps in order:\n\n"
            f"STEP 1 — Open comments: tap the comment icon (speech bubble) to open the comment section.\n"
            f"STEP 2 — Type ONCE: use 'tap_and_type' on the comment input field (rounded box at BOTTOM "
            f"of screen) to type the comment. Do this EXACTLY ONCE.\n"
            f"STEP 3 — Submit: use the 'submit' action to press Enter and post the comment.\n"
            f"STEP 4 — Confirm: once the comment appears in the list, use 'done'.\n\n"
            f"IMPORTANT: If the comment text is already in the input field, skip to STEP 3 immediately — "
            f"do NOT retype it, just use 'submit'.\n\n"
            f'Comment text: "{comment_text}"\n\n'
            f"Do NOT tap any post image, video content, or content area outside the comment UI."
        )
        return await self._run_goal(goal, pre_url=post_url)

    async def reply_to_comment(self, post_url: str, target_author: str, reply_text: str) -> ActionResult:
        """
        Navigate to a post, find a specific comment by author, and reply to it.

        Args:
            post_url: Direct URL to the post.
            target_author: Username of the comment to reply to.
            reply_text: The reply text.

        Returns:
            ActionResult indicating success/failure.
        """
        goal = (
            f"Navigate to the comments section of this post. "
            f"Find the comment by the user '{target_author}'. "
            f"Tap the Reply button on that specific comment — it is a small text link below the comment, NOT in the post image area. "
            f"The reply input field will appear near the BOTTOM of the screen. "
            f"Use 'tap_and_type' to tap the CENTER of the reply input field and type the reply, then submit it:\n\n"
            f'Reply: "{reply_text}"\n\n'
            f"Once the reply is posted, use 'done'. "
            f"If you cannot find the comment by '{target_author}', scroll down to look for it. "
            f"If still not found after scrolling, use 'fail'."
        )
        return await self._run_goal(goal, pre_url=post_url)

    async def read_post_content(self, post_url: str) -> ActionResult:
        """
        Navigate to a post and extract its full content (caption, media description, stats).

        Returns:
            ActionResult with post content in .data
        """
        goal = (
            "Read and extract the full content of this post including: "
            "the author/username, caption/description text, number of likes, "
            "number of comments, and any other visible metadata. "
            "Put all extracted information in the 'data' field and use 'done'."
        )
        return await self._run_goal(goal, pre_url=post_url)

    async def _read_engagement_from_current_screen(self) -> dict:
        """
        Read engagement counts (likes, comments, shares) from whatever is currently
        on screen — no navigation.  Sends a single screenshot to Gemini Vision and
        parses the JSON response.

        Returns:
            dict with keys ``likes``, ``comments``, ``shares`` (int or None each).
            Returns an empty dict on failure.
        """
        try:
            screenshot_path = await self._screenshot()
        except Exception as exc:
            logger.warning("_read_engagement_from_current_screen: screenshot failed: %s", exc)
            return {}

        screen_file = self.gemini.files.upload(file=screenshot_path)

        if self._screen_width and self._screen_height:
            right_bar_x = int(self._screen_width * 0.85)
            layout_hint = (
                f"Screen resolution: {self._screen_width}x{self._screen_height} pixels. "
                f"For Reels/Shorts (full-screen vertical video), engagement icons are stacked in "
                f"the right-side rail (x >= {right_bar_x}): heart icon = likes, speech-bubble icon = comments, "
                f"arrow/paper-plane icon = shares. Each icon has a number directly below or beside it. "
                f"For feed posts (static image with bottom bar), the like/comment/share counts appear "
                f"in the interaction bar at the bottom of the screen."
            )
        else:
            layout_hint = (
                "For Reels/Shorts, engagement icons (heart = likes, speech bubble = comments, "
                "arrow = shares) are in the right-side rail with counts beside them. "
                "For feed posts, the interaction bar with counts is at the bottom."
            )

        prompt = (
            "You are reading a social media post or Reel on an Android phone.\n"
            "Your ONLY task is to find and read the engagement counts shown next to the icons.\n\n"
            f"{layout_hint}\n\n"
            "Counts may be formatted as:\n"
            "  - Plain integers: 1234\n"
            "  - K-abbreviated: 1.2K → 1200, 45K → 45000\n"
            "  - M-abbreviated: 4.5M → 4500000\n"
            "Convert any abbreviations to whole integers.\n\n"
            "Respond with ONLY valid JSON (no markdown fences, no prose):\n"
            '{"likes": <integer or null>, "comments": <integer or null>, "shares": <integer or null>}\n\n'
            "Use null for any count that is not visible or clearly readable. Do not guess."
        )

        try:
            response = self.gemini.models.generate_content(
                model=self.model_id,
                contents=[screen_file, prompt],
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            result = json.loads(text.strip())
            logger.info("_read_engagement_from_current_screen: %s", result)
            return result
        except Exception as exc:
            logger.warning("_read_engagement_from_current_screen: Gemini/parse error: %s", exc)
            return {}

    async def read_engagement_counts(self, post_url: str) -> dict:
        """
        Open a post/reel, read engagement counts from the loaded screen, then navigate back.

        Used when the phone is not already on the post (e.g. the tool-call path).
        For cases where the reel is already on screen, call
        ``_read_engagement_from_current_screen()`` directly.

        Returns:
            dict with keys ``likes``, ``comments``, ``shares`` (int or None each).
        """
        await self.adb.open_url(post_url)
        await asyncio.sleep(3)
        result = await self._read_engagement_from_current_screen()
        try:
            await self.adb.press_back()
        except Exception:
            pass  # best-effort back navigation; failure is non-fatal
        return result

    async def like_post(self, post_url: str) -> ActionResult:
        """Navigate to a post and like it."""
        goal = (
            "Find and tap the Like/Heart button on this post. "
            "If the post is already liked, use 'done' immediately. "
            "Once liked, use 'done'."
        )
        return await self._run_goal(goal, pre_url=post_url)

    async def follow_account(self, profile_url: str) -> ActionResult:
        """Navigate to a profile and follow the account."""
        goal = (
            "Find and tap the Follow button on this profile. "
            "If already following, use 'done' immediately. "
            "Once the follow action is confirmed, use 'done'."
        )
        return await self._run_goal(goal, pre_url=profile_url)

    async def execute_custom(self, url: str, instruction: str) -> ActionResult:
        """
        Execute an arbitrary instruction on the phone.
        This is the escape hatch for any social media action not covered above.

        Args:
            url: URL to open first (or empty string to skip navigation).
            instruction: Free-form instruction for the vision agent.
        """
        return await self._run_goal(instruction, pre_url=url if url else None)

    async def generate_infographic_via_phone(self, image_prompt: str) -> bytes | None:
        """
        Fallback infographic generator: drives the Gemini Android app via ADB
        to produce an image when the Gemini API is over capacity.

        Opens the Gemini app, submits the image prompt, waits for generation,
        screenshots the result, and returns the PNG bytes.
        Returns None if the flow fails.
        """
        GEMINI_APP = "com.google.android.apps.bard"
        logger.info("Infographic API unavailable — falling back to Gemini phone app.")

        # Launch the Gemini app fresh
        await self.adb.wake_and_unlock()
        await self.adb._run_checked(
            [
                "shell",
                "am",
                "start",
                "-n",
                f"{GEMINI_APP}/com.google.android.apps.bard.ui.BardActivity",
            ]
        )
        await asyncio.sleep(3)

        # Use the vision loop to navigate to a new chat and submit the prompt
        setup_result = await self._run_goal(
            goal=(
                "Open a new chat if one is not already open. "
                "Tap the message input field, type the following prompt exactly, "
                "then tap the send button. "
                f"PROMPT: {image_prompt[:800]}"
            )
        )

        if not setup_result.success:
            logger.warning("Phone Gemini prompt submission failed: %s", setup_result.error)
            return None

        # Wait for image generation — Gemini app typically takes 10–30s
        logger.info("Waiting for Gemini app to generate image...")
        await asyncio.sleep(20)

        # Use the vision loop to confirm the image is visible and capture it
        confirm_result = await self._run_goal(
            goal=(
                "An AI-generated image should now be visible in the chat. "
                "If it is still loading, wait and check again. "
                "Once the image is fully rendered, report done. "
                "If an error message is shown instead, report fail."
            )
        )

        if not confirm_result.success:
            logger.warning("Gemini app did not produce an image: %s", confirm_result.error)
            return None

        # Take a final clean screenshot and return the raw PNG bytes
        screenshot_path = str(self._scratch_dir / "infographic_capture.png")
        await self.adb.take_screenshot(screenshot_path)

        with open(screenshot_path, "rb") as f:
            return f.read()
