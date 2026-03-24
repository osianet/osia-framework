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
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from google import genai
from google.genai import types
from src.gateways.adb_device import ADBDevice

logger = logging.getLogger("osia.social_agent")

# How long to wait for a screen transition after a tap/swipe
_TRANSITION_WAIT = (1.5, 3.0)  # random range in seconds — looks more human

# Maximum vision-action iterations per high-level command
_MAX_STEPS = 15


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
        self.base_dir = Path(base_dir or os.getenv(
            "OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent
        ))
        self._scratch_dir = self.base_dir / "tmp" / "social_agent"
        self._scratch_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core vision-action loop
    # ------------------------------------------------------------------

    async def _screenshot(self) -> str:
        """Take a screenshot and return the local file path."""
        path = str(self._scratch_dir / "screen.png")
        await self.adb.take_screenshot(path)
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

        prompt = f"""You are an autonomous agent controlling a physical Android phone.
Your current goal: {goal}

Analyze this screenshot and decide the SINGLE next action to take.

Respond with ONLY valid JSON (no markdown, no code fences):
{{
    "description": "Brief description of what's currently on screen",
    "action": "tap|swipe_down|swipe_up|type|tap_and_type|back|done|fail",
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
- "tap_and_type" should be used when you need to select an input field and immediately type into it.
- Use "done" when the goal has been achieved. Put the final result in "data".
- Use "fail" if the goal cannot be achieved from the current state.
- If you need to scroll to see more content, use "swipe_down" or "swipe_up".
- Be precise with coordinates — they must land on the intended UI element.
- If a keyboard is visible and you need to submit, tap the Send/Post button rather than using Enter.
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
            logger.warning("Gemini returned non-JSON response: %s", text[:200])
            return {
                "description": text[:200],
                "action": "fail",
                "reasoning": "Could not parse vision response as JSON",
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
                    swipe["x1"], swipe["y1"], swipe["x2"], swipe["y2"],
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

        elif act == "back":
            logger.info("Pressing back — %s", action.get("reasoning", ""))
            await self.adb.press_back()

        await self._human_delay()

    async def _run_goal(self, goal: str, pre_url: str | None = None) -> ActionResult:
        """
        Execute a vision-action loop until the goal is achieved or we hit the step limit.
        Optionally opens a URL first to navigate to the right content.
        """
        if pre_url:
            await self.adb.open_url(pre_url)
            await asyncio.sleep(4)  # wait for app to load the deep link

        collected_data: list[str] = []

        for step in range(_MAX_STEPS):
            screenshot_path = await self._screenshot()
            analysis = await self._analyze_screen(screenshot_path, goal)

            logger.info(
                "Step %d/%d — screen: %s | action: %s",
                step + 1, _MAX_STEPS,
                analysis.get("description", "?")[:80],
                analysis.get("action", "?"),
            )

            # Accumulate any data the model extracts along the way
            if analysis.get("data"):
                collected_data.append(analysis["data"])

            if analysis["action"] == "done":
                return ActionResult(
                    success=True,
                    data="\n\n".join(collected_data) if collected_data else analysis.get("data", ""),
                )

            if analysis["action"] == "fail":
                return ActionResult(
                    success=False,
                    error=analysis.get("reasoning", "Agent could not achieve the goal."),
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
            f"(author + text). Scroll down up to {max_scrolls} times to load more comments. "
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
            f"Navigate to the comments section of this post. "
            f"Tap the comment input field, type the following comment, "
            f"then tap the Post/Send button to submit it.\n\n"
            f"Comment to post: \"{comment_text}\"\n\n"
            f"Once the comment is successfully posted (you can see it appear), use 'done'. "
            f"If the comment fails to post, use 'fail' with the reason."
        )
        return await self._run_goal(goal, pre_url=post_url)

    async def reply_to_comment(
        self, post_url: str, target_author: str, reply_text: str
    ) -> ActionResult:
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
            f"Tap the Reply button on that specific comment. "
            f"Type the following reply and submit it:\n\n"
            f"Reply: \"{reply_text}\"\n\n"
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
            await self.adb._run_checked([
                "shell", "am", "start", "-n",
                f"{GEMINI_APP}/com.google.android.apps.bard.ui.BardActivity",
            ])
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
