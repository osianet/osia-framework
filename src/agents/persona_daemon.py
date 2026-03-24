"""
Persona 1 Daemon

A long-running service that periodically wakes the phone, opens a social media
app, scrolls the feed like a human, and occasionally interacts with posts
(likes, comments) guided by the DIRECTIVES.md analytical lens.

The goal is to build a credible, organic-looking social media presence over time.
Activity is randomized to mimic natural human behavior patterns:
- Heavier usage during "waking hours" (8am-11pm local time)
- Variable session lengths (2-8 scroll actions per session)
- Most sessions are passive (just scrolling/liking)
- Comments are infrequent and contextually appropriate
- Cooldowns between sessions (20-90 minutes)
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from src.gateways.adb_device import ADBDevice
from src.agents.social_media_agent import SocialMediaAgent

logger = logging.getLogger("osia.persona")

# Apps David uses, weighted by how often a normal person opens them
APPS = [
    {"name": "Facebook", "package": "com.facebook.katana", "weight": 30},
    {"name": "Instagram", "package": "com.instagram.android", "weight": 35},
    {"name": "YouTube", "package": "com.google.android.youtube", "weight": 25},
    {"name": "Upscrolled", "package": "com.upscrolled.app", "weight": 10},
]

# Session timing
MIN_SESSION_GAP_MINUTES = 20
MAX_SESSION_GAP_MINUTES = 90
QUIET_HOURS_START = 23  # 11pm — David goes to bed
QUIET_HOURS_END = 8     # 8am — David wakes up
QUIET_HOUR_GAP_MINUTES = (120, 300)  # much less active at night


@dataclass
class SessionStats:
    """Tracks activity to enforce daily limits."""
    likes: int = 0
    comments: int = 0
    sessions: int = 0
    date: str = ""

    def reset_if_new_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.date != today:
            self.likes = 0
            self.comments = 0
            self.sessions = 0
            self.date = today

    @property
    def can_like(self) -> bool:
        return self.likes < 30  # daily cap

    @property
    def can_comment(self) -> bool:
        return self.comments < 5  # very conservative daily cap


class PersonaDaemon:
    """
    Runs a social media persona as a background service.

    Each session:
    1. Pick a random app (weighted)
    2. Open it and let the feed load
    3. Scroll through 2-8 posts
    4. For each post, Gemini Vision decides: scroll past / like / comment
    5. Close the app and sleep until next session
    """

    def __init__(self):
        load_dotenv()
        self.base_dir = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))
        self.adb = ADBDevice(device_id=os.getenv("ADB_DEVICE_PERSONA_1"))
        self.gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
        self.agent = SocialMediaAgent(
            adb=self.adb,
            gemini_client=self.gemini,
            model_id=self.model_id,
            base_dir=self.base_dir,
        )
        self.stats = SessionStats()
        self._directives = self._load_directives()
        self._persona = self._build_persona()
        self._tz_offset = int(os.getenv("PERSONA_TZ_OFFSET", "10"))  # AEST default

    def _load_directives(self) -> str:
        path = self.base_dir / "DIRECTIVES.md"
        if path.exists():
            return path.read_text()
        logger.warning("DIRECTIVES.md not found at %s", path)
        return ""

    def _build_persona(self) -> str:
        persona_name = os.getenv("PERSONA_1_NAME", "A generic persona")
        return f"""You are {persona_name}, a 34-year-old Australian bloke from Perth.
You work in IT infrastructure and have a dry, sardonic sense of humor.
You're politically engaged but not preachy — you care about workers' rights,
housing affordability, and tech ethics. You follow news, science, and memes.

Your social media style:
- Casual, Australian English (occasional slang like "reckon", "mate", "bloody")
- Short comments, rarely more than 1-2 sentences
- You like posts about: tech, science, geopolitics, workers' rights, funny stuff, dogs
- You sometimes make dry observations or jokes
- You never use hashtags excessively (1 max, usually none)
- You don't argue with strangers but might make a pointed observation
- You share genuine reactions, not performative ones
- You occasionally use emoji but sparingly (👍, 😂, 🤔)
- You NEVER sound like a bot, corporate account, or AI

When deciding whether to engage with a post, consider the OSIA directives:
""" + self._directives

    def _is_quiet_hours(self) -> bool:
        """Check if it's David's sleeping hours."""
        local_hour = (datetime.now(timezone.utc) + timedelta(hours=self._tz_offset)).hour
        if QUIET_HOURS_START > QUIET_HOURS_END:
            return local_hour >= QUIET_HOURS_START or local_hour < QUIET_HOURS_END
        return QUIET_HOURS_START <= local_hour < QUIET_HOURS_END

    def _pick_app(self) -> dict:
        """Weighted random app selection."""
        weights = [a["weight"] for a in APPS]
        return random.choices(APPS, weights=weights, k=1)[0]

    def _next_session_delay(self) -> float:
        """Calculate seconds until next session."""
        if self._is_quiet_hours():
            minutes = random.uniform(*QUIET_HOUR_GAP_MINUTES)
            logger.info("Quiet hours — next session in %.0f minutes", minutes)
        else:
            minutes = random.uniform(MIN_SESSION_GAP_MINUTES, MAX_SESSION_GAP_MINUTES)
        return minutes * 60

    async def _decide_interaction(self, screenshot_path: str) -> dict:
        """
        Ask Gemini to look at the current post on screen and decide what
        David would do as a normal person.
        """
        screen_file = self.gemini.files.upload(file=screenshot_path)

        prompt = f"""{self._persona}

---

You are scrolling through your social media feed on your phone.
Look at this screenshot and decide what {persona_name} would naturally do.

Consider:
- Is this post interesting enough to engage with?
- Would a normal person interact with this, or just scroll past?
- Most of the time (60-70%), you just scroll past.
- Sometimes (20-30%) you like/react to something.
- Rarely (5-10%) you leave a short comment.

Respond with ONLY valid JSON (no markdown, no code fences):
{{
    "post_summary": "Brief description of what the post is about",
    "action": "scroll|like|comment",
    "comment_text": "Only if action is comment — what David would say",
    "reasoning": "Why David would or wouldn't engage"
}}
"""

        response = self.gemini.models.generate_content(
            model=self.model_id,
            contents=[screen_file, prompt],
        )

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse interaction decision: %s", text[:200])
            return {"action": "scroll", "reasoning": "Parse error, defaulting to scroll"}

    async def _execute_interaction(self, decision: dict):
        """Execute the decided interaction on the current screen."""
        action = decision.get("action", "scroll")

        if action == "like" and self.stats.can_like:
            logger.info("Liking post: %s", decision.get("post_summary", "")[:60])
            result = await self.agent.execute_custom(
                "",
                "Find and tap the Like/Heart/Thumbs-up button on the post currently visible on screen. "
                "If already liked, just use 'done'. Once liked, use 'done'.",
            )
            if result.success:
                self.stats.likes += 1

        elif action == "comment" and self.stats.can_comment:
            comment = decision.get("comment_text", "")
            if comment:
                logger.info("Commenting: %s", comment[:60])
                result = await self.agent.execute_custom(
                    "",
                    f"Tap the comment button/icon on the post currently visible. "
                    f"Tap the comment input field, type this comment, then tap Post/Send:\n\n"
                    f"\"{comment}\"\n\n"
                    f"Once posted, press back to return to the feed and use 'done'.",
                )
                if result.success:
                    self.stats.comments += 1
            else:
                logger.info("Comment action but no text generated, scrolling instead.")

        # For "scroll" or fallthrough, just swipe to next post
        if action == "scroll" or (action == "like" and not self.stats.can_like) or (action == "comment" and not self.stats.can_comment):
            w, h = await self.adb.get_screen_size()
            # Natural-looking scroll with slight horizontal variance
            x = w // 2 + random.randint(-30, 30)
            await self.adb.swipe(x, h * 3 // 4, x, h // 4, duration_ms=random.randint(300, 600))

    async def run_session(self):
        """Run a single browsing session."""
        self.stats.reset_if_new_day()
        self.stats.sessions += 1

        app = self._pick_app()
        num_posts = random.randint(2, 8)

        logger.info(
            "Session #%d — Opening %s, browsing ~%d posts (today: %d likes, %d comments)",
            self.stats.sessions, app["name"], num_posts,
            self.stats.likes, self.stats.comments,
        )

        # Open the app
        await self.adb.wake_and_unlock()
        await self.adb._run_checked([
            "shell", "monkey", "-p", app["package"],
            "-c", "android.intent.category.LAUNCHER", "1",
        ])
        await asyncio.sleep(random.uniform(3, 6))  # wait for app to load

        for i in range(num_posts):
            try:
                # Small pause between posts — like a human reading
                await asyncio.sleep(random.uniform(2, 5))

                screenshot_path = await self.agent._screenshot()
                decision = await self._decide_interaction(screenshot_path)

                logger.info(
                    "Post %d/%d — %s → %s",
                    i + 1, num_posts,
                    decision.get("post_summary", "?")[:50],
                    decision.get("action", "scroll"),
                )

                await self._execute_interaction(decision)

                # Random longer pause occasionally (like reading a long post)
                if random.random() < 0.2:
                    await asyncio.sleep(random.uniform(5, 15))

            except Exception as e:
                logger.warning("Error during post interaction: %s", e)
                # Try to recover by scrolling
                try:
                    w, h = await self.adb.get_screen_size()
                    await self.adb.swipe(w // 2, h * 3 // 4, w // 2, h // 4)
                except Exception:
                    pass

        # Close the app — press home
        await self.adb._run_checked(["shell", "input", "keyevent", "3"])
        logger.info("Session complete. Likes today: %d, Comments today: %d", self.stats.likes, self.stats.comments)

    async def run_forever(self):
        """Main daemon loop — run sessions with randomized gaps."""
        logger.info(f"Persona daemon starting for: {os.getenv('PERSONA_1_NAME', 'Persona 1')}")

        while True:
            try:
                await self.run_session()
            except Exception as e:
                logger.error("Session failed: %s", e)

            delay = self._next_session_delay()
            logger.info("Next session in %.0f minutes.", delay / 60)
            await asyncio.sleep(delay)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    daemon = PersonaDaemon()
    asyncio.run(daemon.run_forever())
