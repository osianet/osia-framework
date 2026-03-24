"""
Persona Daemon

A long-running service that periodically wakes a phone, opens a social media
app, scrolls the feed like a human, and interacts with posts (likes, comments,
shares) guided by the persona profile and DIRECTIVES.md analytical lens.

Supports multiple personas via a persona_id parameter — each persona gets its
own ADB device, name, bio, and env var namespace (PERSONA_<id>_*).

Activity is randomized to mimic natural human behavior:
- Heavier usage during "waking hours" (configurable per persona)
- Variable session lengths (3-12 scroll actions per session)
- Active engagement: ~35-45% like, ~15-20% comment, rest scroll
- Cooldowns between sessions (15-60 minutes during active hours)
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

# Apps and their weights — can be overridden per persona via env
DEFAULT_APPS = [
    {"name": "Facebook", "package": "com.facebook.katana", "weight": 30},
    {"name": "Instagram", "package": "com.instagram.android", "weight": 35},
    {"name": "YouTube", "package": "com.google.android.youtube", "weight": 25},
    {"name": "Upscrolled", "package": "com.upscrolled.app", "weight": 10},
]


@dataclass
class SessionStats:
    """Tracks activity to enforce daily limits."""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    sessions: int = 0
    date: str = ""

    def reset_if_new_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.date != today:
            self.likes = 0
            self.comments = 0
            self.shares = 0
            self.sessions = 0
            self.date = today


class PersonaDaemon:
    """
    Runs a social media persona as a background service.

    Args:
        persona_id: Identifier used to look up env vars (e.g. "1" reads PERSONA_1_NAME,
                     ADB_DEVICE_PERSONA_1, etc.). Defaults to "1".
    """

    def __init__(self, persona_id: str = "1"):
        load_dotenv()
        self.persona_id = persona_id
        self.base_dir = Path(os.getenv("OSIA_BASE_DIR", Path(__file__).resolve().parent.parent.parent))

        # Persona-specific config from env
        self.persona_name = os.getenv(f"PERSONA_{persona_id}_NAME", f"Persona {persona_id}")
        device_id = os.getenv(f"ADB_DEVICE_PERSONA_{persona_id}") or None
        self._tz_offset = int(os.getenv(f"PERSONA_{persona_id}_TZ_OFFSET",
                                         os.getenv("PERSONA_TZ_OFFSET", "10")))

        # Daily limits — configurable per persona
        self._daily_like_cap = int(os.getenv(f"PERSONA_{persona_id}_LIKE_CAP", "50"))
        self._daily_comment_cap = int(os.getenv(f"PERSONA_{persona_id}_COMMENT_CAP", "15"))

        # Session timing
        self._min_gap = int(os.getenv(f"PERSONA_{persona_id}_MIN_GAP", "15"))
        self._max_gap = int(os.getenv(f"PERSONA_{persona_id}_MAX_GAP", "60"))
        self._quiet_start = int(os.getenv(f"PERSONA_{persona_id}_QUIET_START", "23"))
        self._quiet_end = int(os.getenv(f"PERSONA_{persona_id}_QUIET_END", "8"))

        # Persona bio — can be overridden entirely via env or file
        self._persona_bio = os.getenv(f"PERSONA_{persona_id}_BIO", "")

        # ADB + Gemini
        self.adb = ADBDevice(device_id=device_id)
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
        self._persona_prompt = self._build_persona()

        logger.info(
            "Persona '%s' (id=%s) initialized — device=%s, tz=%+d, caps=%d likes/%d comments",
            self.persona_name, persona_id, device_id or "any",
            self._tz_offset, self._daily_like_cap, self._daily_comment_cap,
        )

    def _load_directives(self) -> str:
        path = self.base_dir / "DIRECTIVES.md"
        if path.exists():
            return path.read_text()
        logger.warning("DIRECTIVES.md not found at %s", path)
        return ""

    def _load_persona_bio(self) -> str:
        """Load persona bio from env var, file, or fall back to default."""
        # Check for a persona-specific bio file first
        bio_file = self.base_dir / "config" / f"persona_{self.persona_id}_bio.txt"
        if bio_file.exists():
            return bio_file.read_text().strip()
        if self._persona_bio:
            return self._persona_bio
        # Default bio
        return (
            f"a 34-year-old Australian bloke from Perth. "
            f"Works in IT infrastructure and has a dry, sardonic sense of humor. "
            f"Politically engaged but not preachy — cares about workers' rights, "
            f"housing affordability, and tech ethics. Follows news, science, and memes."
        )

    def _build_persona(self) -> str:
        bio = self._load_persona_bio()
        return f"""You are {self.persona_name}, {bio}

Your social media style:
- Casual, natural language — sound like a real person, not a brand
- Short comments, usually 1-2 sentences, occasionally a bit longer if you have something to say
- You engage with posts about: tech, science, geopolitics, workers' rights, funny stuff, animals, music
- You make dry observations, jokes, or share genuine reactions
- You never use hashtags excessively (1 max, usually none)
- You don't start arguments but you'll make a pointed observation when something is wrong
- You occasionally use emoji but sparingly (👍, 😂, 🤔, 💯)
- You NEVER sound like a bot, corporate account, or AI
- You sometimes ask questions in comments ("anyone else reckon...?", "wait is this real?")
- You react to things you find genuinely interesting, funny, or outrageous

When deciding whether to engage with a post, consider these principles:
{self._directives}"""

    def _is_quiet_hours(self) -> bool:
        local_hour = (datetime.now(timezone.utc) + timedelta(hours=self._tz_offset)).hour
        if self._quiet_start > self._quiet_end:
            return local_hour >= self._quiet_start or local_hour < self._quiet_end
        return self._quiet_start <= local_hour < self._quiet_end

    def _pick_app(self) -> dict:
        weights = [a["weight"] for a in DEFAULT_APPS]
        return random.choices(DEFAULT_APPS, weights=weights, k=1)[0]

    def _next_session_delay(self) -> float:
        if self._is_quiet_hours():
            minutes = random.uniform(120, 300)
            logger.info("[%s] Quiet hours — next session in %.0f minutes", self.persona_name, minutes)
        else:
            minutes = random.uniform(self._min_gap, self._max_gap)
        return minutes * 60

    async def _decide_interaction(self, screenshot_path: str) -> dict:
        """
        Ask Gemini to look at the current post on screen and decide what
        the persona would naturally do.

        Tuned for more active engagement than a purely passive scroller.
        """
        screen_file = self.gemini.files.upload(file=screenshot_path)

        can_like = self.stats.likes < self._daily_like_cap
        can_comment = self.stats.comments < self._daily_comment_cap

        # Dynamic engagement hints based on remaining budget
        if can_comment and can_like:
            engagement_hint = (
                "You're in an engaging mood today. Be active:\n"
                "- About 35-45% of the time, like/react to the post\n"
                "- About 15-20% of the time, leave a comment\n"
                "- The rest of the time, just scroll past\n"
                "- If something is genuinely funny, interesting, or outrageous, ALWAYS engage"
            )
        elif can_like:
            engagement_hint = (
                "You've commented enough for today, but still like things freely:\n"
                "- About 40-50% of the time, like/react\n"
                "- Don't comment, just scroll or like"
            )
        else:
            engagement_hint = "You've been pretty active today. Just scroll and read for now."

        prompt = f"""{self._persona_prompt}

---

You are scrolling through your social media feed on your phone.
Look at this screenshot and decide what {self.persona_name} would naturally do.

{engagement_hint}

Also consider these interaction types:
- "like" — tap the like/heart/thumbs-up button
- "comment" — leave a short, natural comment
- "share" — share/repost something particularly noteworthy (rare, ~5%)
- "scroll" — just move on to the next post
- "watch" — if it's a video, watch it for a bit before deciding (tap play if needed)

Respond with ONLY valid JSON (no markdown, no code fences):
{{
    "post_summary": "Brief description of what the post is about",
    "action": "scroll|like|comment|share|watch",
    "comment_text": "Only if action is comment — what {self.persona_name} would say",
    "reasoning": "Brief reason for the choice"
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
            logger.warning("[%s] Could not parse interaction decision: %s", self.persona_name, text[:200])
            return {"action": "scroll", "reasoning": "Parse error, defaulting to scroll"}

    async def _execute_interaction(self, decision: dict):
        """Execute the decided interaction on the current screen."""
        action = decision.get("action", "scroll")
        name = self.persona_name

        if action == "like" and self.stats.likes < self._daily_like_cap:
            logger.info("[%s] Liking: %s", name, decision.get("post_summary", "")[:60])
            result = await self.agent.execute_custom(
                "",
                "Find and tap the Like/Heart/Thumbs-up button on the post currently visible on screen. "
                "If already liked, just use 'done'. Once liked, use 'done'.",
            )
            if result.success:
                self.stats.likes += 1
            return  # don't scroll after liking — stay on the post briefly

        elif action == "comment" and self.stats.comments < self._daily_comment_cap:
            comment = decision.get("comment_text", "")
            if comment:
                logger.info("[%s] Commenting: %s", name, comment[:80])
                result = await self.agent.execute_custom(
                    "",
                    f"Tap the comment button/icon on the post currently visible. "
                    f"Tap the comment input field, type this comment, then tap Post/Send:\n\n"
                    f"\"{comment}\"\n\n"
                    f"Once posted, press back to return to the feed and use 'done'.",
                )
                if result.success:
                    self.stats.comments += 1
                return

        elif action == "share":
            logger.info("[%s] Sharing: %s", name, decision.get("post_summary", "")[:60])
            result = await self.agent.execute_custom(
                "",
                "Find and tap the Share/Repost button on the post currently visible. "
                "If a share dialog appears, tap 'Share to Feed' or 'Repost' or the equivalent. "
                "Once shared, use 'done'. If sharing isn't possible, use 'done' anyway.",
            )
            if result.success:
                self.stats.shares += 1
            return

        elif action == "watch":
            logger.info("[%s] Watching video: %s", name, decision.get("post_summary", "")[:60])
            # Tap the center of the screen to play, then wait
            w, h = await self.adb.get_screen_size()
            await self.adb.tap(w // 2, h // 2)
            watch_time = random.uniform(8, 25)
            logger.info("[%s] Watching for %.0f seconds", name, watch_time)
            await asyncio.sleep(watch_time)
            # After watching, there's a good chance we like it
            if random.random() < 0.6 and self.stats.likes < self._daily_like_cap:
                logger.info("[%s] Liked after watching", name)
                await self.agent.execute_custom(
                    "",
                    "Find and tap the Like/Heart/Thumbs-up button on the post currently visible. "
                    "If already liked, just use 'done'. Once liked, use 'done'.",
                )
                self.stats.likes += 1
            return

        # Default: scroll to next post
        w, h = await self.adb.get_screen_size()
        x = w // 2 + random.randint(-30, 30)
        await self.adb.swipe(x, h * 3 // 4, x, h // 4, duration_ms=random.randint(300, 600))

    async def run_session(self):
        """Run a single browsing session."""
        self.stats.reset_if_new_day()
        self.stats.sessions += 1

        app = self._pick_app()
        num_posts = random.randint(3, 12)

        logger.info(
            "[%s] Session #%d — Opening %s, browsing ~%d posts "
            "(today: %d likes, %d comments, %d shares)",
            self.persona_name, self.stats.sessions, app["name"], num_posts,
            self.stats.likes, self.stats.comments, self.stats.shares,
        )

        # Open the app
        await self.adb.wake_and_unlock()
        await self.adb._run_checked([
            "shell", "monkey", "-p", app["package"],
            "-c", "android.intent.category.LAUNCHER", "1",
        ])
        await asyncio.sleep(random.uniform(3, 6))

        for i in range(num_posts):
            try:
                # Reading pause — varies like a real person
                await asyncio.sleep(random.uniform(1.5, 4))

                screenshot_path = await self.agent._screenshot()
                decision = await self._decide_interaction(screenshot_path)

                logger.info(
                    "[%s] Post %d/%d — %s → %s",
                    self.persona_name, i + 1, num_posts,
                    decision.get("post_summary", "?")[:50],
                    decision.get("action", "scroll"),
                )

                await self._execute_interaction(decision)

                # Occasional longer pause (reading comments, watching a video, etc.)
                if random.random() < 0.25:
                    await asyncio.sleep(random.uniform(4, 12))

            except Exception as e:
                logger.warning("[%s] Error during post interaction: %s", self.persona_name, e)
                try:
                    w, h = await self.adb.get_screen_size()
                    await self.adb.swipe(w // 2, h * 3 // 4, w // 2, h // 4)
                except Exception:
                    pass

        # Press home to close
        await self.adb._run_checked(["shell", "input", "keyevent", "3"])
        logger.info(
            "[%s] Session complete — today: %d likes, %d comments, %d shares",
            self.persona_name, self.stats.likes, self.stats.comments, self.stats.shares,
        )

    async def run_forever(self):
        """Main daemon loop."""
        logger.info("[%s] Persona daemon starting (id=%s)", self.persona_name, self.persona_id)

        while True:
            try:
                await self.run_session()
            except Exception as e:
                logger.error("[%s] Session failed: %s", self.persona_name, e)

            delay = self._next_session_delay()
            logger.info("[%s] Next session in %.0f minutes.", self.persona_name, delay / 60)
            await asyncio.sleep(delay)


def main():
    """Entry point — reads PERSONA_ID env var to support multiple instances."""
    import argparse
    parser = argparse.ArgumentParser(description="OSIA Persona Daemon")
    parser.add_argument("--persona", default=os.getenv("PERSONA_ID", "1"),
                        help="Persona ID (reads PERSONA_<id>_* env vars)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    daemon = PersonaDaemon(persona_id=args.persona)
    asyncio.run(daemon.run_forever())


if __name__ == "__main__":
    main()
