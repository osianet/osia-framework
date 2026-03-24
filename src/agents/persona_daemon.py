"""
Persona Daemon

A long-running service that periodically wakes a phone, opens a social media
app, scrolls the feed like a human, and interacts with posts (likes, comments,
shares) guided by the persona profile and DIRECTIVES.md analytical lens.

Supports multiple personas via a persona_id parameter — each persona gets its
own ADB device, name, bio, and env var namespace (PERSONA_<id>_*).

When the persona encounters a video, reel, or short, it extracts the URL and
downloads via yt-dlp (falling back to screen recording), uploads to Gemini for
multimodal analysis, then evaluates against DIRECTIVES.md to decide engagement.

The persona also creates original posts — opinions, observations, reactions to
current events (sourced from the RSS daily digest in Redis), and casual thoughts.
Posts are platform-appropriate and capped at a configurable daily limit.

Activity is tuned for high engagement to build a realistic online presence:
- 5-18 posts browsed per session, 8-35 min gaps between sessions
- ~50-60% like rate, ~25-30% comment rate on content seen
- ~20-30% chance of creating an original post each session
- Daily caps: 80 likes, 30 comments, 3 original posts (all configurable)
"""

import asyncio
import json
import logging
import os
import random
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from google import genai
from dotenv import load_dotenv
import redis.asyncio as aioredis
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
    posts: int = 0
    sessions: int = 0
    date: str = ""

    def reset_if_new_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.date != today:
            self.likes = 0
            self.comments = 0
            self.shares = 0
            self.posts = 0
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
        self._daily_like_cap = int(os.getenv(f"PERSONA_{persona_id}_LIKE_CAP", "80"))
        self._daily_comment_cap = int(os.getenv(f"PERSONA_{persona_id}_COMMENT_CAP", "30"))
        self._daily_post_cap = int(os.getenv(f"PERSONA_{persona_id}_POST_CAP", "3"))

        # Session timing
        self._min_gap = int(os.getenv(f"PERSONA_{persona_id}_MIN_GAP", "8"))
        self._max_gap = int(os.getenv(f"PERSONA_{persona_id}_MAX_GAP", "35"))
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

        # Redis — for pulling RSS digest as post inspiration
        self.redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

        self.stats = SessionStats()
        self._directives = self._load_directives()
        self._persona_prompt = self._build_persona()

        logger.info(
            "Persona '%s' (id=%s) initialized — device=%s, tz=%+d, caps=%d likes/%d comments/%d posts",
            self.persona_name, persona_id, device_id or "any",
            self._tz_offset, self._daily_like_cap, self._daily_comment_cap, self._daily_post_cap,
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

    def _parse_json_response(self, text: str) -> dict | None:
        """Strip markdown fences and parse JSON from a Gemini response."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    async def _decide_interaction(self, screenshot_path: str) -> dict:
        """
        Ask Gemini to look at the current post on screen and decide what
        the persona would naturally do.

        Detects video content (reels, shorts, stories) and flags it for the
        capture-and-analyze pipeline instead of a simple screenshot decision.
        """
        screen_file = self.gemini.files.upload(file=screenshot_path)

        can_like = self.stats.likes < self._daily_like_cap
        can_comment = self.stats.comments < self._daily_comment_cap

        # Dynamic engagement hints based on remaining budget
        if can_comment and can_like:
            engagement_hint = (
                "You're feeling social today. Be VERY active — engage with most things:\n"
                "- About 50-60% of the time, like/react to the post\n"
                "- About 25-30% of the time, leave a comment\n"
                "- Only scroll past ~15-20% of posts — the boring or irrelevant ones\n"
                "- If something is genuinely funny, interesting, or outrageous, ALWAYS engage\n"
                "- Don't be shy — real people interact with their feeds a lot"
            )
        elif can_like:
            engagement_hint = (
                "You've commented enough for today, but still like things freely:\n"
                "- About 55-65% of the time, like/react\n"
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
- "watch_video" — if this is a video, reel, short, or story, flag it for deeper viewing

IMPORTANT: If the post contains a video, reel, short, or story (look for play buttons,
video progress bars, reel/shorts UI, or video thumbnails), use "watch_video" so we can
actually watch and understand the content before deciding how to engage.

Respond with ONLY valid JSON (no markdown, no code fences):
{{
    "post_summary": "Brief description of what the post is about",
    "is_video": true/false,
    "action": "scroll|like|comment|share|watch_video",
    "comment_text": "Only if action is comment — what {self.persona_name} would say",
    "reasoning": "Brief reason for the choice"
}}
"""

        response = self.gemini.models.generate_content(
            model=self.model_id,
            contents=[screen_file, prompt],
        )

        parsed = self._parse_json_response(response.text)
        if parsed:
            return parsed
        logger.warning("[%s] Could not parse interaction decision: %s", self.persona_name, response.text[:200])
        return {"action": "scroll", "reasoning": "Parse error, defaulting to scroll"}

    # ------------------------------------------------------------------
    # Video comprehension pipeline
    # ------------------------------------------------------------------

    async def _extract_post_url(self) -> str | None:
        """
        Use the vision agent to tap the share button on the current post and
        copy the link. Returns the URL string or None if extraction fails.
        """
        result = await self.agent.execute_custom(
            "",
            "Find and tap the Share button on the post currently visible on screen. "
            "When the share sheet appears, look for 'Copy link' or 'Copy URL' and tap it. "
            "If there's no share sheet but a link/URL is visible, read it. "
            "Once you have the URL, put it in the 'data' field and use 'done'. "
            "If you can't get a URL, use 'done' with whatever info you have.",
        )
        if result.success and result.data:
            # Try to extract a URL from whatever the agent returned
            match = re.search(r"https?://[^\s\"'<>]+", result.data)
            if match:
                url = match.group(0)
                logger.info("[%s] Extracted post URL: %s", self.persona_name, url)
                # Press back to dismiss share sheet and return to feed
                await self.adb.press_back()
                await asyncio.sleep(0.5)
                return url
        # Dismiss any open dialogs
        await self.adb.press_back()
        await asyncio.sleep(0.5)
        return None

    async def _download_video_ytdlp(self, url: str) -> str | None:
        """
        Use yt-dlp to download the video directly. Works for YouTube, Instagram
        reels, Facebook videos, and TikTok. Returns local file path or None.
        """
        local_path = str(self.agent._scratch_dir / f"dl_{self.persona_id}.mp4")
        yt_dlp_bin = self.base_dir / ".venv" / "bin" / "yt-dlp"
        if not yt_dlp_bin.exists():
            yt_dlp_bin = "yt-dlp"  # fall back to system PATH

        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        cmd = [
            str(yt_dlp_bin),
            "--no-playlist",
            "--max-filesize", "50m",
            "-f", "best[filesize<50M]/best",
            "--user-agent", user_agent,
            "--geo-bypass",
            "-o", local_path,
            url,
        ]
        # Use cookies if available (helps with age-gated / login-walled content)
        cookies_path = self.base_dir / "config" / "youtube_cookies.txt"
        if cookies_path.exists():
            cmd.insert(-1, "--cookies")
            cmd.insert(-1, str(cookies_path))

        try:
            logger.info("[%s] Attempting yt-dlp download: %s", self.persona_name, url[:80])
            proc = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0 and Path(local_path).exists():
                logger.info("[%s] yt-dlp download succeeded (%s)", self.persona_name, local_path)
                return local_path
            else:
                logger.info("[%s] yt-dlp failed (rc=%d): %s", self.persona_name, proc.returncode, proc.stderr[:200])
        except subprocess.TimeoutExpired:
            logger.warning("[%s] yt-dlp timed out for %s", self.persona_name, url[:80])
        except Exception as e:
            logger.warning("[%s] yt-dlp error: %s", self.persona_name, e)
        return None

    async def _record_screen_fallback(self, duration: int = 15) -> str | None:
        """
        Fallback: record the phone screen while a video plays via ADB screenrecord.
        """
        remote_path = "/sdcard/osia_persona_capture.mp4"
        local_path = str(self.agent._scratch_dir / f"capture_{self.persona_id}.mp4")

        try:
            # Tap center to ensure video is playing
            w, h = await self.adb.get_screen_size()
            await self.adb.tap(w // 2, h // 2)
            await asyncio.sleep(1)

            logger.info("[%s] Screen recording fallback for %ds...", self.persona_name, duration)
            cmd = self.adb._build_cmd([
                "shell", "screenrecord", "--time-limit", str(duration), remote_path,
            ])
            await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
            await self.adb.pull_file(remote_path, local_path)
            await self.adb._run(["shell", "rm", "-f", remote_path])
            return local_path
        except Exception as e:
            logger.warning("[%s] Screen recording failed: %s", self.persona_name, e)
            return None

    async def _capture_video(self, duration: int = 15) -> str | None:
        """
        Tiered video capture:
        1. Extract the post URL via the share button
        2. Try yt-dlp to download the actual video (fast, full quality)
        3. Fall back to ADB screen recording if yt-dlp can't handle it
        """
        # Tier 1: try to get the URL and download directly
        url = await self._extract_post_url()
        if url:
            video_path = await self._download_video_ytdlp(url)
            if video_path:
                return video_path
            logger.info("[%s] yt-dlp couldn't grab it, falling back to screen recording", self.persona_name)

        # Tier 2: screen recording
        return await self._record_screen_fallback(duration=duration)

    async def _analyze_video_content(self, video_path: str) -> dict:
        """
        Upload a captured video to Gemini and get a comprehensive content analysis
        including transcription, visual description, and thematic summary.
        """
        logger.info("[%s] Uploading video for analysis...", self.persona_name)
        video_file = self.gemini.files.upload(file=video_path)

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            video_file = self.gemini.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            logger.warning("[%s] Video processing failed in Gemini", self.persona_name)
            return {"error": "Video processing failed"}

        prompt = """Analyze this video comprehensively. Provide:

1. TRANSCRIPTION: Transcribe any spoken words, narration, or dialogue
2. VISUAL: Describe what's shown — people, places, actions, text overlays, graphics
3. TOPIC: What is this video about? What's the core message or narrative?
4. TONE: Is it serious, funny, outrageous, educational, promotional, political, etc.?
5. THEMES: List the key themes (e.g. workers' rights, environment, tech, humor, propaganda, corporate, military, indigenous rights, housing, etc.)

Respond with ONLY valid JSON (no markdown, no code fences):
{
    "transcription": "What was said in the video",
    "visual_description": "What was shown",
    "topic": "Core topic/message in 1-2 sentences",
    "tone": "serious|funny|outrageous|educational|promotional|political|emotional|neutral",
    "themes": ["theme1", "theme2"],
    "notable_claims": "Any specific claims, statistics, or assertions made"
}"""

        response = self.gemini.models.generate_content(
            model=self.model_id,
            contents=[video_file, prompt],
        )

        parsed = self._parse_json_response(response.text)
        if parsed:
            return parsed
        logger.warning("[%s] Could not parse video analysis: %s", self.persona_name, response.text[:200])
        return {"topic": response.text[:500], "themes": [], "tone": "unknown"}

    async def _evaluate_and_engage_video(self, analysis: dict, app_name: str) -> None:
        """
        Given a video content analysis, evaluate it against DIRECTIVES.md and
        decide whether to like, comment, share, or just move on.

        Comments are only posted when the persona genuinely has something to say.
        Shares/reshares happen when content aligns with core values.
        """
        name = self.persona_name
        can_like = self.stats.likes < self._daily_like_cap
        can_comment = self.stats.comments < self._daily_comment_cap

        topic = analysis.get("topic", "unknown content")
        themes = ", ".join(analysis.get("themes", []))
        tone = analysis.get("tone", "unknown")
        transcription = analysis.get("transcription", "")
        notable = analysis.get("notable_claims", "")

        prompt = f"""{self._persona_prompt}

---

You just watched a video on {app_name}. Here's what was in it:

TOPIC: {topic}
TONE: {tone}
THEMES: {themes}
WHAT WAS SAID: {transcription[:800]}
NOTABLE CLAIMS: {notable[:300]}

Based on who you are and the principles you follow, decide how to engage.

Rules:
- Only comment if you genuinely have something good, funny, or insightful to say
- Keep comments SHORT — 1-2 sentences max, sound like a real person
- Like the video if it resonates with you at all — low bar
- Share/reshare ONLY if the content strongly aligns with your core values (anti-imperialism, workers' rights, indigenous sovereignty, environmental justice, tech ethics, data sovereignty)
- If the content is corporate propaganda, military glorification, or extractivist cheerleading, just scroll past — don't engage negatively
- If it's just neutral/boring content, scroll past
- {"You can like, comment, and share." if can_like and can_comment else "You can only like." if can_like else "You've hit your limits — just scroll."}

Respond with ONLY valid JSON (no markdown, no code fences):
{{
    "action": "scroll|like|comment|like_and_comment|share|like_and_share",
    "comment_text": "Only if commenting — what {name} would say (1-2 sentences, casual)",
    "reasoning": "Brief reason",
    "values_alignment": "none|low|medium|high"
}}"""

        response = self.gemini.models.generate_content(
            model=self.model_id,
            contents=[prompt],
        )

        parsed = self._parse_json_response(response.text)
        if not parsed:
            logger.warning("[%s] Could not parse video engagement decision", name)
            return

        action = parsed.get("action", "scroll")
        alignment = parsed.get("values_alignment", "none")
        logger.info(
            "[%s] Video verdict: %s (alignment=%s) — %s",
            name, action, alignment, parsed.get("reasoning", "")[:80],
        )

        # Execute the engagement
        should_like = action in ("like", "like_and_comment", "like_and_share")
        should_comment = action in ("comment", "like_and_comment")
        should_share = action in ("share", "like_and_share")

        if should_like and can_like:
            logger.info("[%s] Liking video: %s", name, topic[:60])
            result = await self.agent.execute_custom(
                "",
                "Find and tap the Like/Heart/Thumbs-up button on the post currently visible. "
                "If already liked, just use 'done'. Once liked, use 'done'.",
            )
            if result.success:
                self.stats.likes += 1

        if should_comment and can_comment:
            comment = parsed.get("comment_text", "")
            if comment:
                logger.info("[%s] Commenting on video: %s", name, comment[:80])
                result = await self.agent.execute_custom(
                    "",
                    f"Tap the comment button/icon on the post currently visible. "
                    f"Tap the comment input field, type this comment, then tap Post/Send:\n\n"
                    f"\"{comment}\"\n\n"
                    f"Once posted, press back to return to the feed and use 'done'.",
                )
                if result.success:
                    self.stats.comments += 1

        if should_share:
            logger.info("[%s] Sharing values-aligned video: %s", name, topic[:60])
            result = await self.agent.execute_custom(
                "",
                "Find and tap the Share/Repost button on the post currently visible. "
                "If a share dialog appears, tap 'Share to Feed' or 'Repost' or the equivalent. "
                "Once shared, use 'done'. If sharing isn't possible, use 'done' anyway.",
            )
            if result.success:
                self.stats.shares += 1

    # ------------------------------------------------------------------
    # Original content creation
    # ------------------------------------------------------------------

    # Post types the persona can create, with platform suitability
    _POST_TYPES = [
        {"type": "opinion", "desc": "a short opinion or hot take on something in the news", "platforms": ["Facebook", "Instagram", "Upscrolled"]},
        {"type": "observation", "desc": "a casual observation about daily life, tech, or something funny", "platforms": ["Facebook", "Instagram", "YouTube", "Upscrolled"]},
        {"type": "share_article", "desc": "sharing a link or summarizing an interesting article you read", "platforms": ["Facebook", "Upscrolled"]},
        {"type": "question", "desc": "asking your followers a genuine question to spark discussion", "platforms": ["Facebook", "Instagram", "Upscrolled"]},
        {"type": "reaction", "desc": "reacting to a trending topic or current event", "platforms": ["Facebook", "Instagram", "Upscrolled"]},
    ]

    async def _get_post_inspiration(self) -> str:
        """
        Pull recent RSS digest items from Redis to give the persona something
        topical to post about. Returns a summary string or empty if nothing available.
        """
        try:
            # Peek at the daily digest without draining it (LRANGE, not LPOP)
            items = await self.redis.lrange("osia:rss:daily_digest", 0, 9)
            if items:
                summaries = [item.decode() if isinstance(item, bytes) else item for item in items[:5]]
                return "Recent news headlines you've been reading:\n" + "\n".join(
                    f"- {s[:200]}" for s in summaries
                )
        except Exception as e:
            logger.debug("[%s] Could not fetch RSS digest for post inspiration: %s", self.persona_name, e)
        return ""

    async def _generate_post_content(self, app_name: str) -> dict | None:
        """
        Ask Gemini to generate an original post as the persona.
        Returns dict with 'text' and optionally 'type', or None if generation fails.
        """
        # Pick a post type suitable for this platform
        suitable = [p for p in self._POST_TYPES if app_name in p["platforms"]]
        if not suitable:
            suitable = self._POST_TYPES
        post_type = random.choice(suitable)

        inspiration = await self._get_post_inspiration()

        local_time = datetime.now(timezone.utc) + timedelta(hours=self._tz_offset)
        time_context = local_time.strftime("%A, %I:%M %p")

        prompt = f"""{self._persona_prompt}

---

It's {time_context} and you're on {app_name}. You want to make a post.

Post type: {post_type['desc']}

{inspiration}

Write a post that {self.persona_name} would naturally make. Rules:
- Sound like a REAL person, not a brand or AI
- Keep it short — 1-3 sentences for most posts, max 4-5 if you really have something to say
- Match the platform style ({app_name})
- Be genuine — dry humor, real opinions, casual language
- No hashtags on Facebook. On Instagram, 0-2 relevant hashtags MAX at the end
- Don't be preachy or lecture-y. If it's political, make it a pointed observation not a speech
- Sometimes be funny, sometimes be thoughtful, sometimes just share something interesting
- NEVER sound like you're trying to go viral or get engagement

Respond with ONLY valid JSON (no markdown, no code fences):
{{
    "text": "The actual post text",
    "type": "{post_type['type']}",
    "reasoning": "Why this feels natural right now"
}}"""

        response = self.gemini.models.generate_content(
            model=self.model_id,
            contents=[prompt],
        )

        parsed = self._parse_json_response(response.text)
        if parsed and parsed.get("text"):
            return parsed
        logger.warning("[%s] Could not generate post content", self.persona_name)
        return None

    async def _create_post_on_app(self, app_name: str, post_text: str) -> bool:
        """
        Use the vision agent to create a new post on the currently open app.
        Returns True if the post was successfully created.
        """
        # Platform-specific instructions for creating a post
        if app_name == "Facebook":
            instruction = (
                "You are on the Facebook app home feed. Find and tap the 'What's on your mind?' "
                "text box at the top of the feed (or the create post button). "
                "Once the post composer opens, tap the text input area and type this post:\n\n"
                f"\"{post_text}\"\n\n"
                "Then tap the 'Post' button to publish it. Once posted, use 'done'."
            )
        elif app_name == "Instagram":
            instruction = (
                "You are on Instagram. Tap the '+' (create/new post) button, usually at the bottom "
                "center or top right of the screen. If it asks what type of content, select 'Post'. "
                "If it asks to select a photo, just pick the most recent photo in the gallery "
                "(or take a quick photo if needed — the text is what matters). "
                "On the caption screen, tap the caption field and type:\n\n"
                f"\"{post_text}\"\n\n"
                "Then tap 'Share' or 'Post' to publish. Once posted, use 'done'. "
                "If you can't create a text-only post, use 'fail'."
            )
        elif app_name == "YouTube":
            instruction = (
                "You are on YouTube. Tap the '+' create button at the bottom center. "
                "Select 'Create a post' (community post). "
                "Tap the text area and type:\n\n"
                f"\"{post_text}\"\n\n"
                "Then tap 'Post' to publish. Once posted, use 'done'. "
                "If community posts aren't available, use 'fail'."
            )
        else:
            instruction = (
                f"You are on {app_name}. Find the create post / new post button. "
                "Tap it to open the post composer. Type this post:\n\n"
                f"\"{post_text}\"\n\n"
                "Then tap Post/Share/Submit to publish. Once posted, use 'done'."
            )

        result = await self.agent.execute_custom("", instruction)
        return result.success

    async def _maybe_create_post(self, app_name: str) -> bool:
        """
        Decide whether to create an original post this session.
        ~25% chance per session, respects daily post cap.
        Returns True if a post was created.
        """
        if self.stats.posts >= self._daily_post_cap:
            return False

        # 25% chance to post, slightly higher in the evening (people post more then)
        local_hour = (datetime.now(timezone.utc) + timedelta(hours=self._tz_offset)).hour
        post_chance = 0.30 if 17 <= local_hour <= 22 else 0.20
        if random.random() > post_chance:
            return False

        logger.info("[%s] Generating original post for %s...", self.persona_name, app_name)
        content = await self._generate_post_content(app_name)
        if not content:
            return False

        post_text = content["text"]
        logger.info(
            "[%s] Posting (%s): %s",
            self.persona_name, content.get("type", "?"), post_text[:80],
        )

        success = await self._create_post_on_app(app_name, post_text)
        if success:
            self.stats.posts += 1
            logger.info("[%s] Post created successfully (today: %d/%d)", self.persona_name, self.stats.posts, self._daily_post_cap)
        else:
            logger.warning("[%s] Failed to create post on %s", self.persona_name, app_name)
        return success

    async def _execute_interaction(self, decision: dict, app_name: str = ""):
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
            return

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

        elif action in ("watch_video", "watch"):
            logger.info("[%s] Video detected: %s", name, decision.get("post_summary", "")[:60])
            # Capture the video playing on screen
            capture_duration = random.randint(12, 20)
            video_path = await self._capture_video(duration=capture_duration)

            if video_path:
                # Analyze the video content with Gemini multimodal
                analysis = await self._analyze_video_content(video_path)
                logger.info(
                    "[%s] Video analysis — topic: %s, themes: %s",
                    name,
                    analysis.get("topic", "?")[:60],
                    ", ".join(analysis.get("themes", []))[:60],
                )
                # Evaluate against directives and engage if warranted
                await self._evaluate_and_engage_video(analysis, app_name)
                # Clean up capture file
                try:
                    Path(video_path).unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                # Fallback: couldn't capture, just watch briefly and maybe like
                watch_time = random.uniform(8, 20)
                logger.info("[%s] Capture failed, watching for %.0fs", name, watch_time)
                await asyncio.sleep(watch_time)
                if random.random() < 0.4 and self.stats.likes < self._daily_like_cap:
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
        num_posts = random.randint(5, 18)

        logger.info(
            "[%s] Session #%d — Opening %s, browsing ~%d posts "
            "(today: %d likes, %d comments, %d shares, %d posts)",
            self.persona_name, self.stats.sessions, app["name"], num_posts,
            self.stats.likes, self.stats.comments, self.stats.shares, self.stats.posts,
        )

        # Open the app
        await self.adb.wake_and_unlock()
        await self.adb._run_checked([
            "shell", "monkey", "-p", app["package"],
            "-c", "android.intent.category.LAUNCHER", "1",
        ])
        await asyncio.sleep(random.uniform(3, 6))

        # Maybe create an original post before browsing
        await self._maybe_create_post(app["name"])

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

                await self._execute_interaction(decision, app_name=app["name"])

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
            "[%s] Session complete — today: %d likes, %d comments, %d shares, %d posts",
            self.persona_name, self.stats.likes, self.stats.comments, self.stats.shares, self.stats.posts,
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
