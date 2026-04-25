"""
Instagram account pool manager.

Redis-backed state machine: CREATED → WARMING → ACTIVE → FLAGGED → RETIRED

Cookie content is stored directly in Redis at osia:ig:cookies:<account_id>
(Netscape format string). No on-disk files are required; get_active_cookie_path()
materialises a temp file for yt-dlp right before use.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import redis.asyncio as redis

logger = logging.getLogger("osia.ig_pool")

SMSPOOL_BASE = "https://api.smspool.net"
IG_SERVICE = 457  # SMSPool service ID for "Instagram / Threads"

_ACTIVE_MIN = int(os.getenv("IG_POOL_ACTIVE_MIN", "3"))
_WARMING_MIN = int(os.getenv("IG_POOL_WARMING_MIN", "3"))
_WARMUP_DAYS = int(os.getenv("IG_WARMUP_DAYS", "7"))

# Redis keys
_ACCOUNTS_HASH = "osia:ig:accounts"
_ACTIVE_SET = "osia:ig:pool:active"
_WARMING_SET = "osia:ig:pool:warming"
_FLAGGED_SET = "osia:ig:pool:flagged"
_CURRENT_KEY = "osia:ig:current"
_COOKIE_KEY_PREFIX = "osia:ig:cookies:"  # + account_id → Netscape cookie string

_ALL_STATE_SETS = (_ACTIVE_SET, _WARMING_SET, _FLAGGED_SET)
_TEMP_DIR = Path(tempfile.gettempdir())


@dataclass
class InstagramAccount:
    id: str
    username: str
    password: str
    email: str
    phone: str
    phone_country: str
    vpn_country: str
    smspool_order_id: str
    state: str
    created_at: int
    warmed_since: int | None = None
    promoted_at: int | None = None
    flagged_at: int | None = None
    cookies_path: str = ""
    flag_reason: str | None = None
    warmup_sessions: int = 0
    last_warmed_at: int | None = None
    has_profile_pic: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "InstagramAccount":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


class SMSPoolClient:
    """Thin async wrapper around the SMSPool REST API."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def purchase_number(self, country: str, service: int = IG_SERVICE) -> dict:
        """Purchase a number for OTP. Returns {orderid, phonenumber, ...}."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{SMSPOOL_BASE}/purchase/sms",
                params={"key": self._api_key, "country": country, "service": service},
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("success"):
                raise RuntimeError(f"SMSPool purchase failed: {data}")
            return data

    async def poll_otp(self, order_id: str, timeout: int = 120, interval: int = 5) -> str | None:
        """Poll until OTP arrives or timeout. Returns OTP string or None."""
        deadline = time.time() + timeout
        async with httpx.AsyncClient(timeout=30) as client:
            while time.time() < deadline:
                r = await client.get(
                    f"{SMSPOOL_BASE}/sms/check",
                    params={"key": self._api_key, "orderid": order_id},
                )
                r.raise_for_status()
                data = r.json()
                if data.get("sms"):
                    return str(data["sms"])
                await asyncio.sleep(interval)
        return None

    async def cancel_order(self, order_id: str) -> None:
        """Recycle / cancel a number order (e.g. OTP never arrived)."""
        async with httpx.AsyncClient(timeout=30) as client:
            await client.get(
                f"{SMSPOOL_BASE}/sms/cancel",
                params={"key": self._api_key, "orderid": order_id},
            )


class InstagramAccountManager:
    """
    Redis-backed Instagram account pool.

    Maintains state sets (active/warming/flagged) and a per-account hash.
    Instantiate once and reuse — no teardown required.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        base_dir: Path | None = None,
        smspool_api_key: str | None = None,
    ):
        self._redis = redis_client
        self._base_dir = base_dir or Path(__file__).resolve().parent.parent.parent

        api_key = smspool_api_key or os.getenv("SMSPOOL_API_KEY", "")
        self.smspool: SMSPoolClient | None = SMSPoolClient(api_key) if api_key else None

    # ------------------------------------------------------------------ CRUD

    async def get(self, account_id: str) -> InstagramAccount | None:
        raw = await self._redis.hget(_ACCOUNTS_HASH, account_id)
        if not raw:
            return None
        try:
            return InstagramAccount.from_dict(json.loads(raw))
        except Exception:
            return None

    async def _save(self, account: InstagramAccount) -> None:
        await self._redis.hset(_ACCOUNTS_HASH, account.id, json.dumps(account.to_dict()))

    async def list_all(self) -> list[InstagramAccount]:
        raw = await self._redis.hgetall(_ACCOUNTS_HASH)
        accounts = []
        for v in raw.values():
            try:
                accounts.append(InstagramAccount.from_dict(json.loads(v)))
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
        return sorted(accounts, key=lambda a: a.created_at)

    async def register(
        self,
        username: str,
        password: str,
        email: str,
        phone: str,
        phone_country: str = "AU",
        vpn_country: str = "AU",
        smspool_order_id: str = "",
    ) -> InstagramAccount:
        account_id = str(uuid.uuid4())
        account = InstagramAccount(
            id=account_id,
            username=username,
            password=password,
            email=email,
            phone=phone,
            phone_country=phone_country,
            vpn_country=vpn_country,
            smspool_order_id=smspool_order_id,
            state="CREATED",
            created_at=int(time.time()),
            cookies_path=str(self._cookie_dir / f"{account_id}.txt"),
        )
        await self._save(account)
        logger.info("Registered new account %s (%s) as CREATED", account_id, username)
        return account

    # --------------------------------------------------------- State transitions

    async def start_warming(self, account_id: str) -> None:
        """CREATED → WARMING."""
        account = await self._require(account_id)
        account.state = "WARMING"
        account.warmed_since = int(time.time())
        await self._save(account)
        await self._redis.sadd(_WARMING_SET, account_id)
        logger.info("Account %s → WARMING", account_id)

    async def promote(self, account_id: str) -> None:
        """WARMING → ACTIVE."""
        account = await self._require(account_id)
        account.state = "ACTIVE"
        account.promoted_at = int(time.time())
        await self._save(account)
        await self._redis.srem(_WARMING_SET, account_id)
        await self._redis.sadd(_ACTIVE_SET, account_id)
        logger.info("Account %s → ACTIVE", account_id)

    async def flag(self, account_id: str, reason: str = "") -> None:
        """ACTIVE/WARMING → FLAGGED."""
        account = await self._require(account_id)
        account.state = "FLAGGED"
        account.flagged_at = int(time.time())
        account.flag_reason = reason
        await self._save(account)
        for s in _ALL_STATE_SETS:
            await self._redis.srem(s, account_id)
        await self._redis.sadd(_FLAGGED_SET, account_id)
        logger.warning("Account %s → FLAGGED: %s", account_id, reason)

    async def retire(self, account_id: str) -> None:
        """Any state → RETIRED. Removes from all state sets and deletes stored cookies."""
        account = await self._require(account_id)
        account.state = "RETIRED"
        await self._save(account)
        for s in _ALL_STATE_SETS:
            await self._redis.srem(s, account_id)
        await self._redis.delete(f"{_COOKIE_KEY_PREFIX}{account_id}")
        logger.info("Account %s → RETIRED", account_id)

    async def unflag(self, account_id: str) -> None:
        """FLAGGED → WARMING (for manual false-positive recovery)."""
        account = await self._require(account_id)
        account.state = "WARMING"
        account.flag_reason = None
        account.flagged_at = None
        account.warmed_since = account.warmed_since or int(time.time())
        await self._save(account)
        await self._redis.srem(_FLAGGED_SET, account_id)
        await self._redis.sadd(_WARMING_SET, account_id)
        logger.info("Account %s FLAGGED → WARMING (unflagged)", account_id)

    async def increment_warmup_session(self, account_id: str) -> None:
        account = await self.get(account_id)
        if not account:
            return
        account.warmup_sessions += 1
        await self._save(account)

    async def record_warmup_session(self, account_id: str) -> None:
        """Increment warmup_sessions and stamp last_warmed_at."""
        account = await self.get(account_id)
        if not account:
            return
        account.warmup_sessions += 1
        account.last_warmed_at = int(time.time())
        await self._save(account)

    async def mark_has_profile_pic(self, account_id: str) -> None:
        account = await self.get(account_id)
        if not account:
            return
        account.has_profile_pic = True
        await self._save(account)

    # --------------------------------------------------------- Cookie management

    async def get_cookie_content(self, account_id: str) -> str | None:
        """Return the stored Netscape cookie string for an account, or None.

        Lazy-migrates from the on-disk cookies_path if Redis is empty — handles
        accounts that were created before Phase 3 moved storage to Redis.
        """
        raw = await self._redis.get(f"{_COOKIE_KEY_PREFIX}{account_id}")
        if raw is not None:
            return raw if isinstance(raw, str) else raw.decode()

        # Lazy migration: seed Redis from the on-disk file recorded in account data
        try:
            account = await self.get(account_id)
            if account and account.cookies_path:
                path = Path(account.cookies_path)
                if path.exists():
                    content = path.read_text(encoding="utf-8", errors="replace")
                    await self.set_cookie_content(account_id, content)
                    logger.info(
                        "Lazy-migrated cookies from disk to Redis for account %s (@%s)",
                        account_id,
                        account.username,
                    )
                    return content
        except Exception as exc:
            logger.warning("Cookie lazy-migration failed for %s: %s", account_id, exc)

        return None

    async def set_cookie_content(self, account_id: str, content: str) -> None:
        """Store Netscape cookie string in Redis for an account."""
        await self._redis.set(f"{_COOKIE_KEY_PREFIX}{account_id}", content)
        logger.debug("Cookie content stored for account %s", account_id)

    async def import_cookies(self, account_id: str, source_path: Path) -> None:
        """Read a Netscape cookie file from disk and store its content in Redis."""
        content = source_path.read_text(encoding="utf-8", errors="replace")
        await self.set_cookie_content(account_id, content)
        # Keep cookies_path as a record of where the file came from
        account = await self._require(account_id)
        account.cookies_path = str(source_path)
        await self._save(account)
        logger.info("Imported cookies for account %s from %s", account_id, source_path)

    def _materialize_cookie(self, account_id: str, content: str) -> Path:
        """Write cookie content to a temp file and return the path."""
        temp_path = _TEMP_DIR / f"osia_ig_{account_id}.txt"
        temp_path.write_text(content, encoding="utf-8")
        return temp_path

    async def get_active_cookie_path(self) -> tuple[str, Path] | None:
        """
        Round-robin over ACTIVE accounts. Returns (account_id, temp_cookie_path) or None.
        Cookie content is read from Redis and materialised to a temp file for yt-dlp.
        """
        active_ids = [a.decode() if isinstance(a, bytes) else a for a in await self._redis.smembers(_ACTIVE_SET)]
        if not active_ids:
            return None

        active_ids.sort()
        raw_current = await self._redis.get(_CURRENT_KEY)
        current = raw_current.decode() if isinstance(raw_current, bytes) else raw_current
        if current and current in active_ids:
            idx = (active_ids.index(current) + 1) % len(active_ids)
        else:
            idx = 0

        for _ in range(len(active_ids)):
            candidate_id = active_ids[idx % len(active_ids)]
            content = await self.get_cookie_content(candidate_id)
            if content:
                await self._redis.set(_CURRENT_KEY, candidate_id)
                return candidate_id, self._materialize_cookie(candidate_id, content)
            idx += 1

        return None

    async def get_next_active_cookie_path(self, skip_ids: str | set[str]) -> tuple[str, Path] | None:
        """Like get_active_cookie_path() but skips one or more already-tried account IDs."""
        exclude = {skip_ids} if isinstance(skip_ids, str) else set(skip_ids)
        active_ids = [
            a.decode() if isinstance(a, bytes) else a
            for a in await self._redis.smembers(_ACTIVE_SET)
            if (a.decode() if isinstance(a, bytes) else a) not in exclude
        ]
        if not active_ids:
            return None
        active_ids.sort()
        for aid in active_ids:
            content = await self.get_cookie_content(aid)
            if content:
                await self._redis.set(_CURRENT_KEY, aid)
                return aid, self._materialize_cookie(aid, content)
        return None

    # --------------------------------------------------------- Pool health

    async def pool_counts(self) -> dict[str, int]:
        active = await self._redis.scard(_ACTIVE_SET)
        warming = await self._redis.scard(_WARMING_SET)
        flagged = await self._redis.scard(_FLAGGED_SET)
        return {"active": int(active), "warming": int(warming), "flagged": int(flagged)}

    async def needs_creation(self) -> bool:
        counts = await self.pool_counts()
        return counts["active"] < _ACTIVE_MIN

    async def needs_warming_slot(self) -> bool:
        counts = await self.pool_counts()
        return counts["warming"] < _WARMING_MIN

    async def eligible_for_promotion(self) -> list[str]:
        """Return WARMING account_ids that meet both the age and session-count thresholds."""
        warming_ids = await self._redis.smembers(_WARMING_SET)
        min_sessions = int(os.getenv("IG_WARMUP_MIN_SESSIONS", "5"))
        threshold = int(time.time()) - _WARMUP_DAYS * 86400
        eligible = []
        for aid in warming_ids:
            account = await self.get(aid)
            if (
                account
                and account.warmed_since
                and account.warmed_since <= threshold
                and account.warmup_sessions >= min_sessions
            ):
                eligible.append(aid)
        return eligible

    # ---------------------------------------------------------------- Helpers

    async def _require(self, account_id: str) -> InstagramAccount:
        account = await self.get(account_id)
        if not account:
            raise ValueError(f"Account {account_id} not found")
        return account
