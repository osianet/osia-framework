"""
Instagram account pool manager.

Redis-backed state machine: CREATED → WARMING → ACTIVE → FLAGGED → RETIRED

Cookie files live at config/ig_cookies/<account_id>.txt.
SMSPool API is used to purchase OTP phone numbers for account creation.
"""

import asyncio
import json
import logging
import os
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
_WARMUP_QUEUE = "osia:ig:warmup:queue"

_ALL_STATE_SETS = (_ACTIVE_SET, _WARMING_SET, _FLAGGED_SET)


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
        self._cookie_dir = self._base_dir / "config" / "ig_cookies"
        self._cookie_dir.mkdir(parents=True, exist_ok=True)

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
            except Exception:
                pass
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
        """Any state → RETIRED (removed from all active sets)."""
        account = await self._require(account_id)
        account.state = "RETIRED"
        await self._save(account)
        for s in _ALL_STATE_SETS:
            await self._redis.srem(s, account_id)
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

    # --------------------------------------------------------- Cookie management

    async def import_cookies(self, account_id: str, source_path: Path) -> Path:
        """Copy an external cookie file into the pool's managed directory."""
        account = await self._require(account_id)
        dest = self._cookie_dir / f"{account_id}.txt"
        dest.write_bytes(source_path.read_bytes())
        account.cookies_path = str(dest)
        await self._save(account)
        logger.info("Imported cookies for account %s → %s", account_id, dest)
        return dest

    async def get_active_cookie_path(self) -> tuple[str, Path] | None:
        """
        Round-robin over ACTIVE accounts. Returns (account_id, cookie_path) or None.
        Advances osia:ig:current to the chosen account on each call.
        """
        active_ids = list(await self._redis.smembers(_ACTIVE_SET))
        if not active_ids:
            return None

        active_ids.sort()
        current = await self._redis.get(_CURRENT_KEY)
        if current and current in active_ids:
            idx = (active_ids.index(current) + 1) % len(active_ids)
        else:
            idx = 0

        for _ in range(len(active_ids)):
            candidate_id = active_ids[idx % len(active_ids)]
            account = await self.get(candidate_id)
            if account and account.cookies_path and Path(account.cookies_path).exists():
                await self._redis.set(_CURRENT_KEY, candidate_id)
                return candidate_id, Path(account.cookies_path)
            idx += 1

        return None

    async def get_next_active_cookie_path(self, skip_id: str) -> tuple[str, Path] | None:
        """Like get_active_cookie_path() but skips a specific account_id (e.g. one just flagged)."""
        active_ids = list(await self._redis.smembers(_ACTIVE_SET))
        active_ids = [a for a in active_ids if a != skip_id]
        if not active_ids:
            return None
        active_ids.sort()
        for aid in active_ids:
            account = await self.get(aid)
            if account and account.cookies_path and Path(account.cookies_path).exists():
                await self._redis.set(_CURRENT_KEY, aid)
                return aid, Path(account.cookies_path)
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
        """Return WARMING account_ids that have exceeded IG_WARMUP_DAYS."""
        warming_ids = await self._redis.smembers(_WARMING_SET)
        threshold = int(time.time()) - _WARMUP_DAYS * 86400
        eligible = []
        for aid in warming_ids:
            account = await self.get(aid)
            if account and account.warmed_since and account.warmed_since <= threshold:
                eligible.append(aid)
        return eligible

    # ---------------------------------------------------------------- Helpers

    async def _require(self, account_id: str) -> InstagramAccount:
        account = await self.get(account_id)
        if not account:
            raise ValueError(f"Account {account_id} not found")
        return account
