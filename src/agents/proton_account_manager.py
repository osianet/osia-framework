"""
Proton Mail account pool manager.

Redis-backed. Accounts are created via ProtonCreator (Camoufox) and stored here.
Used as the email source for Instagram account creation.

State machine: CREATED → AVAILABLE → USED → RETIRED
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import redis.asyncio as redis

logger = logging.getLogger("osia.proton_pool")

_ACCOUNTS_HASH = "osia:proton:accounts"
_AVAILABLE_SET = "osia:proton:pool:available"
_USED_SET = "osia:proton:pool:used"
_CURRENT_KEY = "osia:proton:current"

_ALL_STATE_SETS = (_AVAILABLE_SET, _USED_SET)


@dataclass
class ProtonAccount:
    id: str
    username: str  # without @proton.me
    password: str
    state: str  # CREATED | AVAILABLE | USED | RETIRED
    created_at: int
    used_at: int | None = None
    used_for: str | None = None  # e.g. Instagram account ID it was used for

    @property
    def email(self) -> str:
        return f"{self.username}@proton.me"

    @classmethod
    def from_dict(cls, data: dict) -> "ProtonAccount":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


class ProtonAccountManager:
    """Redis-backed Proton Mail account pool."""

    def __init__(self, redis_client: redis.Redis, base_dir: Path | None = None):
        self._redis = redis_client
        self._base_dir = base_dir or Path(__file__).resolve().parent.parent.parent

    # ------------------------------------------------------------------ CRUD

    async def get(self, account_id: str) -> ProtonAccount | None:
        raw = await self._redis.hget(_ACCOUNTS_HASH, account_id)
        if not raw:
            return None
        try:
            return ProtonAccount.from_dict(json.loads(raw))
        except Exception:
            return None

    async def _save(self, account: ProtonAccount) -> None:
        await self._redis.hset(_ACCOUNTS_HASH, account.id, json.dumps(account.to_dict()))

    async def list_all(self) -> list[ProtonAccount]:
        raw = await self._redis.hgetall(_ACCOUNTS_HASH)
        accounts = []
        for v in raw.values():
            try:
                accounts.append(ProtonAccount.from_dict(json.loads(v)))
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)
        return sorted(accounts, key=lambda a: a.created_at)

    async def register(self, username: str, password: str) -> ProtonAccount:
        """Register a newly-created account as AVAILABLE."""
        account_id = str(uuid.uuid4())
        account = ProtonAccount(
            id=account_id,
            username=username,
            password=password,
            state="AVAILABLE",
            created_at=int(time.time()),
        )
        await self._save(account)
        await self._redis.sadd(_AVAILABLE_SET, account_id)
        logger.info("Registered Proton account %s (%s)", account_id[:8], account.email)
        return account

    # --------------------------------------------------------- State transitions

    async def mark_used(self, account_id: str, used_for: str = "") -> None:
        """AVAILABLE → USED (consumed for Instagram creation)."""
        account = await self._require(account_id)
        account.state = "USED"
        account.used_at = int(time.time())
        account.used_for = used_for
        await self._save(account)
        await self._redis.srem(_AVAILABLE_SET, account_id)
        await self._redis.sadd(_USED_SET, account_id)
        logger.info("Proton account %s marked USED (for %s)", account_id[:8], used_for or "unknown")

    async def retire(self, account_id: str) -> None:
        account = await self._require(account_id)
        account.state = "RETIRED"
        await self._save(account)
        for s in _ALL_STATE_SETS:
            await self._redis.srem(s, account_id)
        logger.info("Proton account %s → RETIRED", account_id[:8])

    # --------------------------------------------------------- Pool access

    async def pool_counts(self) -> dict[str, int]:
        available = await self._redis.scard(_AVAILABLE_SET)
        used = await self._redis.scard(_USED_SET)
        return {"available": int(available), "used": int(used)}

    async def claim_available(self, used_for: str = "") -> ProtonAccount | None:
        """
        Claim one AVAILABLE account (round-robin). Marks it USED immediately.
        Returns None if the pool is empty.
        """
        ids = [a.decode() if isinstance(a, bytes) else a for a in await self._redis.smembers(_AVAILABLE_SET)]
        if not ids:
            return None

        ids.sort()
        raw_current = await self._redis.get(_CURRENT_KEY)
        current = raw_current.decode() if isinstance(raw_current, bytes) else raw_current
        if current and current in ids:
            idx = (ids.index(current) + 1) % len(ids)
        else:
            idx = 0

        account_id = ids[idx]
        await self._redis.set(_CURRENT_KEY, account_id)
        await self.mark_used(account_id, used_for=used_for)
        return await self.get(account_id)

    # ---------------------------------------------------------------- Helpers

    async def _require(self, account_id: str) -> ProtonAccount:
        account = await self.get(account_id)
        if not account:
            raise ValueError(f"Proton account {account_id} not found")
        return account

    def env_var_name(self, account_id: str) -> str:
        return f"PROTON_{account_id[:8].upper().replace('-', '_')}"
