# Instagram Account Pool — Implementation Plan

**Goal:** Replace the single `config/instagram_cookies.txt` with a managed pool of Instagram
accounts that are automatically created, warmed up, rotated, and retired — keeping yt-dlp
ingestion of operatives' reels reliably online at all times.

**Status:** Planning  
**Branch:** `feat/instagram-account-pool`

---

## Background

- yt-dlp with `--cookies instagram_cookies.txt` works perfectly for reel download
- Accounts are flagged/banned within ~24h of creation (no warm-up, single IP, fresh account)
- SMSPool API creds are in `.env` (`SMSPOOL_API_KEY`, `SMSPOOL_USER`)
- Surfshark WireGuard is available on the server for country-matched IPs
- Existing ADB device + `social_media_agent.py` + `persona_daemon.py` can drive warm-up
- Camoufox (already in stack from browser-automation spike) handles realistic browser fingerprinting
- Single ADB device — contention managed via `osia:adb:lock` Redis key

---

## Architecture

### Account State Machine

```
CREATED → WARMING → ACTIVE → FLAGGED → RETIRED
```

| State   | Description                                              |
|---------|----------------------------------------------------------|
| CREATED | Credentials exist, account not yet verified/active       |
| WARMING | Active warm-up period (7+ days of simulated activity)    |
| ACTIVE  | Ready for yt-dlp ingestion; rotated round-robin          |
| FLAGGED | 401/429 received from yt-dlp; excluded from rotation     |
| RETIRED | Confirmed banned; credentials archived                   |

**Pool targets (configurable via env):**
- `IG_POOL_ACTIVE_MIN=3` — trigger creation when ACTIVE drops below this
- `IG_POOL_WARMING_MIN=3` — keep this many accounts warming at all times
- `IG_WARMUP_DAYS=7` — minimum days in WARMING before promoting to ACTIVE

### Redis Keys

| Key | Type | Content |
|-----|------|---------|
| `osia:ig:accounts` | Hash | `{account_id: JSON blob}` |
| `osia:ig:pool:active` | Set | account_ids in ACTIVE state |
| `osia:ig:pool:warming` | Set | account_ids in WARMING state |
| `osia:ig:pool:flagged` | Set | account_ids in FLAGGED state |
| `osia:ig:current` | String | account_id currently being used by yt-dlp |
| `osia:ig:warmup:queue` | List | account_ids queued for next warm-up session |

### Account Record (JSON in Redis hash)

```json
{
  "id": "uuid4",
  "username": "...",
  "password": "...",
  "email": "...",
  "phone": "+61412345678",
  "phone_country": "AU",
  "vpn_country": "AU",
  "smspool_order_id": "...",
  "state": "WARMING",
  "created_at": 1234567890,
  "warmed_since": 1234567890,
  "promoted_at": null,
  "flagged_at": null,
  "cookies_path": "config/ig_cookies/uuid4.txt",
  "flag_reason": null,
  "warmup_sessions": 3
}
```

### Cookie Storage

Each account gets its own cookie file at `config/ig_cookies/<account_id>.txt`.
The orchestrator's `_fetch_yt_dlp_metadata` and related methods will be updated to
pull the current ACTIVE account's cookie path from Redis rather than the hardcoded
`instagram_cookies.txt`.

---

## Files To Create / Modify

### New Files

| File | Purpose |
|------|---------|
| `src/agents/instagram_account_manager.py` | Pool state machine, SMSPool integration, credential store |
| `src/agents/instagram_warmup_agent.py` | ADB + Camoufox warm-up sessions, reuses SocialMediaAgent |
| `src/agents/instagram_creator.py` | Account creation flow via Camoufox browser automation |
| `src/cron/instagram_pool_health.py` | Timer job: check pool depth, trigger creation if below threshold |
| `scripts/ig_pool_admin.py` | CLI tool: inspect pool, promote/retire accounts manually, export cookies |

### Modified Files

| File | Change |
|------|--------|
| `src/orchestrator.py` | Replace hardcoded `instagram_cookies.txt` with pool manager lookup (~3 locations) |
| `config/` | Add `ig_cookies/` directory (gitignored) |
| `.env` | Add `IG_POOL_ACTIVE_MIN`, `IG_POOL_WARMING_MIN`, `IG_WARMUP_DAYS` |
| `osia.sh` | Add `instagram-pool-health` timer to start/stop/status |

---

## Implementation Tasks

### Phase 1 — Account Pool Manager (no ADB yet)
- [x] **1.1** Create `src/agents/instagram_account_manager.py`
  - Redis-backed CRUD for account records
  - State transition methods (`promote`, `flag`, `retire`, `unflag`)
  - `get_active_cookie_path()` — round-robin over ACTIVE set, returns cookie file path
  - `get_next_active_cookie_path(skip_id)` — rotation after flagging
  - Pool depth checks (`needs_creation()`, `needs_warmup_slot()`)
  - SMSPool API client (number purchase, OTP polling, order cancel)
- [x] **1.2** Create `scripts/ig_pool_admin.py`
  - `--list` — tabular view of all accounts + states
  - `--promote <id>` / `--flag <id>` / `--unflag <id>` / `--retire <id>` — manual overrides
  - `--start-warming <id>` — CREATED → WARMING
  - `--import-cookies <id> <path>` — load existing cookie file into pool
  - `--export-cookies <id>` — dump cookie file path for manual yt-dlp testing
  - `--create-account <country>` — full automated creation via Camoufox (Phase 2)
- [x] **1.3** Update orchestrator to use pool manager
  - `_resolve_instagram_cookie()` — tries pool first, falls back to legacy `instagram_cookies.txt`
  - `_yt_dlp_auth_error()` — detects 401/429/login-required in yt-dlp stderr
  - `_fetch_yt_dlp_metadata`: uses pool cookie, flags account on auth error
  - `_download_video_yt_dlp`: same + automatic retry with next ACTIVE account
  - `source_account` field written to Qdrant payload on every Instagram reel
- [x] **1.4** `--sync-cookies` command in `ig_pool_admin.py`
  - Fixes Windows→Linux cookie paths in Redis records
  - Promotes any WARMING/CREATED account whose cookie file now exists
  - Run after each batch copy from desktop: `uv run python scripts/ig_pool_admin.py --sync-cookies`

### Phase 2 — Account Creation (Camoufox)
- [x] **2.1** Create `src/agents/instagram_creator.py`
  - Purchases SMSPool number, switches VPN, runs Camoufox signup
  - Polls SMSPool OTP, enters it, exports Netscape cookies
  - Registers account in pool as CREATED → WARMING on success
  - Retires account + cancels SMSPool order on any failure
  - Restores original VPN in `finally` block
- [x] **2.2** VPN country switching
  - Integrated directly into `InstagramCreator._switch_vpn()`
  - Parses current wg0.conf endpoint to identify and restore original slug
  - Supported countries: any slug in `/etc/wireguard/countries/` (31 available)
- [x] **2.3** End-to-end creation test — complete
  - Headed mode (`--headed`) used on Windows desktop; browser visible for CAPTCHA solving
  - OTP space-stripping added to handle SMSPool codes with whitespace
  - Multiple batches of accounts created and imported; 15 ACTIVE as of 2026-04-19

### Phase 3 — Warm-up Scheduler (ADB + Camoufox)  ← **NEXT**
- [ ] **3.1** Create `src/agents/instagram_warmup_agent.py`
  - Inherits/reuses `SocialMediaAgent` ADB patterns
  - Per-session warm-up routine: open Instagram app, browse feed 3-5 min, like 1-2 posts
  - **Follow accounts from `osia:ig:intel_sources`** Redis set (395 handles as of 2026-04-19)
    - Pick a random subset per session (e.g. 2-3 follows) from the set
  - Tracks `warmup_sessions` count on account record
  - After `IG_WARMUP_DAYS` days AND min sessions met → `promote()` to ACTIVE
  - Respects `osia:adb:lock` (will not run if orchestrator holds the lock)
- [ ] **3.2** Create `src/cron/instagram_pool_health.py`
  - Runs as a timer job (every 4h)
  - Checks ACTIVE count vs `IG_POOL_ACTIVE_MIN` → triggers creation if low
  - Checks WARMING accounts → runs one warm-up session per WARMING account (if ADB free)
  - Promotes eligible WARMING accounts to ACTIVE
  - Alerts to Signal if ACTIVE pool drops to 0
- [ ] **3.3** Wire up systemd timer
  - `osia-instagram-pool-health.service` + `.timer` (every 4h)
  - Add to `osia.sh`

### Phase 4 — Hardening & Monitoring
- [ ] **4.1** Fingerprint consistency
  - Store User-Agent per account in Redis record
  - Ensure Camoufox warm-up and creation sessions use same UA as account profile
- [ ] **4.2** Behavioural realism
  - Random human delays between ADB actions (already in `SocialMediaAgent._human_delay`)
  - Vary warm-up session length (3-8 min) and actions (browse only / browse+like / browse+like+follow)
  - Spread sessions across day, not just quiet window
- [ ] **4.3** Pool health metric in status API
  - Add `/status/instagram` endpoint to `osia-status-api` showing ACTIVE/WARMING/FLAGGED counts
- [ ] **4.4** Quiet-hours scheduling
  - Warm-up sessions only run 22:00–06:00 local time (low OSIA load)
  - Creation pipeline can run anytime (no ADB needed for browser creation)

---

## Environment Variables (to add to `.env`)

```bash
# Instagram account pool
IG_POOL_ACTIVE_MIN=3        # trigger creation below this
IG_POOL_WARMING_MIN=3       # keep this many warming
IG_WARMUP_DAYS=7            # days in WARMING before promotion
IG_CREATION_COUNTRY=AU      # default country for new accounts (AU/US/UK)

# SMSPool (already present)
SMSPOOL_API_KEY=...
SMSPOOL_USER=...
```

---

## Surfshark / WireGuard Notes

- Configs live at `/etc/wireguard/` — check available country configs before coding
- Need to identify correct interface names for AU, US, UK exits
- Switch must be atomic: bring down old, bring up new, verify connectivity before proceeding
- Restore original config (likely `wg0`) after creation finishes

---

## Open Questions

1. ~~Does Camoufox support headless mode?~~ — Resolved: headless unstable for Instagram signup; use `--headed` on Windows desktop instead
2. ~~SMSPool AU country code~~ — Resolved: service ID 457, AU works
3. Should FLAGGED accounts auto-retry after 48h or go straight to RETIRED?
4. Can we run ADB warm-up sessions for different accounts sequentially on the same device, or does Instagram fingerprint the device itself?

---

## Session Log

| Date | Work Done |
|------|-----------|
| 2026-04-19 | Architecture design, task document created |
| 2026-04-19 | Phase 1 complete: `instagram_account_manager.py`, `ig_pool_admin.py` |
| 2026-04-19 | Phase 2 complete: `instagram_creator.py` with Camoufox signup + VPN switching + OTP + cookie export |
| 2026-04-19 | Orchestrator wiring complete (Phase 1.3): pool cookie rotation, auth-failure flagging, retry logic |
| 2026-04-19 | `--sync-cookies` added to `ig_pool_admin.py` — fixes Windows paths + promotes in one command |
| 2026-04-19 | Social account dossiers: `entities/social-accounts/instagram/<handle>` in wiki; INTSUM pages link to creator dossier; `osia:ig:intel_sources` Redis set populated with 395 handles via text-mine backfill |
| 2026-04-19 | Pool state: 15 ACTIVE, 2 WARMING (missing cookies), 0 FLAGGED |
