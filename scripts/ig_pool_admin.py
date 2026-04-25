#!/usr/bin/env python3
"""
Instagram account pool administration CLI.

Usage:
    uv run python scripts/ig_pool_admin.py --list
    uv run python scripts/ig_pool_admin.py --promote <id>
    uv run python scripts/ig_pool_admin.py --flag <id> [--reason "..."]
    uv run python scripts/ig_pool_admin.py --unflag <id>
    uv run python scripts/ig_pool_admin.py --retire <id>
    uv run python scripts/ig_pool_admin.py --start-warming <id>
    uv run python scripts/ig_pool_admin.py --import-cookies <id> <path>
    uv run python scripts/ig_pool_admin.py --export-cookies <id>
    uv run python scripts/ig_pool_admin.py --create --username foo --password bar \\
        --email foo@bar.com --phone +61400000000 [--phone-country AU] [--vpn-country AU]
    uv run python scripts/ig_pool_admin.py --sync-cookies
    uv run python scripts/ig_pool_admin.py --warm <id> [--headed] [--no-avatar]
    uv run python scripts/ig_pool_admin.py --warm-all [--headed] [--no-avatar]
"""

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import redis.asyncio as redis_async  # noqa: E402

from src.agents.instagram_account_manager import InstagramAccountManager  # noqa: E402
from src.agents.instagram_creator import InstagramCreator  # noqa: E402
from src.agents.instagram_warmup_agent import InstagramWarmupSession  # noqa: E402


def _ts(unix: int | None) -> str:
    if not unix:
        return ""
    return datetime.fromtimestamp(unix, UTC).strftime("%Y-%m-%d %H:%M")


async def cmd_sync_cookies(mgr: InstagramAccountManager) -> None:
    """
    Import cookie file content into Redis for any account that has a cookie file on disk.
    Promotes WARMING/CREATED accounts whose cookie content is now in Redis.
    Retires WARMING/CREATED accounts with no cookies anywhere — creation failures.
    """
    cookie_dir = Path(__file__).resolve().parent.parent / "config" / "ig_cookies"
    accounts = await mgr.list_all()
    imported = promoted = retired = already_in_redis = 0

    for acc in accounts:
        if acc.state == "RETIRED":
            continue

        # Already in Redis — check for promotion
        existing = await mgr.get_cookie_content(acc.id)
        if existing:
            already_in_redis += 1
            if acc.state in ("WARMING", "CREATED"):
                await mgr.promote(acc.id)
                print(f"Promoted {acc.username} ({acc.id[:8]}…) → ACTIVE")
                promoted += 1
            continue

        # Try importing from disk
        expected = cookie_dir / f"{acc.id}.txt"
        if expected.exists():
            await mgr.import_cookies(acc.id, expected)
            print(f"Imported  {acc.username} ({acc.id[:8]}…) → Redis")
            imported += 1
            if acc.state in ("WARMING", "CREATED"):
                await mgr.promote(acc.id)
                print(f"Promoted  {acc.username} ({acc.id[:8]}…) → ACTIVE")
                promoted += 1
            continue

        # No cookies anywhere — creation failure; retire
        if acc.state in ("WARMING", "CREATED"):
            await mgr.retire(acc.id)
            print(f"Retired   {acc.username} ({acc.id[:8]}…) — no cookies (failed creation)")
            retired += 1

    print(f"\nDone. in_redis={already_in_redis} imported={imported} promoted={promoted} retired_orphans={retired}")


async def cmd_list(mgr: InstagramAccountManager) -> None:
    accounts = await mgr.list_all()
    counts = await mgr.pool_counts()
    print(f"\nPool: {counts['active']} ACTIVE | {counts['warming']} WARMING | {counts['flagged']} FLAGGED\n")
    col = "{:<36}  {:<20}  {:<10}  {:<16}  {:<8}  {:<16}  {}"
    print(col.format("ID", "USERNAME", "STATE", "CREATED", "WARMUPS", "PROMOTED", "FLAG_REASON"))
    print("-" * 130)
    for acc in accounts:
        print(
            col.format(
                acc.id,
                acc.username[:20],
                acc.state,
                _ts(acc.created_at),
                str(acc.warmup_sessions),
                _ts(acc.promoted_at),
                acc.flag_reason or "",
            )
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Instagram account pool admin")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Show all accounts")
    group.add_argument("--promote", metavar="ID", help="WARMING → ACTIVE")
    group.add_argument("--flag", metavar="ID", help="ACTIVE/WARMING → FLAGGED")
    group.add_argument("--unflag", metavar="ID", help="FLAGGED → WARMING")
    group.add_argument("--retire", metavar="ID", help="Any → RETIRED")
    group.add_argument("--start-warming", metavar="ID", dest="start_warming", help="CREATED → WARMING")
    group.add_argument("--import-cookies", nargs=2, metavar=("ID", "PATH"), dest="import_cookies")
    group.add_argument("--export-cookies", metavar="ID", dest="export_cookies")
    group.add_argument("--create", action="store_true", help="Register a new account manually")
    group.add_argument(
        "--sync-cookies",
        action="store_true",
        dest="sync_cookies",
        help="Fix paths + promote accounts whose cookie file exists",
    )
    group.add_argument(
        "--create-account",
        metavar="COUNTRY",
        dest="create_account",
        help="Full automated creation via Camoufox (e.g. AU, US, UK). Requires root for VPN switching.",
    )
    group.add_argument(
        "--warm",
        metavar="ID",
        help="Run a single warm-up session for the given account ID.",
    )
    group.add_argument(
        "--warm-all",
        action="store_true",
        dest="warm_all",
        help="Run a warm-up session for every WARMING account (with inter-session delay).",
    )
    group.add_argument(
        "--relogin",
        metavar="ID",
        help="Open headed browser, fill stored credentials, capture fresh cookies.",
    )
    group.add_argument(
        "--relogin-all",
        action="store_true",
        dest="relogin_all",
        help="Re-login every ACTIVE/WARMING account in sequence to refresh cookies.",
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in visible (non-headless) mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save a screenshot at each warmup step to logs/ig_debug/<id>/",
    )
    parser.add_argument(
        "--no-avatar",
        action="store_true",
        dest="no_avatar",
        help="Skip profile picture upload during warm-up",
    )
    parser.add_argument("--reason", default="", help="Reason string for --flag")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--email")
    parser.add_argument("--phone")
    parser.add_argument("--phone-country", default="AU", dest="phone_country")
    parser.add_argument("--vpn-country", default="AU", dest="vpn_country")
    parser.add_argument("--smspool-order-id", default="", dest="smspool_order_id")

    args = parser.parse_args()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = redis_async.from_url(redis_url, decode_responses=True)
    mgr = InstagramAccountManager(r)

    try:
        if args.sync_cookies:
            await cmd_sync_cookies(mgr)

        elif args.list:
            await cmd_list(mgr)

        elif args.promote:
            await mgr.promote(args.promote)
            print(f"Promoted {args.promote} → ACTIVE")

        elif args.flag:
            await mgr.flag(args.flag, reason=args.reason)
            print(f"Flagged {args.flag}" + (f": {args.reason}" if args.reason else ""))

        elif args.unflag:
            await mgr.unflag(args.unflag)
            print(f"Unflagged {args.unflag} → WARMING")

        elif args.retire:
            await mgr.retire(args.retire)
            print(f"Retired {args.retire}")

        elif args.start_warming:
            await mgr.start_warming(args.start_warming)
            print(f"Started warming for {args.start_warming}")

        elif args.import_cookies:
            account_id, path = args.import_cookies
            await mgr.import_cookies(account_id, Path(path))
            print(f"Imported cookies for {account_id} into Redis")

        elif args.export_cookies:
            account_id = args.export_cookies
            content = await mgr.get_cookie_content(account_id)
            if not content:
                print(f"No cookie content in Redis for {account_id}", file=sys.stderr)
                sys.exit(1)
            # Write to a file next to the cookie dir for manual use
            out_path = Path(__file__).resolve().parent.parent / "config" / "ig_cookies" / f"{account_id}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(out_path)

        elif args.create:
            missing = [f for f in ("username", "password", "email", "phone") if not getattr(args, f)]
            if missing:
                print(f"ERROR: --{missing[0]} is required for --create", file=sys.stderr)
                sys.exit(1)
            account = await mgr.register(
                username=args.username,
                password=args.password,
                email=args.email,
                phone=args.phone,
                phone_country=args.phone_country,
                vpn_country=args.vpn_country,
                smspool_order_id=args.smspool_order_id,
            )
            print(f"Created: {account.id}  ({account.username})  state=CREATED")
            print(f"Next: --import-cookies {account.id} <path>  then  --start-warming {account.id}")

        elif args.create_account:
            country = args.create_account.upper()
            headed = args.headed
            print(f"Starting account creation for country={country} (VPN managed externally, headed={headed})...")
            creator = InstagramCreator(mgr, headless=not headed)
            account = await creator.create_new(country=country, skip_vpn=True)
            print(f"\nSuccess: {account.id}  ({account.username})  state=WARMING")
            print(f"Cookie path: {account.cookies_path}")

        elif args.warm:
            upload_avatar = not args.no_avatar
            session = InstagramWarmupSession(mgr, r, headed=args.headed, upload_avatar=upload_avatar, debug=args.debug)
            print(
                f"Running warm-up for {args.warm} (headed={args.headed}, avatar={upload_avatar}, debug={args.debug})…"
            )
            success = await session.run(args.warm)
            print("Done." if success else "Session failed — check logs.")

        elif args.warm_all:
            warming_ids = list(await r.smembers("osia:ig:pool:warming"))
            if not warming_ids:
                print("No WARMING accounts.")
            else:
                upload_avatar = not args.no_avatar
                session = InstagramWarmupSession(
                    mgr, r, headed=args.headed, upload_avatar=upload_avatar, debug=args.debug
                )
                inter_delay = int(os.getenv("IG_INTER_SESSION_DELAY_SECS", "900"))
                warming_accounts = []
                for aid in warming_ids:
                    acc = await mgr.get(aid)
                    if acc:
                        warming_accounts.append(acc)
                warming_accounts.sort(key=lambda a: a.last_warmed_at or 0)
                for i, acc in enumerate(warming_accounts):
                    if i > 0:
                        print(f"Waiting {inter_delay}s before next account…")
                        await asyncio.sleep(inter_delay)
                    print(f"[{i + 1}/{len(warming_accounts)}] Warming @{acc.username}…")
                    success = await session.run(acc.id)
                    print("  OK" if success else "  FAILED")

        elif args.relogin:
            session = InstagramWarmupSession(mgr, r, debug=args.debug)
            print(f"Re-login for {args.relogin}…")
            success = await session.relogin(args.relogin)
            print("Fresh cookies saved." if success else "Re-login failed — check logs.")

        elif args.relogin_all:
            all_accounts = await mgr.list_all()
            targets = [a for a in all_accounts if a.state in ("ACTIVE", "WARMING")]
            if not targets:
                print("No ACTIVE or WARMING accounts.")
            else:
                session = InstagramWarmupSession(mgr, r, debug=args.debug)
                for i, acc in enumerate(targets):
                    print(f"\n[{i + 1}/{len(targets)}] Re-login @{acc.username} ({acc.id[:8]}…)")
                    success = await session.relogin(acc.id)
                    print("  Fresh cookies saved." if success else "  FAILED — skipping.")
                    if i < len(targets) - 1:
                        input("\nPress Enter when ready for the next account…")

    finally:
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
