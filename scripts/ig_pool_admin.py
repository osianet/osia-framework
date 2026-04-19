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


def _ts(unix: int | None) -> str:
    if not unix:
        return ""
    return datetime.fromtimestamp(unix, UTC).strftime("%Y-%m-%d %H:%M")


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
        "--create-account",
        metavar="COUNTRY",
        dest="create_account",
        help="Full automated creation via Camoufox (e.g. AU, US, UK). Requires root for VPN switching.",
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in visible (non-headless) mode — pauses at CAPTCHA for human interaction",
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
        if args.list:
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
            dest = await mgr.import_cookies(account_id, Path(path))
            print(f"Imported cookies for {account_id} → {dest}")

        elif args.export_cookies:
            account = await mgr.get(args.export_cookies)
            if not account:
                print(f"Account {args.export_cookies} not found", file=sys.stderr)
                sys.exit(1)
            print(account.cookies_path)

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
            print(f"Cookie path: {account.cookies_path}")
            print(f"Next: import cookies with --import-cookies {account.id} <path>, then --start-warming {account.id}")

        elif args.create_account:
            country = args.create_account.upper()
            headed = args.headed
            print(f"Starting account creation for country={country} (VPN managed externally, headed={headed})...")
            creator = InstagramCreator(mgr, headless=not headed)
            account = await creator.create_new(country=country, skip_vpn=True)
            print(f"\nSuccess: {account.id}  ({account.username})  state=WARMING")
            print(f"Cookie path: {account.cookies_path}")

    finally:
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
