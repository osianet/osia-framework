#!/usr/bin/env python3
"""
Proton Mail account pool administration CLI.

Usage:
    uv run python scripts/proton_admin.py --list
    uv run python scripts/proton_admin.py --create [--count N] [--headed] [--debug]
    uv run python scripts/proton_admin.py --show <id>
    uv run python scripts/proton_admin.py --retire <id>
    uv run python scripts/proton_admin.py --claim [--for <ig_account_id>]
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

from src.agents.proton_account_manager import ProtonAccountManager  # noqa: E402
from src.agents.proton_creator import ProtonCreator  # noqa: E402
from src.agents.proton_reader import ProtonMailReader  # noqa: E402


def _ts(unix: int | None) -> str:
    if not unix:
        return ""
    return datetime.fromtimestamp(unix, UTC).strftime("%Y-%m-%d %H:%M")


async def cmd_list(mgr: ProtonAccountManager) -> None:
    accounts = await mgr.list_all()
    counts = await mgr.pool_counts()
    print(f"\nPool: {counts['available']} AVAILABLE | {counts['used']} USED\n")
    col = "{:<36}  {:<32}  {:<10}  {:<16}  {}"
    print(col.format("ID", "EMAIL", "STATE", "CREATED", "USED_FOR"))
    print("-" * 115)
    for acc in accounts:
        print(
            col.format(
                acc.id,
                acc.email[:32],
                acc.state,
                _ts(acc.created_at),
                acc.used_for or "",
            )
        )


async def cmd_show(mgr: ProtonAccountManager, account_id: str) -> None:
    account = await mgr.get(account_id)
    if not account:
        print(f"Account {account_id} not found", file=sys.stderr)
        sys.exit(1)
    print(f"ID:       {account.id}")
    print(f"Email:    {account.email}")
    print(f"Password: {account.password}")
    print(f"State:    {account.state}")
    print(f"Created:  {_ts(account.created_at)}")
    if account.used_at:
        print(f"Used:     {_ts(account.used_at)} (for {account.used_for or 'unknown'})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Proton Mail account pool admin")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Show all accounts")
    group.add_argument("--create", action="store_true", help="Create one or more Proton accounts")
    group.add_argument("--show", metavar="ID", help="Show credentials for an account")
    group.add_argument("--test", metavar="ID", help="Log in with stored credentials and confirm inbox access")
    group.add_argument(
        "--check-inbox", metavar="ID", dest="check_inbox", help="Poll inbox and print the first verification code found"
    )
    group.add_argument("--retire", metavar="ID", help="Retire an account")
    group.add_argument(
        "--claim",
        action="store_true",
        help="Claim one AVAILABLE account (mark it USED and print credentials)",
    )

    parser.add_argument("--count", type=int, default=1, help="Number of accounts to create (with --create)")
    parser.add_argument("--sender", default=None, help="Filter emails by sender substring (with --check-inbox)")
    parser.add_argument("--subject", default=None, help="Filter emails by subject substring (with --check-inbox)")
    parser.add_argument(
        "--timeout", type=int, default=120, help="Seconds to wait for email (with --check-inbox, default 120)"
    )
    parser.add_argument("--headed", action="store_true", help="Run browser in visible mode")
    parser.add_argument("--debug", action="store_true", help="Save screenshots at each step")
    parser.add_argument(
        "--for", dest="used_for", default="", metavar="IG_ID", help="Instagram account ID (with --claim)"
    )

    args = parser.parse_args()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = redis_async.from_url(redis_url, decode_responses=True)
    mgr = ProtonAccountManager(r)

    try:
        if args.list:
            await cmd_list(mgr)

        elif args.show:
            await cmd_show(mgr, args.show)

        elif args.test:
            account = await mgr.get(args.test)
            if not account:
                print(f"Account {args.test} not found", file=sys.stderr)
                sys.exit(1)
            print(f"Testing login for {account.email}…")
            ok = await ProtonCreator(mgr, headless=False).test_login(account)
            print("Login OK — inbox confirmed." if ok else "Login FAILED — check logs/proton_debug/.")
            sys.exit(0 if ok else 1)

        elif args.check_inbox:
            account = await mgr.get(args.check_inbox)
            if not account:
                print(f"Account {args.check_inbox} not found", file=sys.stderr)
                sys.exit(1)
            print(f"Polling inbox for {account.email} (timeout={args.timeout}s)…")
            if args.sender:
                print(f"  sender filter: {args.sender}")
            if args.subject:
                print(f"  subject filter: {args.subject}")
            reader = ProtonMailReader(mgr, headless=False)
            code = await reader.get_verification_code(
                account,
                sender_contains=args.sender,
                subject_contains=args.subject,
                timeout=args.timeout,
            )
            if code:
                print(f"\nCode: {code}")
            else:
                print("No verification code found within timeout.", file=sys.stderr)
                sys.exit(1)

        elif args.retire:
            await mgr.retire(args.retire)
            print(f"Retired {args.retire}")

        elif args.claim:
            account = await mgr.claim_available(used_for=args.used_for)
            if not account:
                print("No AVAILABLE Proton accounts in pool.", file=sys.stderr)
                sys.exit(1)
            print(f"Email:    {account.email}")
            print("Password: [REDACTED]")
            print(f"ID:       {account.id}")

        elif args.create:
            count = max(1, args.count)
            creator = ProtonCreator(mgr, headless=not args.headed)
            created: list[str] = []
            failed = 0

            for i in range(count):
                if i > 0:
                    input(f"\n[{i}/{count}] Press Enter when ready for the next account…")

                print(f"\n[{i + 1}/{count}] Creating Proton account (headed={args.headed})…")
                try:
                    account = await creator.create_new()
                    created.append(f"  {account.id}  {account.email}")
                    print(f"  OK → {account.email}")
                except Exception as exc:
                    failed += 1
                    print(f"  FAILED: {exc}")

            print(f"\n{'─' * 60}")
            print(f"Done. {len(created)} created, {failed} failed.")
            if created:
                print("AVAILABLE accounts:")
                for line in created:
                    print(line)

    finally:
        await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
