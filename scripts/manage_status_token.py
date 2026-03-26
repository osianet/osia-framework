#!/usr/bin/env python3
"""
OSIA Status API — token management utility.

Usage:
    uv run python scripts/manage_status_token.py           # show current token
    uv run python scripts/manage_status_token.py --rotate  # generate and write a new token
    uv run python scripts/manage_status_token.py --ua      # show the UA sentinel
"""

import argparse
import re
import secrets
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


def _read_env() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    result = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def _write_key(key: str, value: str) -> None:
    """Update or append a key=value pair in .env."""
    if not ENV_FILE.exists():
        print(f"ERROR: {ENV_FILE} not found. Copy .env.example first.", file=sys.stderr)
        sys.exit(1)

    content = ENV_FILE.read_text()
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)

    if pattern.search(content):
        content = pattern.sub(f"{key}={value}", content)
    else:
        content = content.rstrip("\n") + f"\n{key}={value}\n"

    ENV_FILE.write_text(content)


def main():
    parser = argparse.ArgumentParser(description="Manage the OSIA Status API token")
    parser.add_argument("--rotate", action="store_true", help="Generate and save a new token")
    parser.add_argument("--ua", action="store_true", help="Show the User-Agent sentinel")
    args = parser.parse_args()

    env = _read_env()

    if args.ua:
        sentinel = env.get("STATUS_API_UA_SENTINEL", "osia-monitor/1")
        print(f"UA sentinel: {sentinel}")
        return

    if args.rotate:
        new_token = secrets.token_urlsafe(32)
        _write_key("STATUS_API_TOKEN", new_token)
        print(f"New token written to {ENV_FILE}")
        print(f"Token: {new_token}")
        print()
        print("Restart the status API service to pick up the new token:")
        print("  sudo systemctl restart osia-status-api.service")
        return

    # Default: show current token
    token = env.get("STATUS_API_TOKEN", "")
    if not token:
        print("STATUS_API_TOKEN is not set in .env")
        print("Run with --rotate to generate one.")
    else:
        print(f"Current token: {token}")
        sentinel = env.get("STATUS_API_UA_SENTINEL", "osia-monitor/1")
        port = env.get("STATUS_API_PORT", "8099")
        print(f"UA sentinel:   {sentinel}")
        print(f"Port:          {port}")
        print()
        print("Example curl:")
        print(f'  curl -s -H "Authorization: Bearer {token}" \\')
        print(f'       -H "User-Agent: {sentinel}" \\')
        print(f'       http://localhost:{port}/status | python3 -m json.tool')


if __name__ == "__main__":
    main()
