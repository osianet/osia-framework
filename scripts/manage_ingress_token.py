#!/usr/bin/env python3
"""
OSIA Ingress API — token management utility.

The token is never printed to stdout. It is written directly to .env and
read from there by the service at startup.

Usage:
    uv run python scripts/manage_ingress_token.py           # show token status
    uv run python scripts/manage_ingress_token.py --rotate  # generate and write a new token
    uv run python scripts/manage_ingress_token.py --ua      # show the UA sentinel
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
    parser = argparse.ArgumentParser(description="Manage the OSIA Ingress API token")
    parser.add_argument("--rotate", action="store_true", help="Generate and save a new token")
    parser.add_argument("--ua", action="store_true", help="Show the User-Agent sentinel")
    args = parser.parse_args()

    env = _read_env()

    if args.ua:
        sentinel = env.get("INGRESS_API_UA_SENTINEL", "osia-ingress/1")
        print(f"UA sentinel: {sentinel}")
        return

    if args.rotate:
        _write_key("INGRESS_API_TOKEN", secrets.token_urlsafe(32))
        print(f"Token rotated and saved to {ENV_FILE}")
        print("Restart the ingress API service to apply:")
        print("  sudo systemctl restart osia-ingress-api.service")
        return

    # Default: show token status without revealing the value
    configured = bool(env.get("INGRESS_API_TOKEN", ""))
    sentinel = env.get("INGRESS_API_UA_SENTINEL", "osia-ingress/1")
    port = env.get("INGRESS_API_PORT", "8097")

    print(f"Token configured: {'yes' if configured else 'NO — run with --rotate to generate one'}")
    print(f"UA sentinel:      {sentinel}")
    print(f"Port:             {port}")
    if configured:
        print()
        print("Example curl — submit a query:")
        print("  curl -s -X POST https://<ingress-host>/ingest \\")
        print('       -H "Authorization: Bearer <token>" \\')
        print(f'       -H "User-Agent: {sentinel}" \\')
        print('       -H "Content-Type: application/json" \\')
        print('       -d \'{"query": "...", "label": "cli"}\'')
        print()
        print("Retrieve the token directly from .env:")
        print("  grep INGRESS_API_TOKEN .env")


if __name__ == "__main__":
    main()
