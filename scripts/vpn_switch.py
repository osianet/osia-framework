#!/usr/bin/env python3
"""
Switch the active Surfshark WireGuard exit node.

Usage:
    sudo python3 scripts/vpn_switch.py US          # random US server
    sudo python3 scripts/vpn_switch.py us-nyc      # specific server
    sudo python3 scripts/vpn_switch.py AU          # random AU server
    sudo python3 scripts/vpn_switch.py --list      # show available countries/servers
    sudo python3 scripts/vpn_switch.py --status    # show current endpoint

Requires sudo — modifies /etc/wireguard/wg0.conf and restarts wg-quick@wg0.
ZeroTier connectivity is unaffected (those subnets bypass the VPN tunnel).
"""

import argparse
import random
import re
import subprocess
import sys
import time
from pathlib import Path

WG_CONF = Path("/etc/wireguard/wg0.conf")
COUNTRIES_DIR = Path("/etc/wireguard/countries")


def list_servers() -> dict[str, list[str]]:
    """Return {country_code: [server_slug, ...]} from available configs."""
    mapping: dict[str, list[str]] = {}
    for conf in sorted(COUNTRIES_DIR.glob("*.conf")):
        slug = conf.stem
        cc = slug.split("-")[0].upper()
        mapping.setdefault(cc, []).append(slug)
    return mapping


def resolve_target(target: str) -> str:
    """Return a specific server slug given a country code or slug."""
    servers = list_servers()
    upper = target.upper()
    if upper in servers:
        return random.choice(servers[upper])
    # Check if it's a specific slug
    slug = target.lower()
    conf = COUNTRIES_DIR / f"{slug}.conf"
    if conf.exists():
        return slug
    print(f"ERROR: '{target}' is not a known country code or server slug.", file=sys.stderr)
    print("Run with --list to see available options.", file=sys.stderr)
    sys.exit(1)


def parse_peer_block(conf_path: Path) -> tuple[str, str]:
    """Extract (PublicKey, Endpoint) from a WireGuard config file."""
    text = conf_path.read_text()
    pub = re.search(r"^\s*PublicKey\s*=\s*(.+)$", text, re.MULTILINE)
    ep = re.search(r"^\s*Endpoint\s*=\s*(.+)$", text, re.MULTILINE)
    if not pub or not ep:
        print(f"ERROR: Could not parse [Peer] from {conf_path}", file=sys.stderr)
        sys.exit(1)
    return pub.group(1).strip(), ep.group(1).strip()


def parse_interface_block(conf_path: Path) -> str:
    """Extract the full [Interface] section (including comments and PostUp/PreDown)."""
    text = conf_path.read_text()
    # Everything up to (but not including) the [Peer] section
    peer_idx = text.find("[Peer]")
    if peer_idx == -1:
        print(f"ERROR: No [Peer] section found in {conf_path}", file=sys.stderr)
        sys.exit(1)
    return text[:peer_idx].rstrip()


def current_slug() -> str | None:
    """Return the slug (e.g. 'au-mel') matching the currently active wg0.conf peer, or None."""
    try:
        text = WG_CONF.read_text()
        ep_match = re.search(r"^\s*Endpoint\s*=\s*(\S+)", text, re.MULTILINE)
        if not ep_match:
            return None
        active_ep = ep_match.group(1).strip()
        for conf in COUNTRIES_DIR.glob("*.conf"):
            _, ep = parse_peer_block(conf)
            if ep == active_ep:
                return conf.stem
    except Exception:  # noqa: BLE001 — best-effort slug detection, failure is non-fatal
        pass
    return None


def current_endpoint() -> str | None:
    """Return the current WireGuard endpoint from wg show, or None."""
    try:
        result = subprocess.run(["wg", "show", "wg0", "endpoints"], capture_output=True, text=True, check=True)
        # Output: "<pubkey>  <ip>:<port>"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    except subprocess.CalledProcessError:  # wg0 not up or wg not installed
        pass
    return None


def switch(slug: str) -> None:
    conf_path = COUNTRIES_DIR / f"{slug}.conf"
    new_pubkey, new_endpoint = parse_peer_block(conf_path)
    interface_block = parse_interface_block(WG_CONF)

    new_conf = (
        f"{interface_block}\n\n[Peer]\nPublicKey = {new_pubkey}\nAllowedIPs = 0.0.0.0/0\nEndpoint = {new_endpoint}\n"
    )

    print(f"Switching to: {slug} ({new_endpoint})", flush=True)

    # Write new config
    WG_CONF.write_text(new_conf)

    # Restart the tunnel
    print("Bringing tunnel down...", flush=True)
    subprocess.run(["wg-quick", "down", "wg0"], check=True)
    time.sleep(1)

    print("Bringing tunnel up...", flush=True)
    subprocess.run(["wg-quick", "up", "wg0"], check=True)
    time.sleep(2)

    # Verify endpoint via wg show
    endpoint = current_endpoint()
    if endpoint:
        print(f"Active endpoint: {endpoint}", flush=True)
    else:
        print("WARNING: Could not verify new endpoint via 'wg show'", flush=True)

    # Connectivity check — force through wg0 interface to bypass the api-bypass routes
    result = subprocess.run(
        ["curl", "-s", "--max-time", "8", "--interface", "wg0", "https://ifconfig.me"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"VPN public IP: {result.stdout.strip()}", flush=True)
    else:
        print("WARNING: Connectivity check via wg0 failed — tunnel may need manual inspection", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch Surfshark WireGuard exit node")
    parser.add_argument("target", nargs="?", help="Country code (US, AU, UK) or server slug (us-nyc)")
    parser.add_argument("--list", action="store_true", help="List available countries and servers")
    parser.add_argument("--status", action="store_true", help="Show current active endpoint")
    parser.add_argument("--get-slug", action="store_true", help="Print current server slug and exit (for scripting)")
    args = parser.parse_args()

    if args.list:
        for cc, servers in sorted(list_servers().items()):
            print(f"  {cc}: {', '.join(servers)}")
        return

    if args.get_slug:
        slug = current_slug()
        if slug:
            print(slug)
        else:
            sys.exit(1)
        return

    if args.status:
        ep = current_endpoint()
        print(f"Current endpoint: {ep or 'unknown'}")
        slug = current_slug()
        if slug:
            print(f"Current slug:     {slug}")
        result = subprocess.run(
            ["curl", "-s", "--max-time", "8", "--interface", "wg0", "https://ifconfig.me"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"VPN public IP: {result.stdout.strip()}")
        return

    if not args.target:
        parser.print_help()
        sys.exit(1)

    slug = resolve_target(args.target)
    switch(slug)


if __name__ == "__main__":
    main()
