#!/usr/bin/env bash
# Create one Instagram account with privilege separation.
#
# Privileged steps (VPN switch/restore) run via sudo.
# Browser automation + SMSPool + Redis run as the calling user — no root.
#
# Usage:
#   ./scripts/ig_create_account.sh [COUNTRY]
#   COUNTRY defaults to AU (any country code in /etc/wireguard/countries/)
#
# Examples:
#   ./scripts/ig_create_account.sh AU
#   ./scripts/ig_create_account.sh US
#   ./scripts/ig_create_account.sh UK

set -euo pipefail

COUNTRY="${1:-AU}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Capture current VPN slug before we touch anything
ORIGINAL_SLUG="$(sudo python3 "$SCRIPT_DIR/vpn_switch.py" --get-slug 2>/dev/null || true)"

# Restore VPN on exit (success, failure, or SIGINT/SIGTERM)
restore_vpn() {
    if [[ -n "$ORIGINAL_SLUG" ]]; then
        echo "[ig_create_account] Restoring VPN to $ORIGINAL_SLUG..."
        sudo python3 "$SCRIPT_DIR/vpn_switch.py" "$ORIGINAL_SLUG" || \
            echo "[ig_create_account] WARNING: VPN restore failed — check manually" >&2
    fi
}
trap restore_vpn EXIT

# Switch VPN to target country (privileged)
echo "[ig_create_account] Switching VPN to $COUNTRY..."
sudo python3 "$SCRIPT_DIR/vpn_switch.py" "$COUNTRY"

# Run account creation as the calling user (no root)
echo "[ig_create_account] Starting browser signup for country=$COUNTRY..."
cd "$REPO_DIR"
uv run python "$SCRIPT_DIR/ig_pool_admin.py" --create-account "$COUNTRY"
