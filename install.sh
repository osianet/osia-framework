#!/usr/bin/env bash
# OSIA Framework — Install Script
# Installs and enables all systemd services for the current user and directory.
#
# Usage:
#   ./install.sh              # install with auto-detected user and path
#   ./install.sh --start      # install and immediately start all services
#   ./install.sh --uninstall  # disable and remove all installed services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"
INSTALL_USER="${SUDO_USER:-$(whoami)}"
INSTALL_DIR="$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[osia]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC} $*"; }
error()   { echo -e "${RED}[error]${NC} $*" >&2; }

# ── Services to install ───────────────────────────────────────────────────────
# Persistent daemons (enabled + optionally started)
SERVICES=(
    osia-qdrant
    osia-adb-server
    osia-mcp-wikipedia-bridge
    osia-mcp-arxiv-bridge
    osia-mcp-tavily-bridge
    osia-mcp-time-bridge
    osia-mcp-semantic-scholar-bridge
    osia-queue-api
    osia-status-api
    osia-orchestrator
    osia-signal-ingress
    osia-persona-daemon
)

# Timer units (enable the timer; the .service fires automatically)
TIMERS=(
    osia-rss-ingress
    osia-research-worker
    osia-daily-sitrep
    osia-iran-israel-monitor
)

# ── Argument parsing ──────────────────────────────────────────────────────────
DO_START=false
DO_UNINSTALL=false
for arg in "$@"; do
    case "$arg" in
        --start)      DO_START=true ;;
        --uninstall)  DO_UNINSTALL=true ;;
        --help|-h)
            echo "Usage: $0 [--start] [--uninstall]"
            echo "  --start      Enable and start all services immediately"
            echo "  --uninstall  Disable and remove all installed service files"
            exit 0 ;;
        *) error "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Privilege check ───────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "This script must be run with sudo:"
    error "  sudo ./install.sh"
    exit 1
fi

# ── Uninstall ─────────────────────────────────────────────────────────────────
if $DO_UNINSTALL; then
    info "Uninstalling OSIA services..."
    ALL_UNITS=()
    for svc in "${SERVICES[@]}"; do ALL_UNITS+=("${svc}.service"); done
    for tmr in "${TIMERS[@]}"; do ALL_UNITS+=("${tmr}.timer" "${tmr}.service"); done

    for unit in "${ALL_UNITS[@]}"; do
        if systemctl is-enabled --quiet "$unit" 2>/dev/null; then
            systemctl disable --now "$unit" 2>/dev/null || true
            info "  disabled $unit"
        fi
        if [[ -f "$SYSTEMD_DIR/$unit" ]]; then
            rm -f "$SYSTEMD_DIR/$unit"
            info "  removed $SYSTEMD_DIR/$unit"
        fi
    done
    systemctl daemon-reload
    info "Uninstall complete."
    exit 0
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Installing OSIA Framework services"
info "  Install directory : $INSTALL_DIR"
info "  Service user      : $INSTALL_USER"
echo

if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    if [[ -f "$INSTALL_DIR/.env.example" ]]; then
        warn ".env not found — copying .env.example to .env"
        cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
        warn "Edit $INSTALL_DIR/.env and add your API keys before starting services."
    else
        warn ".env not found. Create $INSTALL_DIR/.env with your API keys before starting."
    fi
fi

if ! command -v docker &>/dev/null; then
    warn "docker not found — osia-qdrant will fail to start. Install Docker first."
fi

if ! command -v adb &>/dev/null; then
    warn "adb not found — ADB services will fail. Install android-tools-adb."
fi

if ! groups "$INSTALL_USER" | grep -qw plugdev; then
    info "Adding $INSTALL_USER to plugdev group (required for ADB USB access)..."
    usermod -aG plugdev "$INSTALL_USER"
    warn "plugdev group added — $INSTALL_USER must log out and back in for it to take full effect."
fi

if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
    warn ".venv not found — run 'uv sync' in $INSTALL_DIR before starting services."
fi

# ── Install service files ─────────────────────────────────────────────────────
install_unit() {
    local name="$1"   # e.g. osia-orchestrator.service
    local src="$INSTALL_DIR/systemd/$name"

    if [[ ! -f "$src" ]]; then
        warn "  skipping $name (not found in systemd/)"
        return
    fi

    local dest="$SYSTEMD_DIR/$name"

    # Substitute hardcoded paths and user with actual install values
    sed \
        -e "s|/home/ubuntu/osia-framework|$INSTALL_DIR|g" \
        -e "s|User=ubuntu|User=$INSTALL_USER|g" \
        -e "s|Group=ubuntu|Group=$INSTALL_USER|g" \
        "$src" > "$dest"

    chmod 644 "$dest"
    info "  installed $name"
}

echo "Installing unit files..."
for svc in "${SERVICES[@]}"; do
    install_unit "${svc}.service"
done
for tmr in "${TIMERS[@]}"; do
    install_unit "${tmr}.service"
    install_unit "${tmr}.timer"
done

# ── Reload and enable ─────────────────────────────────────────────────────────
echo
info "Reloading systemd daemon..."
systemctl daemon-reload

echo
echo "Enabling persistent services..."
for svc in "${SERVICES[@]}"; do
    if [[ -f "$SYSTEMD_DIR/${svc}.service" ]]; then
        systemctl enable "${svc}.service"
        info "  enabled ${svc}.service"
    fi
done

echo
echo "Enabling timers..."
for tmr in "${TIMERS[@]}"; do
    if [[ -f "$SYSTEMD_DIR/${tmr}.timer" ]]; then
        systemctl enable "${tmr}.timer"
        info "  enabled ${tmr}.timer"
    fi
done

# ── Optionally start everything ───────────────────────────────────────────────
if $DO_START; then
    echo
    info "Starting services..."
    for svc in "${SERVICES[@]}"; do
        if [[ -f "$SYSTEMD_DIR/${svc}.service" ]]; then
            systemctl start "${svc}.service" && info "  started ${svc}.service" \
                || warn "  ${svc}.service failed to start (check: journalctl -u ${svc})"
        fi
    done
    echo
    info "Starting timers..."
    for tmr in "${TIMERS[@]}"; do
        if [[ -f "$SYSTEMD_DIR/${tmr}.timer" ]]; then
            systemctl start "${tmr}.timer" && info "  started ${tmr}.timer" \
                || warn "  ${tmr}.timer failed to start"
        fi
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
info "Installation complete."
if ! $DO_START; then
    echo
    echo "  To start all services now, run:"
    echo "    sudo ./install.sh --start"
    echo
    echo "  Or start individually:"
    echo "    sudo systemctl start osia-orchestrator"
fi
echo
echo "  Useful commands:"
echo "    journalctl -u osia-orchestrator -f     # orchestrator logs"
echo "    journalctl -u osia-signal-ingress -f   # signal gateway logs"
echo "    systemctl status osia-research-worker.timer"
