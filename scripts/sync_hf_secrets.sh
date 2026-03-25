#!/bin/bash
# Syncs required HF Jobs secrets from .env to the HuggingFace dataset repo.
# Run once after initial setup, or whenever secrets rotate.
#
# Usage:
#   ./scripts/sync_hf_secrets.sh
#
# Requires: huggingface-cli (installed via uv sync)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

# ANSI colours
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}ERROR: .env not found at $ENV_FILE${NC}"
    exit 1
fi

# Load .env (strip comments and blank lines)
_env_val() {
    grep -E "^${1}=" "$ENV_FILE" | head -1 | cut -d'=' -f2- | sed 's/[[:space:]]*#.*//' | xargs
}

HF_TOKEN=$(_env_val "HF_TOKEN")
HF_NAMESPACE=$(_env_val "HF_NAMESPACE")

if [ -z "$HF_TOKEN" ] || [[ "$HF_TOKEN" == *"your_"* ]]; then
    echo -e "${RED}ERROR: HF_TOKEN not set in .env${NC}"
    exit 1
fi
if [ -z "$HF_NAMESPACE" ] || [[ "$HF_NAMESPACE" == *"your_"* ]]; then
    echo -e "${RED}ERROR: HF_NAMESPACE not set in .env${NC}"
    exit 1
fi

REPO="${HF_NAMESPACE}/osia-jobs"
HF_CLI="$SCRIPT_DIR/../.venv/bin/huggingface-cli"

if [ ! -f "$HF_CLI" ]; then
    echo -e "${RED}ERROR: huggingface-cli not found — run: uv sync${NC}"
    exit 1
fi

echo -e "${YELLOW}Syncing secrets to HF dataset repo: ${REPO}${NC}"
echo ""

# Secrets to sync: ENV_VAR_NAME -> HF secret name (same name)
SECRETS=(
    "QUEUE_API_TOKEN"
    "QUEUE_API_UA_SENTINEL"
    "QDRANT_API_KEY"
    "HF_TOKEN"
    "TAVILY_API_KEY"
    "HF_ENDPOINT_DOLPHIN_24B"
    "HF_ENDPOINT_HERMES_70B"
)

ok=0
skipped=0
failed=0

for secret in "${SECRETS[@]}"; do
    value=$(_env_val "$secret")

    if [ -z "$value" ] || [[ "$value" == *"your_"* ]]; then
        echo -e "  ${YELLOW}SKIP${NC}  $secret (not set in .env)"
        skipped=$((skipped + 1))
        continue
    fi

    if HF_TOKEN="$HF_TOKEN" "$HF_CLI" secret set "$secret" \
        --repo "$REPO" \
        --repo-type dataset \
        --value "$value" 2>/dev/null; then
        echo -e "  ${GREEN}OK${NC}    $secret"
        ok=$((ok + 1))
    else
        echo -e "  ${RED}FAIL${NC}  $secret"
        failed=$((failed + 1))
    fi
done

echo ""
echo -e "Done: ${GREEN}${ok} synced${NC}, ${YELLOW}${skipped} skipped${NC}, ${RED}${failed} failed${NC}"

if [ "$failed" -gt 0 ]; then
    echo -e "${YELLOW}Tip: make sure the repo exists — run: huggingface-cli repo create osia-jobs --type dataset --private${NC}"
    exit 1
fi
