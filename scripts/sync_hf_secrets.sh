#!/bin/bash
# Syncs required HF Jobs secrets from .env to the HuggingFace dataset repo.
# Uses the huggingface_hub Python library directly (no CLI needed).
#
# Usage:
#   ./scripts/sync_hf_secrets.sh
#
# Requires: uv sync (huggingface-hub must be in the project dependencies)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}ERROR: .env not found at $ENV_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}Syncing HF Jobs secrets from .env...${NC}"
echo ""

uv run python - <<'EOF'
import os
import sys
from pathlib import Path

# Load .env manually so we don't need python-dotenv in this context
env_file = Path(__file__).resolve().parent.parent / ".env"
env = {}
for line in env_file.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, val = line.partition("=")
    val = val.split("#")[0].strip()  # strip inline comments
    env[key.strip()] = val

HF_TOKEN = env.get("HF_TOKEN", "")
HF_NAMESPACE = env.get("HF_NAMESPACE", "")

if not HF_TOKEN or "your_" in HF_TOKEN:
    print("ERROR: HF_TOKEN not set in .env")
    sys.exit(1)
if not HF_NAMESPACE or "your_" in HF_NAMESPACE:
    print("ERROR: HF_NAMESPACE not set in .env")
    sys.exit(1)

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

api = HfApi(token=HF_TOKEN)
repo_id = f"{HF_NAMESPACE}/osia-jobs"

# Ensure repo exists
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"  \033[0;32mOK\033[0m    repo: {repo_id}")
except Exception as e:
    print(f"  \033[0;31mFAIL\033[0m  repo creation: {e}")
    sys.exit(1)

# Secrets to sync
SECRETS = [
    "QUEUE_API_TOKEN",
    "QUEUE_API_UA_SENTINEL",
    "QDRANT_API_KEY",
    "HF_TOKEN",
    "TAVILY_API_KEY",
    "HF_ENDPOINT_DOLPHIN_24B",
    "HF_ENDPOINT_HERMES_70B",
]

ok = skipped = failed = 0

for secret in SECRETS:
    value = env.get(secret, "")
    if not value or "your_" in value:
        print(f"  \033[1;33mSKIP\033[0m  {secret} (not set in .env)")
        skipped += 1
        continue
    try:
        api.add_space_secret(repo_id=repo_id, key=secret, value=value, repo_type="dataset")
        print(f"  \033[0;32mOK\033[0m    {secret}")
        ok += 1
    except Exception as e:
        print(f"  \033[0;31mFAIL\033[0m  {secret}: {e}")
        failed += 1

print(f"\nDone: {ok} synced, {skipped} skipped, {failed} failed")
if failed:
    sys.exit(1)
EOF
