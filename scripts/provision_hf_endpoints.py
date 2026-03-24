"""
OSIA — HuggingFace Inference Endpoints Provisioner

Creates (or verifies) dedicated HuggingFace Inference Endpoints for uncensored
models used by OSIA intelligence desks. Endpoints are configured with scale-to-zero
so you only pay for active inference time.

Usage:
    uv run python scripts/provision_hf_endpoints.py              # provision all
    uv run python scripts/provision_hf_endpoints.py --status      # check status of all endpoints
    uv run python scripts/provision_hf_endpoints.py --pause        # pause all endpoints (stop billing)
    uv run python scripts/provision_hf_endpoints.py --resume       # resume all paused endpoints

Requires:
    - HF_TOKEN env var (write-scoped Hugging Face token)
    - huggingface_hub >= 0.21 (added to pyproject.toml)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("osia.hf_provision")

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_NAMESPACE = os.getenv("HF_NAMESPACE")  # your HF username or org

if not HF_TOKEN:
    logger.error("HF_TOKEN is not set in .env — need a write-scoped HuggingFace token.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Endpoint definitions — one per uncensored model we want available
# ---------------------------------------------------------------------------
# Each entry maps to a dedicated HF Inference Endpoint.
# The endpoint name becomes part of the URL:
#   https://<name>.<namespace>.endpoints.huggingface.cloud
#
# Adjust instance_type / instance_size based on model VRAM needs:
#   - 8B models  → nvidia-l4 x1 (24 GB, $0.80/hr)
#   - 70B models → nvidia-a100 x2 (160 GB, $5.00/hr)
#
# All endpoints use scale-to-zero: 0 replicas when idle = $0 cost.
# Cold start is ~2-5 min depending on model size.

ENDPOINTS = [
    {
        "name": "osia-dolphin-8b",
        "repository": "cognitivecomputations/Dolphin3.0-Llama3.1-8B",
        "task": "text-generation",
        "framework": "pytorch",
        "accelerator": "gpu",
        "vendor": "aws",
        "region": "us-east-1",
        "instance_type": "nvidia-l4",
        "instance_size": "x1",
        "type": "protected",
        "min_replica": 0,
        "max_replica": 1,
        "scale_to_zero_timeout": 10,  # minutes
        "custom_image": {
            "health_route": "/health",
            "url": "ghcr.io/huggingface/text-generation-inference:latest",
            "env": {
                "MODEL_ID": "/repository",
                "MAX_INPUT_LENGTH": "4096",
                "MAX_TOTAL_TOKENS": "8192",
                "MAX_BATCH_PREFILL_TOKENS": "4096",
            },
        },
        "description": "Dolphin 3.0 8B — uncensored general-purpose model for HUMINT desk fallback",
    },
    {
        "name": "osia-dolphin-70b",
        "repository": "cognitivecomputations/Dolphin3.0-Llama3.1-70B",
        "task": "text-generation",
        "framework": "pytorch",
        "accelerator": "gpu",
        "vendor": "aws",
        "region": "us-east-1",
        "instance_type": "nvidia-a100",
        "instance_size": "x2",
        "type": "protected",
        "min_replica": 0,
        "max_replica": 1,
        "scale_to_zero_timeout": 10,
        "custom_image": {
            "health_route": "/health",
            "url": "ghcr.io/huggingface/text-generation-inference:latest",
            "env": {
                "MODEL_ID": "/repository",
                "MAX_INPUT_LENGTH": "4096",
                "MAX_TOTAL_TOKENS": "8192",
                "MAX_BATCH_PREFILL_TOKENS": "4096",
            },
        },
        "description": "Dolphin 3.0 70B — primary uncensored model for HUMINT/Cyber/Cultural desks",
    },
]



def get_client():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def provision_endpoint(api, spec: dict):
    """Create an endpoint if it doesn't already exist, or report its current state."""
    from huggingface_hub import create_inference_endpoint, get_inference_endpoint
    from huggingface_hub.utils import HfHubHTTPError

    name = spec["name"]
    namespace = HF_NAMESPACE

    # Check if it already exists
    try:
        existing = get_inference_endpoint(name, namespace=namespace, token=HF_TOKEN)
        logger.info(
            "Endpoint '%s' already exists — status: %s, url: %s",
            name, existing.status, existing.url,
        )
        return existing
    except HfHubHTTPError:
        pass  # doesn't exist yet, create it

    logger.info("Creating endpoint '%s' with model '%s'...", name, spec["repository"])

    endpoint = create_inference_endpoint(
        name=name,
        repository=spec["repository"],
        framework=spec["framework"],
        task=spec["task"],
        accelerator=spec["accelerator"],
        vendor=spec["vendor"],
        region=spec["region"],
        type=spec["type"],
        instance_size=spec["instance_size"],
        instance_type=spec["instance_type"],
        namespace=namespace,
        custom_image=spec.get("custom_image"),
        min_replica=spec.get("min_replica", 0),
        max_replica=spec.get("max_replica", 1),
        scale_to_zero_timeout=spec.get("scale_to_zero_timeout", 15),
        token=HF_TOKEN,
    )

    logger.info(
        "Endpoint '%s' created — status: %s. It will initialize and then scale to zero when idle.",
        name, endpoint.status,
    )
    return endpoint


def show_status():
    """Print status of all OSIA endpoints."""
    from huggingface_hub import get_inference_endpoint
    from huggingface_hub.utils import HfHubHTTPError

    for spec in ENDPOINTS:
        try:
            ep = get_inference_endpoint(spec["name"], namespace=HF_NAMESPACE, token=HF_TOKEN)
            print(f"  {spec['name']:30s}  status={ep.status:15s}  url={ep.url or '(not ready)'}")
        except HfHubHTTPError:
            print(f"  {spec['name']:30s}  NOT PROVISIONED")


def pause_all():
    """Pause all OSIA endpoints (stops billing immediately)."""
    from huggingface_hub import get_inference_endpoint
    from huggingface_hub.utils import HfHubHTTPError

    for spec in ENDPOINTS:
        try:
            ep = get_inference_endpoint(spec["name"], namespace=HF_NAMESPACE, token=HF_TOKEN)
            if ep.status not in ("paused",):
                ep.pause(token=HF_TOKEN)
                logger.info("Paused endpoint '%s'", spec["name"])
            else:
                logger.info("Endpoint '%s' already paused", spec["name"])
        except HfHubHTTPError:
            logger.warning("Endpoint '%s' not found — skipping", spec["name"])


def resume_all():
    """Resume all paused OSIA endpoints."""
    from huggingface_hub import get_inference_endpoint
    from huggingface_hub.utils import HfHubHTTPError

    for spec in ENDPOINTS:
        try:
            ep = get_inference_endpoint(spec["name"], namespace=HF_NAMESPACE, token=HF_TOKEN)
            if ep.status == "paused":
                ep.resume(token=HF_TOKEN)
                logger.info("Resumed endpoint '%s'", spec["name"])
            else:
                logger.info("Endpoint '%s' status is '%s' — not paused", spec["name"], ep.status)
        except HfHubHTTPError:
            logger.warning("Endpoint '%s' not found — skipping", spec["name"])


def main():
    parser = argparse.ArgumentParser(description="OSIA HuggingFace Inference Endpoints manager")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="Show status of all endpoints")
    group.add_argument("--pause", action="store_true", help="Pause all endpoints")
    group.add_argument("--resume", action="store_true", help="Resume all paused endpoints")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.pause:
        pause_all()
    elif args.resume:
        resume_all()
    else:
        api = get_client()
        for spec in ENDPOINTS:
            try:
                provision_endpoint(api, spec)
            except Exception as e:
                logger.error("Failed to provision '%s': %s", spec["name"], e)


if __name__ == "__main__":
    main()
