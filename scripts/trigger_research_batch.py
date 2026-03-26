"""
OSIA Research Batch Trigger — runs on the Pi via systemd timer.

Checks the research queue depth. If it meets the batch threshold,
fires an HF Job to run research_batch.py on HuggingFace infrastructure.

The job script is fetched at runtime from the HF dataset repo
(BadIdeasRory/osia-jobs), which is kept in sync with main via
GitHub Actions (.github/workflows/sync-hf-jobs.yml).

Usage:
  uv run python scripts/trigger_research_batch.py
  uv run python scripts/trigger_research_batch.py --force   # skip threshold check
"""

import argparse
import logging
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("osia.trigger")

QUEUE_API_URL = os.getenv("QUEUE_API_URL", "https://queue.osia.dev")
QUEUE_API_TOKEN = os.getenv("QUEUE_API_TOKEN", "")
QUEUE_API_UA = os.getenv("QUEUE_API_UA_SENTINEL", "osia-worker/1")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_NAMESPACE = os.getenv("HF_NAMESPACE", "")

# Minimum queue depth before we bother spinning up a job
BATCH_THRESHOLD = int(os.getenv("RESEARCH_BATCH_THRESHOLD", "3"))

# HF dataset repo where the batch script lives (synced via GitHub Actions)
HF_JOBS_REPO = "osianet/osia-jobs"
SCRIPT_URL = f"https://huggingface.co/datasets/{HF_JOBS_REPO}/resolve/main/research_batch.py"

# HF Job config
JOB_FLAVOR = os.getenv("RESEARCH_JOB_FLAVOR", "cpu-basic")  # cheap — model runs on HF endpoint
JOB_TIMEOUT = os.getenv("RESEARCH_JOB_TIMEOUT", "2h")


def _get_queue_depth() -> int:
    headers = {
        "Authorization": f"Bearer {QUEUE_API_TOKEN}",
        "User-Agent": QUEUE_API_UA,
    }
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{QUEUE_API_URL}/queue/length",
            headers=headers,
            params={"queue": "osia:research_queue"},
        )
        resp.raise_for_status()
        return resp.json().get("depth", 0)


def _fire_hf_job() -> str:
    """Submit the research batch job to HuggingFace Jobs. Returns job ID."""
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # All secrets are passed inline from .env at submission time.
    # HF Jobs has no persistent secret storage — secrets must be provided per-run.
    # The Pi has all these values in .env, so we just pass them directly.
    env = {
        "QUEUE_API_URL": QUEUE_API_URL,
        "QUEUE_API_TOKEN": QUEUE_API_TOKEN,
        "QUEUE_API_UA_SENTINEL": QUEUE_API_UA,
        "QDRANT_URL": os.getenv("QDRANT_URL", "https://qdrant.osia.dev"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
        "HF_TOKEN": HF_TOKEN,
        "HF_NAMESPACE": HF_NAMESPACE,
        "HF_WAKE_TIMEOUT": os.getenv("HF_WAKE_TIMEOUT", "600"),
        "HF_ENDPOINT_DOLPHIN_24B": os.getenv("HF_ENDPOINT_DOLPHIN_24B", ""),
        "HF_ENDPOINT_HERMES_70B": os.getenv("HF_ENDPOINT_HERMES_70B", ""),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", ""),
        "UV_SCRIPT_URL": SCRIPT_URL,
    }

    # The job downloads research_batch.py from the private HF dataset repo
    # (synced from GitHub via Actions) and runs it with uv.
    job = api.run_job(
        image="ghcr.io/astral-sh/uv:python3.11-bookworm-slim",
        command=[
            "bash",
            "-c",
            (
                'python -c "'
                "import urllib.request, os; "
                "from pathlib import Path; "
                "o = urllib.request.build_opener(); "
                'o.addheaders = [(\\"Authorization\\", \\"Bearer \\" + os.environ[\\"HF_TOKEN\\"])]; '
                'Path(\\"/tmp/research_batch.py\\").write_bytes(o.open(os.environ[\\"UV_SCRIPT_URL\\"]).read())'
                '" && uv run --with httpx --with huggingface-hub /tmp/research_batch.py'
            ),
        ],
        flavor=JOB_FLAVOR,
        env=env,
        timeout=JOB_TIMEOUT,
        namespace="osianet",
    )

    return job.id


def main():
    parser = argparse.ArgumentParser(description="Trigger OSIA research batch job on HuggingFace")
    parser.add_argument("--force", action="store_true", help="Skip queue threshold check")
    args = parser.parse_args()

    if not QUEUE_API_TOKEN:
        logger.error("QUEUE_API_TOKEN not set")
        sys.exit(1)
    if not HF_TOKEN:
        logger.error("HF_TOKEN not set")
        sys.exit(1)
    if not HF_NAMESPACE:
        logger.error("HF_NAMESPACE not set")
        sys.exit(1)

    # Check queue depth — non-fatal: if the queue API is unreachable we fire anyway
    # rather than silently skipping research when the network is flaky.
    depth = None
    try:
        depth = _get_queue_depth()
        logger.info("Research queue depth: %d (threshold: %d)", depth, BATCH_THRESHOLD)
    except Exception as e:
        logger.warning("Could not check queue depth (%s) — proceeding anyway", e)

    if not args.force and depth is not None and depth < BATCH_THRESHOLD:
        logger.info("Queue below threshold — no job needed.")
        sys.exit(0)

    if depth is None:
        logger.info("Queue depth unknown — firing HF Job unconditionally...")
    else:
        logger.info("Threshold met (%d >= %d) — firing HF Job...", depth, BATCH_THRESHOLD)

    try:
        job_id = _fire_hf_job()
        logger.info("HF Job submitted: %s", job_id)
        logger.info("Monitor: https://huggingface.co/osianet/jobs/%s", job_id)
    except Exception as e:
        logger.error("Failed to submit HF Job: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
