"""
OSIA HuggingFace Jobs Toolkit — status, logs, and troubleshooting.

Usage:
  uv run python scripts/hf_jobs.py list              # list all jobs with status
  uv run python scripts/hf_jobs.py status [JOB_ID]   # detailed status (latest if no ID)
  uv run python scripts/hf_jobs.py logs [JOB_ID]     # stream/dump filtered logs
  uv run python scripts/hf_jobs.py logs --raw [ID]   # dump all log lines
  uv run python scripts/hf_jobs.py endpoints         # check HF endpoint configs & status
  uv run python scripts/hf_jobs.py cancel JOB_ID     # cancel a running job
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_NAMESPACE = os.getenv("HF_NAMESPACE", "")
ORG = "osianet"

ANSI = {
    "green": "\033[0;32m",
    "red": "\033[0;31m",
    "yellow": "\033[1;33m",
    "cyan": "\033[0;36m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}

STAGE_COLOR = {
    "COMPLETED": "green",
    "RUNNING": "cyan",
    "PENDING": "yellow",
    "ERROR": "red",
    "CANCELLED": "dim",
}

# Log lines containing these strings are shown by default (non-raw mode)
LOG_KEYWORDS = [
    "ERROR",
    "WARNING",
    "WARN",
    "=== OSIA",
    "Batch complete",
    "Endpoint ready",
    "Endpoint HTTP",
    "Endpoint '",
    "Researching:",
    "Research done",
    "Research failed",
    "Queue depth",
    "Drained",
    "Job complete",
    "body:",
]


def _hf(path: str, method: str = "GET", body: dict | None = None) -> dict | list:
    url = f"https://huggingface.co/api{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _endpoint_api(path: str, method: str = "GET", body: dict | None = None) -> dict:
    url = f"https://api.endpoints.huggingface.cloud/v2{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _c(text: str, color: str) -> str:
    return f"{ANSI.get(color, '')}{text}{ANSI['reset']}"


def _age(iso: str) -> str:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    mins = int((datetime.now(datetime.UTC) - dt).total_seconds() // 60)
    if mins < 60:
        return f"{mins}m ago"
    h, m = divmod(mins, 60)
    return f"{h}h{m:02d}m ago"


def _get_jobs() -> list[dict]:
    jobs = _hf(f"/jobs/{ORG}")
    return sorted(jobs, key=lambda j: j["createdAt"], reverse=True)


def _latest_job(jobs: list[dict] | None = None) -> dict:
    jobs = jobs or _get_jobs()
    if not jobs:
        print(_c("No jobs found.", "yellow"))
        sys.exit(0)
    return jobs[0]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(args):
    jobs = _get_jobs()
    if not jobs:
        print(_c("No jobs found.", "yellow"))
        return

    print(f"\n{'ID':<26}  {'STATUS':<12}  {'AGE':<12}  {'CREATED'}")
    print("─" * 72)
    for j in jobs:
        stage = j["status"]["stage"]
        color = STAGE_COLOR.get(stage, "reset")
        msg = j["status"].get("message") or ""
        # Trim long message
        if len(msg) > 50:
            msg = msg[:47] + "..."
        age = _age(j["createdAt"])
        print(f"{j['id']:<26}  {_c(f'{stage:<12}', color)}  {age:<12}  {j['createdAt'][:19].replace('T', ' ')}")
        if msg and stage == "ERROR":
            print(f"  {_c('↳ ' + msg, 'red')}")
    print()


def cmd_status(args):
    jobs = _get_jobs()
    job = next((j for j in jobs if j["id"] == args.job_id), None) if args.job_id else _latest_job(jobs)
    if args.job_id and not job:
        print(_c(f"Job {args.job_id} not found.", "red"))
        sys.exit(1)

    stage = job["status"]["stage"]
    color = STAGE_COLOR.get(stage, "reset")
    age = _age(job["createdAt"])

    print(f"\n{_c('Job ID:', 'bold')}      {job['id']}")
    print(f"{_c('Status:', 'bold')}      {_c(stage, color)}")
    print(f"{_c('Created:', 'bold')}     {job['createdAt'][:19].replace('T', ' ')} UTC  ({age})")
    print(f"{_c('Org:', 'bold')}         {job['owner']['name']}")
    print(f"{_c('Image:', 'bold')}       {job.get('dockerImage', '?')}")
    print(f"{_c('Flavor:', 'bold')}      {job.get('flavor', '?')}")
    timeout_s = job.get("timeout", 0)
    print(f"{_c('Timeout:', 'bold')}     {timeout_s // 3600}h{(timeout_s % 3600) // 60:02d}m")

    env = job.get("environment", {})
    safe_env = {k: v for k, v in env.items() if "TOKEN" not in k and "KEY" not in k and "SECRET" not in k}
    if safe_env:
        print(f"{_c('Env:', 'bold')}         {json.dumps(safe_env, separators=(',', ':'))}")

    msg = job["status"].get("message")
    if msg:
        print(f"{_c('Message:', 'bold')}     {_c(msg, color)}")

    print(f"\n{_c('Monitor:', 'bold')}     https://huggingface.co/{ORG}/jobs/{job['id']}")
    print()


def cmd_logs(args):
    jobs = _get_jobs()
    job = next((j for j in jobs if j["id"] == args.job_id), None) if args.job_id else _latest_job(jobs)
    if args.job_id and not job:
        print(_c(f"Job {args.job_id} not found.", "red"))
        sys.exit(1)

    job_id = job["id"]
    stage = job["status"]["stage"]
    color = STAGE_COLOR.get(stage, "reset")
    print(f"\n{_c('Logs for job:', 'bold')} {job_id}  [{_c(stage, color)}]")

    if stage == "RUNNING":
        print(_c("Job is live — streaming until it ends or Ctrl-C...\n", "dim"))

    url = f"https://huggingface.co/api/jobs/{ORG}/{job_id}/logs"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})

    shown = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    msg = json.loads(line[6:])["data"]
                except (json.JSONDecodeError, KeyError):
                    continue

                if args.raw or any(k in msg for k in LOG_KEYWORDS):
                    # Colour ERROR/WARNING lines
                    if "ERROR" in msg:
                        print(_c(msg, "red"))
                    elif "WARNING" in msg or "WARN" in msg:
                        print(_c(msg, "yellow"))
                    elif "Research done" in msg or "Job complete" in msg or "Batch complete" in msg:
                        print(_c(msg, "green"))
                    else:
                        print(msg)
                    shown += 1
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(_c(f"\nStream ended: {e}", "dim"))

    print(f"\n{_c(f'({shown} lines shown)', 'dim')}\n")


def cmd_endpoints(args):
    endpoint_names = ["osia-dolphin-r1-24b", "osia-hermes-70b"]
    ns = HF_NAMESPACE or ORG

    print(f"\n{_c('HuggingFace Inference Endpoints', 'bold')}\n")
    for name in endpoint_names:
        try:
            ep = _endpoint_api(f"/endpoint/{ns}/{name}")
        except Exception as e:
            print(f"  {_c(name, 'bold')}: {_c(f'ERROR — {e}', 'red')}")
            continue

        state = ep.get("status", {}).get("state", "?")
        color = {
            "running": "green",
            "scaledToZero": "dim",
            "initializing": "yellow",
            "pending": "yellow",
            "error": "red",
        }.get(state, "reset")
        model = ep.get("model", {})
        env = model.get("env", {})
        compute = ep.get("compute", {})

        print(f"  {_c(name, 'bold')}")
        print(f"    Status:  {_c(state, color)}")
        print(f"    Model:   {model.get('repository', '?')}")
        print(f"    Compute: {compute.get('instanceType', '?')} × {compute.get('instanceSize', '?')}")
        print(f"    Env:     {env if env else _c('(none set)', 'red')}")
        if not env.get("ENABLE_AUTO_TOOL_CHOICE"):
            print(f"    {_c('⚠ ENABLE_AUTO_TOOL_CHOICE not set — tool calling will 422', 'red')}")
        url = ep.get("status", {}).get("url", "")
        if url:
            print(f"    URL:     {url}")
        print()


def cmd_cancel(args):
    if not args.job_id:
        print(_c("Usage: hf_jobs.py cancel JOB_ID", "red"))
        sys.exit(1)
    try:
        _hf(f"/jobs/{ORG}/{args.job_id}/cancel", method="POST")
        print(_c(f"Cancelled job {args.job_id}", "green"))
    except Exception as e:
        print(_c(f"Cancel failed: {e}", "red"))
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if not HF_TOKEN:
        print(_c("HF_TOKEN not set in .env", "red"))
        sys.exit(1)

    parser = argparse.ArgumentParser(description="OSIA HuggingFace Jobs Toolkit")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all jobs")

    p_status = sub.add_parser("status", help="Detailed status for a job")
    p_status.add_argument("job_id", nargs="?", help="Job ID (default: latest)")

    p_logs = sub.add_parser("logs", help="Stream/dump job logs (filtered by default)")
    p_logs.add_argument("job_id", nargs="?", help="Job ID (default: latest)")
    p_logs.add_argument("--raw", action="store_true", help="Show all log lines, not just key events")

    sub.add_parser("endpoints", help="Check HF inference endpoint configs and status")

    p_cancel = sub.add_parser("cancel", help="Cancel a running job")
    p_cancel.add_argument("job_id", help="Job ID to cancel")

    args = parser.parse_args()

    dispatch = {
        "list": cmd_list,
        "status": cmd_status,
        "logs": cmd_logs,
        "endpoints": cmd_endpoints,
        "cancel": cmd_cancel,
    }

    if args.cmd not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
