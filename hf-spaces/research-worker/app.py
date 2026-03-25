"""
OSIA Research Worker — HuggingFace Gradio Space entry point.

The worker loop runs in a background thread. The Gradio UI provides
a live status dashboard so the Space stays alive and you can monitor
job processing without SSH access.

Required Space secrets (set in HF Space Settings → Variables and secrets):
  QUEUE_API_URL          — https://queue.osia.dev
  QUEUE_API_TOKEN        — bearer token
  QUEUE_API_UA_SENTINEL  — osia-worker/1
  QDRANT_URL             — https://qdrant.osia.dev
  QDRANT_API_KEY         — qdrant api key
  GEMINI_API_KEY         — google gemini api key
  TAVILY_API_KEY         — tavily search api key
  HF_TOKEN               — huggingface token (for embeddings API)
"""

import threading
import time
import asyncio
from datetime import datetime, timezone
from collections import deque

import gradio as gr
from worker import worker_loop, QUEUE_API_URL, QDRANT_URL, GEMINI_MODEL

# ---------------------------------------------------------------------------
# Shared state — written by worker thread, read by Gradio UI
# ---------------------------------------------------------------------------

_log: deque[str] = deque(maxlen=200)
_stats = {
    "jobs_processed": 0,
    "jobs_skipped": 0,
    "chunks_stored": 0,
    "last_job": "—",
    "last_job_time": "—",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "status": "starting",
}
_lock = threading.Lock()


def _log_line(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with _lock:
        _log.appendleft(f"[{ts}] {msg}")


def _patch_worker_logging():
    """Intercept worker log output into our shared deque."""
    import logging

    class UIHandler(logging.Handler):
        def emit(self, record):
            _log_line(self.format(record))

    handler = UIHandler()
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s — %(message)s"))
    logging.getLogger("osia").addHandler(handler)


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------

def _run_worker():
    with _lock:
        _stats["status"] = "running"
    _log_line("Worker loop started.")
    try:
        asyncio.run(worker_loop())
    except Exception as e:
        _log_line(f"FATAL: worker crashed — {e}")
        with _lock:
            _stats["status"] = f"crashed: {e}"


def _start_worker():
    _patch_worker_logging()
    t = threading.Thread(target=_run_worker, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _get_status() -> tuple[str, str, str]:
    with _lock:
        s = dict(_stats)
        logs = "\n".join(list(_log)[:50])

    status_md = f"""
**Status:** `{s['status']}`  
**Started:** {s['started_at']}  
**Queue API:** `{QUEUE_API_URL}`  
**Qdrant:** `{QDRANT_URL}`  
**Model:** `{GEMINI_MODEL}`
"""
    metrics_md = f"""
| Metric | Value |
|--------|-------|
| Jobs processed | {s['jobs_processed']} |
| Jobs skipped (dedup) | {s['jobs_skipped']} |
| Chunks stored | {s['chunks_stored']} |
| Last job | {s['last_job']} |
| Last job time | {s['last_job_time']} |
"""
    return status_md, metrics_md, logs


with gr.Blocks(title="OSIA Research Worker", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# OSIA Research Worker")
    gr.Markdown("Polls `osia:research_queue`, runs multi-source research loops, stores results in Qdrant.")

    with gr.Row():
        with gr.Column(scale=1):
            status_box = gr.Markdown()
        with gr.Column(scale=1):
            metrics_box = gr.Markdown()

    log_box = gr.Textbox(
        label="Worker Log (last 50 lines)",
        lines=20,
        max_lines=20,
        interactive=False,
    )

    refresh_btn = gr.Button("Refresh", variant="secondary")

    def refresh():
        return _get_status()

    refresh_btn.click(refresh, outputs=[status_box, metrics_box, log_box])

    # Auto-refresh every 10s
    demo.load(refresh, outputs=[status_box, metrics_box, log_box], every=10)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

_start_worker()

if __name__ == "__main__":
    demo.launch()
