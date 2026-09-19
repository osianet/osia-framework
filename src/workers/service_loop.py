"""Long-running service wrapper for OSIA's batch workers.

The research / hermes / profile-investigation workers were written as
*batch* jobs: their ``main()`` coroutine drains the queue (or walks the
wiki) once and returns. Historically systemd timers re-fired them on a
schedule.

For the containerized deployment we want them as **long-running services**
instead of oneshot timer units. Rather than rewrite each worker's batch
logic, this module wraps the existing entrypoint in a supervised loop:

    run batch  ->  sleep <interval>  ->  run batch  ->  ...

with exponential backoff if a batch raises, so a transient failure (e.g. a
provider 5xx) doesn't turn into a tight crash-loop.

Usage (as a container ``command:``)::

    python -m src.workers.service_loop research_worker:main --interval 900
    python -m src.workers.service_loop hermes_worker:main --interval 86400
    python -m src.workers.service_loop profile_investigation_worker:_main --interval 21600

The batch entrypoint is given as ``module:callable`` relative to
``src.workers``. The interval may also come from ``$WORKER_INTERVAL_SECS``
(the ``--interval`` flag wins when both are set).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable

logger = logging.getLogger("osia.service_loop")

# Backoff bounds for a failing batch (seconds).
_BACKOFF_START = 15
_BACKOFF_MAX = 600


def _resolve_entrypoint(spec: str) -> Callable[[], Awaitable[None]]:
    """Resolve ``module:callable`` (relative to src.workers) to a coroutine fn."""
    if ":" not in spec:
        raise SystemExit(f"entrypoint must be 'module:callable', got: {spec!r}")
    mod_name, _, attr = spec.partition(":")
    module = importlib.import_module(f"src.workers.{mod_name}")
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise SystemExit(f"src.workers.{mod_name} has no callable '{attr}'")
    return fn


async def _run_loop(entrypoint: Callable[[], Awaitable[None]], interval: int, name: str) -> None:
    stop = asyncio.Event()

    def _request_stop() -> None:
        logger.info("[%s] shutdown signal received — stopping after current batch", name)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError):
            # Signal handlers may be unavailable on some platforms; SIGTERM
            # from `docker stop` still terminates the process cleanly.
            pass

    backoff = _BACKOFF_START
    while not stop.is_set():
        try:
            logger.info("[%s] starting batch", name)
            await entrypoint()
            backoff = _BACKOFF_START  # reset after a clean batch
            logger.info("[%s] batch complete — sleeping %ds", name, interval)
            sleep_for = interval
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] batch raised — backing off %ds", name, backoff)
            sleep_for = backoff
            backoff = min(backoff * 2, _BACKOFF_MAX)

        # Interruptible sleep: wake immediately on shutdown.
        try:
            await asyncio.wait_for(stop.wait(), timeout=sleep_for)
        except TimeoutError:
            pass

    logger.info("[%s] loop exited cleanly", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OSIA batch worker as a long-running service.")
    parser.add_argument("entrypoint", help="Batch entrypoint as 'module:callable' relative to src.workers")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Seconds to sleep between batches (default: $WORKER_INTERVAL_SECS or 900)",
    )
    args = parser.parse_args()

    interval = args.interval
    if interval is None:
        interval = int(os.getenv("WORKER_INTERVAL_SECS", "900"))
    if interval < 1:
        raise SystemExit("--interval must be >= 1 second")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    name = args.entrypoint.partition(":")[0]
    entrypoint = _resolve_entrypoint(args.entrypoint)

    # Some batch entrypoints (e.g. hermes_worker.main) parse sys.argv themselves
    # via argparse. Neutralize argv to just the module name so they don't choke
    # on the wrapper's own flags (the entrypoint + --interval).
    sys.argv = [name]

    logger.info("service_loop supervising '%s' every %ds", args.entrypoint, interval)
    asyncio.run(_run_loop(entrypoint, interval, name))


if __name__ == "__main__":
    main()
