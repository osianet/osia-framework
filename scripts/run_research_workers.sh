#!/bin/bash
# Launches RESEARCH_WORKER_INSTANCES parallel research workers and waits for all to finish.
# Used as the ExecStart for osia-research-worker.service so the oneshot service
# tracks the bash process, which exits only after every worker subprocess completes.
set -e

N=${RESEARCH_WORKER_INSTANCES:-1}
PYTHON=/home/ubuntu/osia-framework/.venv/bin/python

echo "Starting $N research worker instance(s)"

pids=()
for i in $(seq 1 "$N"); do
    "$PYTHON" -m src.workers.research_worker &
    pids+=($!)
done

exit_code=0
for pid in "${pids[@]}"; do
    wait "$pid" || exit_code=$?
done

echo "All $N research worker instance(s) finished (exit code: $exit_code)"
exit $exit_code
