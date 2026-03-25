---
name: osia-status
description: Query the live OSIA status API to get service health, metrics, and logs for debugging
---

# OSIA Status API Skill

Use this skill whenever you need to investigate a problem with the running OSIA system — checking if services are up, reading live logs, inspecting queue depth, or getting system metrics.

## Configuration

The API credentials are loaded from the local `.env` file (git-ignored). Read them before making any request:

```python
from dotenv import dotenv_values
env = dotenv_values(".env")
token = env.get("STATUS_API_TOKEN", "")
sentinel = env.get("STATUS_API_UA_SENTINEL", "osia-monitor/1")
```

Base URL: `https://status.osia.dev`

Every authenticated request needs:
- `Authorization: Bearer <STATUS_API_TOKEN>`
- `User-Agent: <STATUS_API_UA_SENTINEL>`

## Available Endpoints

### Full status snapshot
```
GET /status
```
Returns system metrics, all systemd service states, Docker container states, timers, and Redis queue depth in one call. Start here when diagnosing a general problem.

### Service states only
```
GET /status/services
```
Returns `active`/`inactive`/`failed` state, PID, memory usage, and start time for every OSIA systemd service.

### Docker container states
```
GET /status/containers
```
Returns running/stopped state for: osia-anythingllm, osia-qdrant, osia-redis, osia-signal, mailserver, osia-kali.

### Host system metrics
```
GET /status/system
```
Returns CPU load, memory usage, disk usage, CPU temperature, and GPU stats (if available).

### Redis queue
```
GET /status/redis
```
Returns Redis health and current task queue depth. Useful for checking if tasks are backing up.

### Qdrant knowledge base
```
GET /status/qdrant
```
Queries the Qdrant HTTP API directly (localhost:6333) and returns per-collection stats for all desk knowledge bases. Each collection entry includes `name`, `is_desk`, `points_count`, `vectors_count`, `segments_count`, and `status`. Also returns `total_points` across all collections.

### Service logs
```
GET /logs/{service}?lines=100
```
Tail journald logs for a specific service. `service` can omit the `.service` suffix.
Max 500 lines. Default 100.

Valid service names:
- `osia-orchestrator`
- `osia-signal-ingress`
- `osia-persona-daemon`
- `osia-rss-ingress`
- `osia-mcp-arxiv-bridge`
- `osia-mcp-phone-bridge`
- `osia-mcp-semantic-scholar-bridge`
- `osia-mcp-tavily-bridge`
- `osia-mcp-time-bridge`
- `osia-mcp-wikipedia-bridge`
- `osia-cyber-bridge`
- `osia-status-api`

## Example — fetch full status

```python
import httpx
from dotenv import dotenv_values

env = dotenv_values(".env")
token = env["STATUS_API_TOKEN"]
sentinel = env.get("STATUS_API_UA_SENTINEL", "osia-monitor/1")
headers = {"Authorization": f"Bearer {token}", "User-Agent": sentinel}

with httpx.Client() as client:
    r = client.get("https://status.osia.dev/status", headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
```

## Example — fetch logs for a failing service

```python
import httpx
from dotenv import dotenv_values

env = dotenv_values(".env")
headers = {
    "Authorization": f"Bearer {env['STATUS_API_TOKEN']}",
    "User-Agent": env.get("STATUS_API_UA_SENTINEL", "osia-monitor/1"),
}

with httpx.Client() as client:
    r = client.get(
        "https://status.osia.dev/logs/osia-orchestrator",
        headers=headers,
        params={"lines": 200},
        timeout=15,
    )
    r.raise_for_status()
    for line in r.json()["lines"]:
        print(line)
```

## Workflow for diagnosing a problem

1. Call `GET /status` to get a full picture — identify which services are down or degraded.
2. For any failed service, call `GET /logs/{service}?lines=200` to read recent logs.
3. If the orchestrator is running but tasks aren't processing, check `GET /status/redis` for queue depth.
4. If a Docker container is down, check `GET /status/containers` and cross-reference with `GET /logs/osia-orchestrator`.
5. Use system metrics from `GET /status/system` to rule out resource exhaustion (memory, disk, CPU temp on the Orange Pi).
