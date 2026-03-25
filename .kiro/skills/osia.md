---
name: osia
description: Comprehensive development skill for the OSIA framework — understand the system, query live APIs, inspect knowledge bases, and diagnose issues
---

# OSIA Framework Development Skill

Use this skill whenever you need to understand, debug, or develop the OSIA (Open Source Intelligence Agency) framework. It gives you live access to all running services.

## What is OSIA?

OSIA is an event-driven, multi-agent intelligence orchestration framework running on an Orange Pi 5 Plus (ARM64). It mirrors a physical intelligence agency:

- Incoming requests arrive via **Signal messenger** or **RSS feeds**
- A central **Orchestrator** (Chief of Staff) routes tasks via a **Redis queue**
- Tasks are dispatched to specialized **AnythingLLM intelligence desks**
- Research is conducted via **MCP tools** (Wikipedia, ArXiv, Semantic Scholar, Tavily, YouTube)
- Reports are delivered back via **Signal**
- A **persona daemon** autonomously browses social media (Instagram, Facebook, YouTube) via ADB on a physical Moto g06 phone

## Credentials

Load from `.env` before making any request:

```python
from dotenv import dotenv_values
env = dotenv_values(".env")
```

Key variables:
- `QDRANT_API_KEY` — Qdrant auth header `api-key`
- `ANYTHINGLLM_API_KEY` — AnythingLLM Bearer token
- `STATUS_API_TOKEN` — Status API Bearer token
- `STATUS_API_UA_SENTINEL` — Required User-Agent string for status API

## API Endpoints

### 1. Qdrant — Vector Knowledge Base
**Base URL:** `https://qdrant.osia.dev`
**Auth:** `api-key: <QDRANT_API_KEY>` header

Qdrant is the persistent intelligence filing system. Each AnythingLLM desk has its own collection.

#### Collections (desk knowledge bases)
| Collection | Desk | Purpose |
|---|---|---|
| `collection-directorate` | Collection Directorate | Raw ingested intel |
| `geopolitical-and-security-desk` | Geopolitical & Security | Statecraft, conflict, military |
| `cultural-and-theological-intelligence-desk` | Cultural & Theological | Religion, sociology |
| `science-technology-and-commercial-desk` | Science & Tech | Technical breakthroughs |
| `human-intelligence-and-profiling-desk` | Human Intelligence | People profiles, networks |
| `finance-and-economics-directorate` | Finance & Economics | Markets, capital flows |
| `cyber-intelligence-and-warfare-desk` | Cyber Intelligence | Digital infrastructure |
| `the-watch-floor` | The Watch Floor | Final INTSUM synthesis |

#### Key endpoints

```python
import httpx, json
from dotenv import dotenv_values
env = dotenv_values(".env")
headers = {"api-key": env["QDRANT_API_KEY"]}

with httpx.Client(verify=False) as client:
    # List all collections with point counts
    r = client.get("https://qdrant.osia.dev/collections", headers=headers, timeout=10)
    collections = r.json()["result"]["collections"]

    # Get stats for a specific collection
    r = client.get("https://qdrant.osia.dev/collections/human-intelligence-and-profiling-desk", headers=headers, timeout=10)
    info = r.json()["result"]
    print(f"Points: {info['points_count']}, Vectors: {info['vectors_count']}")

    # Search a collection (semantic search)
    r = client.post(
        "https://qdrant.osia.dev/collections/human-intelligence-and-profiling-desk/points/search",
        headers={**headers, "Content-Type": "application/json"},
        json={"vector": [0.0] * 384, "limit": 5, "with_payload": True},  # replace vector with real embedding
        timeout=15,
    )

    # Scroll through points (browse without embedding)
    r = client.post(
        "https://qdrant.osia.dev/collections/collection-directorate/points/scroll",
        headers={**headers, "Content-Type": "application/json"},
        json={"limit": 10, "with_payload": True, "with_vector": False},
        timeout=15,
    )
    points = r.json()["result"]["points"]
```

### 2. AnythingLLM — Intelligence Desks
**Base URL:** `https://chat.osia.dev`
**Auth:** `Authorization: Bearer <ANYTHINGLLM_API_KEY>`

AnythingLLM hosts the intelligence desks as isolated workspaces. Each desk has its own LLM model, system prompt, and vector DB partition.

#### Desk slugs
- `collection-directorate`
- `geopolitical-and-security-desk`
- `cultural-and-theological-intelligence-desk`
- `science-technology-and-commercial-desk`
- `human-intelligence-and-profiling-desk`
- `finance-and-economics-directorate`
- `cyber-intelligence-and-warfare-desk`
- `the-watch-floor`

#### Key endpoints

```python
import httpx
from dotenv import dotenv_values
env = dotenv_values(".env")
headers = {
    "Authorization": f"Bearer {env['ANYTHINGLLM_API_KEY']}",
    "Content-Type": "application/json",
}

with httpx.Client(verify=False) as client:
    # List all workspaces
    r = client.get("https://chat.osia.dev/api/v1/workspaces", headers=headers, timeout=15)
    workspaces = r.json()["workspaces"]

    # Chat with a desk (use @agent prefix to trigger agent mode with custom skills)
    r = client.post(
        "https://chat.osia.dev/api/v1/workspace/human-intelligence-and-profiling-desk/chat",
        headers=headers,
        json={"message": "@agent What do we know about Elon Musk?", "mode": "chat"},
        timeout=120,
    )
    response = r.json()["textResponse"]

    # Ingest raw text into a workspace's vector DB
    r = client.post(
        "https://chat.osia.dev/api/v1/document/raw-text",
        headers=headers,
        json={
            "textContent": "Intel content here...",
            "addToWorkspaces": "human-intelligence-and-profiling-desk",
            "metadata": {"title": "Subject File: John Smith"},
        },
        timeout=60,
    )

    # Get workspace details (model, prompt, settings)
    r = client.get(
        "https://chat.osia.dev/api/v1/workspace/geopolitical-and-security-desk",
        headers=headers,
        timeout=15,
    )
```

### 3. Status API — System Health & Logs
**Base URL:** `https://status.osia.dev`
**Auth:** `Authorization: Bearer <STATUS_API_TOKEN>` + `User-Agent: <STATUS_API_UA_SENTINEL>`

```python
import httpx, json
from dotenv import dotenv_values
env = dotenv_values(".env")
headers = {
    "Authorization": f"Bearer {env['STATUS_API_TOKEN']}",
    "User-Agent": env.get("STATUS_API_UA_SENTINEL", "osia-monitor/1"),
}

with httpx.Client(verify=False) as client:
    # Full system snapshot (services, containers, redis, qdrant, system metrics)
    r = client.get("https://status.osia.dev/status", headers=headers, timeout=15)
    status = r.json()

    # Qdrant collection stats
    r = client.get("https://status.osia.dev/status/qdrant", headers=headers, timeout=15)

    # Redis queue depth
    r = client.get("https://status.osia.dev/status/redis", headers=headers, timeout=15)

    # Tail service logs (max 500 lines)
    r = client.get(
        "https://status.osia.dev/logs/osia-orchestrator",
        headers=headers,
        params={"lines": 100},
        timeout=15,
    )
    for line in r.json()["lines"]:
        print(line)
```

#### Available log services
`osia-orchestrator`, `osia-signal-ingress`, `osia-persona-daemon`, `osia-rss-ingress`,
`osia-mcp-arxiv-bridge`, `osia-mcp-phone-bridge`, `osia-mcp-semantic-scholar-bridge`,
`osia-mcp-tavily-bridge`, `osia-mcp-time-bridge`, `osia-mcp-wikipedia-bridge`,
`osia-cyber-bridge`, `osia-status-api`

## Diagnostic Workflow

### General health check
1. `GET /status` — identify any failed services or containers
2. For failed services → `GET /logs/{service}?lines=200`
3. Check `redis.queue_depth` — if > 0, tasks are backing up
4. Check `qdrant` section for collection point counts

### Investigating knowledge base state
```python
# Quick summary of all collections
with httpx.Client(verify=False) as client:
    r = client.get("https://qdrant.osia.dev/collections", headers={"api-key": env["QDRANT_API_KEY"]}, timeout=10)
    for c in r.json()["result"]["collections"]:
        info = client.get(f"https://qdrant.osia.dev/collections/{c['name']}", headers={"api-key": env["QDRANT_API_KEY"]}, timeout=10).json()["result"]
        print(f"{c['name']}: {info['points_count']} points")
```

### Browsing what's in a collection
```python
# Scroll through recent points without needing an embedding vector
r = client.post(
    "https://qdrant.osia.dev/collections/human-intelligence-and-profiling-desk/points/scroll",
    headers={"api-key": env["QDRANT_API_KEY"], "Content-Type": "application/json"},
    json={"limit": 20, "with_payload": True, "with_vector": False},
    timeout=15,
)
for point in r.json()["result"]["points"]:
    print(point["id"], point.get("payload", {}).get("title", "untitled"))
```

## Intelligence Desk Models (current config)
| Desk | LLM |
|---|---|
| Collection Directorate | Pleias-RAG-350M (Generic OpenAI) |
| Geopolitical & Security | gemini-3-flash |
| Cultural & Theological | gemini-3-flash |
| Science & Tech | claude-sonnet-4-6 |
| Human Intelligence | dolphin3.0-llama3 (Ollama — uncensored) |
| Finance & Economics | gpt-5.4-mini |
| Cyber Intelligence | claude-sonnet-4-6 |
| The Watch Floor | gemini-3.1-pro-preview |

## Key Architecture Notes

- **Redis queue** (`osia:task_queue`) — all work flows through here. Signal ingress and RSS ingress push tasks; orchestrator pops and processes.
- **Qdrant vector tags** — each desk uses its slug as the collection name. AnythingLLM partitions embeddings by `vectorTag`.
- **Embedding model** — `Xenova/all-MiniLM-L6-v2` (384 dimensions) — use this when constructing search vectors.
- **Persona daemon** — runs as `osia-persona-daemon.service`, controls a physical Android phone via ADB. Separate from the orchestrator.
- **MCP bridges** — each runs as a separate systemd service, exposing tools over SSE. The orchestrator connects to them via `MCPDispatcher`.
- **SSL** — all `*.osia.dev` endpoints use a self-signed cert. Always use `verify=False` when calling them from dev.
