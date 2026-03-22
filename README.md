# OSIA Framework (Open Source Intelligence Agency)

An event-driven, multi-agent intelligence orchestration framework designed to automate the collection, analysis, and reporting of open-source intelligence.

## Architecture

OSIA follows a decoupled, microservice-based architecture:

- **Ingress Gateways:** (Input) Specialized listeners for tasking orders via Signal, Email (IMAP), and scheduled jobs (Cron).
- **Task Queue (Redis):** (Nervous System) A central message bus for managing asynchronous intelligence tasks.
- **The Orchestrator:** (Chief of Staff) The central logic engine that develops collection plans and tasks specialized desks.
- **Intelligence Desks (AnythingLLM):** (Analysis) Specialized analytical environments (Geopolitical, Tech, Human Intel, Cultural) for refined synthesis.
- **Egress Gateways:** (Output) Secure delivery of Intelligence Summaries (INTSUM) via Signal and Email.

## Tech Stack

- **Platform:** Linux (Optimized for ARM64/Orange Pi 5 Plus)
- **Engine:** Python 3.12 (Managed by `uv`)
- **Intelligence:** Google Gemini (Cloud) & RK3588 NPU (Local)
- **Storage:** Qdrant (Vector Database) & AnythingLLM (Context Management)
- **Infrastructure:** Docker & Redis
