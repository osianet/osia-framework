# AnythingLLM System Configuration

This document provides a summary of the current system and workspace configurations for the OSIA AnythingLLM instance. **All API keys and sensitive credentials have been excluded.**

## System Settings

### General
*   **Storage Directory:** `/app/server/storage`
*   **Multi-User Mode:** Disabled
*   **Authentication Required:** False

### Vector Database
*   **Vector DB Provider:** Qdrant
*   **Qdrant Endpoint:** `http://172.20.0.1:6333`
*   **Data Partitioning:** We explicitly assign a `vectorTag` to each workspace. This isolates embedding collections inside Qdrant for external Python script integration.

### Embeddings
*   **Embedding Engine:** Native
*   **Embedding Model:** `Xenova/all-MiniLM-L6-v2`

### Global LLM Providers
*   **Default LLM Provider:** Gemini
*   **Default LLM Model:** `gemini-3-flash`
*   **Ollama Endpoint:** `http://ollama.osia.dev:11434`
*   **Generic OpenAI Base Path (NPU):** `https://rkllm.osia.dev/v1`
*   **Configured API Key Integrations:** OpenAI, Gemini, Anthropic, HuggingFace, Tavily (Agent)

---

## Workspace (Desk) Configurations
*(Prompts are now dynamically generated and synchronized using `scripts/update_prompts.py`)*

### 1. OSIA Intelligence Hub
*   **Slug:** `my-workspace`

### 2. Collection Directorate
*   **Slug:** `collection-directorate`
*   **Vector Tag:** `collection_raw`
*   **Chat Provider:** Generic OpenAI (`Pleias-RAG-350M`)

### 3. Geopolitical & Security Desk
*   **Slug:** `geopolitical-and-security-desk`
*   **Vector Tag:** `geopolitical_intel`
*   **Chat Provider:** Gemini (`gemini-3-flash`)

### 4. Cultural & Theological Intelligence Desk
*   **Slug:** `cultural-and-theological-intelligence-desk`
*   **Vector Tag:** `cultural_intel`
*   **Chat Provider:** Gemini (`gemini-3-flash`)

### 5. Science, Technology & Commercial Desk
*   **Slug:** `science-technology-and-commercial-desk`
*   **Vector Tag:** `science_intel`
*   **Chat Provider:** Anthropic (`claude-sonnet-4-6`)

### 6. Human Intelligence & Profiling Desk
*   **Slug:** `human-intelligence-and-profiling-desk`
*   **Vector Tag:** `human_intel`
*   **Chat Provider:** Ollama (`nchapman/dolphin3.0-llama3:latest`)

### 7. The Watch Floor
*   **Slug:** `the-watch-floor`
*   **Vector Tag:** `watch_floor`
*   **Chat Provider:** Gemini (`gemini-3.1-pro-preview`)

### 8. Finance & Economics Directorate
*   **Slug:** `finance-and-economics-directorate`
*   **Vector Tag:** `finance_intel`
*   **Chat Provider:** OpenAI (`gpt-5.4-mini`)

### 9. Cyber Intelligence & Warfare Desk
*   **Slug:** `cyber-intelligence-and-warfare-desk`
*   **Vector Tag:** `cyber_intel`
*   **Chat Provider:** Anthropic (`claude-sonnet-4-6`)

---

## Custom Agent Skills
Custom tools located in `/home/ubuntu/osia-knowledge-base/plugins/agent-skills/` to provide the AnythingLLM agents with external capabilities:
*   **`osia-cyber-ip-intel`**: IP Geolocation & ASN lookup via `ip-api.com`. Used primarily by the Cyber Desk.
*   **`osia-finance-stock-intel`**: Real-time ticker price & market data via Yahoo Finance. Used by the Finance Desk.
*   **`osia-stash-writer`**: Allows agents to write synchronized intelligence reports directly to the host filesystem (`osia_shared_stash.txt`).

---

## Management Scripts
Located in `/home/ubuntu/osia-framework/scripts/`
*   **`update_prompts.py`**: A master configuration sync script. It uses Gemini to merge the Core Mandate (`DIRECTIVES.md`) with desk-specific templates (`templates/prompts/*.txt`). It automatically connects to the AnythingLLM API to push the new system prompts, configure the correct LLM models per desk, ensure custom skills are active, and define the custom Qdrant `vectorTags`.