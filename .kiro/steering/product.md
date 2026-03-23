# OSIA Framework — Product Overview

OSIA (Open Source Intelligence Agency) is an event-driven, multi-agent intelligence orchestration framework. It automates the collection, analysis, and reporting of open-source intelligence (OSINT).

## Core Concept

The framework mirrors a physical intelligence agency. Incoming requests arrive via Signal messenger or RSS feeds, get routed through a central Orchestrator ("Chief of Staff"), and are dispatched to specialized AI analysis desks hosted in isolated AnythingLLM workspaces. Finished reports are delivered back via Signal.

## Intelligence Desks

Each desk is an isolated AnythingLLM workspace with its own model and analytical focus:

- Collection Directorate — raw data acquisition and ingestion
- Geopolitical & Security Desk — statecraft, military, international relations
- Cultural & Theological Desk — sociological and religious drivers
- Science & Tech Desk — technical accuracy and breakthroughs
- Human Intelligence Desk — behavioral profiling and network mapping
- Finance & Economics Directorate — markets, sanctions, auditing
- The Watch Floor — final INTSUM synthesis

## Key Capabilities

- Signal-based encrypted messaging (input and output)
- Multi-turn research via MCP tools (Wikipedia, ArXiv, Semantic Scholar, Tavily, YouTube)
- Physical Android phone control via ADB for social media OSINT (Instagram, TikTok, Facebook, YouTube)
- Vision-driven social media agent using Gemini to interpret screenshots and drive UI actions
- RSS feed monitoring with AI summarization
- YouTube transcript extraction with 3-tier fallback (yt-dlp → MCP → physical capture)
- Automated daily SITREP generation via cron

## Analytical Mandate

All analysis follows the Socialist Intelligence Mandate defined in `DIRECTIVES.md`. Reports use materialist analysis and prioritize anti-imperialism, labor rights, and data sovereignty.
