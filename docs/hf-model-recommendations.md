# HuggingFace Model Recommendations for OSIA Desks

This document recommends specific open-source models for each OSIA intelligence desk,
focusing on models deployable via HuggingFace Inference Endpoints with scale-to-zero billing.

Recommendations are based on each desk's analytical requirements, censorship sensitivity,
tool-calling needs, and the GPU cost of running each model.

---

## Decision Framework

Not every desk needs an uncensored or self-hosted model. The key questions are:

1. **Does the desk's work hit censorship guardrails?** (profiling, offensive security, sensitive cultural topics)
2. **Does the desk need strong reasoning?** (financial modelling, geopolitical forecasting)
3. **Does the desk use agent tools?** (function calling reliability matters)
4. **Is the current cloud API model already doing a good job?**

If the answer to #1 is no and #4 is yes, there's no reason to move off the current provider.
Cloud APIs (Gemini, Claude, GPT-4o) are cheaper per-token than dedicated GPU endpoints for
desks that don't need uncensored output.

---

## Per-Desk Recommendations

### 1. Human Intelligence & Profiling Desk

**Current:** Ollama — `nchapman/dolphin3.0-llama3:latest` (local RTX 3080 Ti)
**Requirement:** Fully uncensored. Must profile individuals, track digital personas, analyze
behavioral patterns without refusals. Uses Username Recon tool.

**Recommended HF model:** `cognitivecomputations/Dolphin3.0-R1-Mistral-24B`
- Why: The R1 variant adds chain-of-thought reasoning on top of Dolphin's uncensored base.
  Profiling work benefits enormously from step-by-step deductive reasoning — connecting
  social media footprints, behavioral patterns, and network relationships.
- 24B fits on a single L4 GPU (24GB VRAM) at ~$0.80/hr. Much cheaper than the 70B on A100.
- Supports function calling and agentic workflows natively.
- The reasoning traces (think tags) give you auditable analytical chains in reports.

**Alternative:** `cognitivecomputations/Dolphin3.0-Llama3.1-8B` as a budget fallback
at the same L4 cost but faster inference. Less capable reasoning but still uncensored.

**GPU:** nvidia-l4 x1 ($0.80/hr active, $0 idle)

---

### 2. Cyber Intelligence & Warfare Desk

**Current:** Anthropic — `claude-sonnet-4-6`
**Requirement:** Must analyze offensive security topics, IoCs, network reconnaissance output,
and state-sponsored cyber operations. Claude will refuse to discuss exploit techniques,
offensive tooling analysis, and some reconnaissance methodologies.

**Recommended HF model:** `NousResearch/Hermes-3-Llama-3.1-70B`
- Why: Hermes 3 is the strongest open model for agentic tool-calling workflows. The Cyber
  desk relies heavily on its Kali Linux Sandbox tools (Nmap, Whois, Dig) — Hermes 3 was
  specifically trained for reliable function calling with structured tool schemas.
- Hermes 3 is not explicitly "uncensored" in the Dolphin sense, but it's significantly
  more permissive than Claude on security topics. It won't refuse to analyze exploit
  chains, discuss offensive techniques in an analytical context, or interpret Nmap output.
- 70B gives the analytical depth needed for complex infrastructure analysis.
- Benchmarks show it competitive with Llama 3.1 70B Instruct on general tasks, with
  superior agentic and multi-turn conversation performance.

**Alternative:** `perplexity-ai/r1-1776-distill-llama-70b`
- DeepSeek R1 reasoning distilled into Llama 70B, with CCP censorship explicitly removed.
- Exceptional at multi-step technical reasoning (MATH-500: 94.8%, MMLU: 88.4%).
- Better pure reasoning than Hermes 3, but weaker at tool calling.
- Consider this if the desk's work is more analytical than tool-driven.

**GPU:** nvidia-a100 x2 ($5.00/hr active, $0 idle) — 70B models need ~140GB VRAM

---

### 3. Cultural & Theological Intelligence Desk

**Current:** Gemini — `gemini-2.5-flash`
**Requirement:** Analyze religious movements, cultural narratives, sociological drivers.
Topics like religious extremism, sectarian violence, and cultural propaganda hit
guardrails on censored models that try to be "balanced" rather than analytical.

**Recommended HF model:** `cognitivecomputations/Dolphin3.0-R1-Mistral-24B`
- Why: Same model as HUMINT but for different reasons. Cultural analysis requires the
  model to engage frankly with religious ideologies, extremist narratives, and
  propaganda without hedging or adding disclaimers. Dolphin's uncensored training
  removes the "I should note that all religions have positive aspects" padding that
  Gemini/Claude inject.
- The R1 reasoning capability helps with the nuanced analytical work — tracing how
  theological positions map to political actions requires multi-step inference.
- 24B is more than sufficient for this desk's text-heavy analytical work.
- Can share the same endpoint as HUMINT (same model, same GPU).

**GPU:** nvidia-l4 x1 ($0.80/hr) — shares endpoint with HUMINT

---

### 4. Finance & Economics Directorate

**Current:** OpenAI — `gpt-4o`
**Recommendation:** Keep GPT-4o.

GPT-4o is excellent at financial analysis, numerical reasoning, and structured data
interpretation. The Finance desk's work (market dynamics, sanctions analysis, labor
economics) doesn't typically hit censorship walls — it's analytical rather than
sensitive. The Stock Market Intel tool works well with GPT-4o's function calling.

If you wanted to move off OpenAI for cost or sovereignty reasons, the best open
alternative would be `Qwen/Qwen3-32B`:
- Top-tier mathematical and financial reasoning among open models.
- Strong agentic/tool-calling capabilities.
- 32B fits on a single A100 80GB ($2.50/hr) or L40S 48GB ($1.80/hr).
- Apache 2.0 licensed.

But honestly, GPT-4o at ~$2.50/M input tokens is probably cheaper than running a
dedicated GPU endpoint for the volume of finance queries OSIA handles.

---

### 5. Geopolitical & Security Desk

**Current:** Gemini — `gemini-2.5-flash`
**Recommendation:** Keep Gemini 2.5 Flash.

Geopolitical analysis is Gemini's sweet spot — large context window, strong reasoning,
good at synthesizing multiple sources. The desk's Country Intel tool works reliably
with Gemini's function calling. Geopolitical topics rarely hit hard censorship walls
(models will discuss wars, coups, and power dynamics freely).

If you wanted a self-hosted option for data sovereignty, consider
`perplexity-ai/r1-1776-distill-llama-70b`:
- Explicitly de-censored for geopolitical topics (CCP censorship removed).
- Exceptional reasoning benchmarks.
- Would give you fully private geopolitical analysis with no data leaving your infra.
- But at $5/hr for 2x A100, it's only worth it if you're processing enough volume
  or have specific sovereignty requirements.

---

### 6. Science, Technology & Commercial Desk

**Current:** Anthropic — `claude-sonnet-4-6`
**Recommendation:** Keep Claude Sonnet.

Claude is genuinely excellent at technical analysis, code review, and scientific
reasoning. The Science desk evaluates software projects (GitHub Intel tool), analyzes
breakthroughs, and assesses technical accuracy — all areas where Claude excels.
It rarely refuses science/tech topics.

If you wanted to self-host, `Qwen/Qwen3-32B` would be the pick here too — strong
at code analysis and technical reasoning. But Claude is hard to beat for this use case.

---

### 7. Collection Directorate

**Current:** Generic OpenAI — `Pleias-RAG-350M` (local NPU)
**Recommendation:** Keep as-is, or upgrade to `Dolphin3.0-Llama3.1-8B`.

The Collection Directorate just ingests and organizes raw data — it doesn't analyze.
The tiny 350M model on the NPU is fine for this. If you find it struggling with
longer documents or more complex extraction, the 8B Dolphin on an L4 would be a
cheap upgrade, but it's probably not necessary.

---

### 8. The Watch Floor

**Current:** Gemini — `gemini-2.5-pro`
**Recommendation:** Keep Gemini 2.5 Pro.

The Watch Floor synthesizes all subordinate desk reports into a final INTSUM. This
requires the strongest possible model with the largest context window — it needs to
ingest multiple full reports and produce a coherent synthesis. Gemini 2.5 Pro with
its 1M+ token context is ideal for this. No open-source model currently matches
this context length at comparable quality.

---

## Proposed Endpoint Configuration

Based on the above, you only need two HF endpoints (not the original two):

| Endpoint Name | Model | GPU | Cost/hr | Desks Served |
|---|---|---|---|---|
| `osia-dolphin-r1-24b` | `cognitivecomputations/Dolphin3.0-R1-Mistral-24B` | nvidia-l4 x1 | $0.80 | HUMINT, Cultural |
| `osia-hermes-70b` | `NousResearch/Hermes-3-Llama-3.1-70B` | nvidia-a100 x2 | $5.00 | Cyber |

This is cheaper than the original plan (dropped the 70B Dolphin) and better matched
to each desk's actual needs. The R1 reasoning variant of Dolphin at 24B gives you
better analytical output than vanilla Dolphin 70B for profiling and cultural work,
at 1/6th the GPU cost.

The Cyber desk gets Hermes 3 70B because tool-calling reliability is critical there —
it's driving Nmap scans and interpreting live network data. Hermes 3 was purpose-built
for this kind of agentic workflow.

---

## Cost Estimates

Assuming each desk handles ~5 queries/day with an average of 10 minutes active
inference per query, and the 10-minute scale-to-zero timeout:

| Endpoint | Active time/day | Daily cost | Monthly cost |
|---|---|---|---|
| `osia-dolphin-r1-24b` | ~2.5 hrs (HUMINT + Cultural combined) | ~$2.00 | ~$60 |
| `osia-hermes-70b` | ~1.5 hrs (Cyber only) | ~$7.50 | ~$225 |
| **Total** | | | **~$285/mo** |

Compare to current cloud API costs for those same desks:
- HUMINT (Ollama/local): $0 but limited by RTX 3080 Ti VRAM (12GB, small models only)
- Cyber (Claude Sonnet): ~$3/M input + $15/M output tokens
- Cultural (Gemini Flash): ~$0.15/M input tokens

The HF endpoints are more expensive in raw dollars but give you uncensored output,
data sovereignty, and no rate limits. The local Ollama setup remains as a free
fallback if the HF endpoints are down.

---

## Migration Path

1. Provision the two new endpoints (update `scripts/provision_hf_endpoints.py`)
2. Update `DESK_ENDPOINT_MAP` in `src/desks/hf_endpoint_manager.py`
3. Use your Gemini MCP to reconfigure AnythingLLM desks:
   - HUMINT → Generic OpenAI pointing at `osia-dolphin-r1-24b`
   - Cultural → Generic OpenAI pointing at `osia-dolphin-r1-24b`
   - Cyber → Generic OpenAI pointing at `osia-hermes-70b`
4. Update `scripts/update_prompts.py` DESK_MODELS to reflect new providers
5. Test each desk with representative queries
6. Keep Ollama Dolphin as local fallback for HUMINT
