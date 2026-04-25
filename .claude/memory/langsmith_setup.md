---
name: LangSmith Monitoring Implementation
description: How LangSmith was integrated; required config; deployment notes
type: reference
originSessionId: 61524b88-846a-4d26-9715-c826e91fc72a
---
## Implementation Summary (2026-04-24)

LangSmith monitoring was added to trace LangGraph pipeline execution and LLM calls for debugging and performance monitoring.

## What Gets Traced

1. **LangGraph nodes:** retrieve_subgraph, find_analogous_rings, generate_reasoning
   - Input/output states, execution duration, errors

2. **LLM calls:** All OpenRouter (Claude/Gemma) API calls
   - Prompts, completions, token counts, costs, latency

3. **State transitions:** All state mutations between nodes

## Required Environment Variables

```env
LANGCHAIN_TRACING_V2=true                           # Enable tracing (set in .env)
LANGSMITH_API_KEY=lsv2_pt_...                      # Get from smith.langchain.com
LANGCHAIN_PROJECT=FraudRingDetectionGraph-RAG      # Dashboard project name
LANGSMITH_ENDPOINT=https://api.smith.langchain.com # API endpoint (default)
```

**Note:** When `LANGCHAIN_TRACING_V2=false` or env vars missing, LangChain silently disables tracing (no errors).

## Accessing Traces

1. Go to https://smith.langchain.com
2. Sign up (free tier)
3. Navigate to project "FraudRingDetectionGraph-RAG"
4. Each `/explain` or `/query` API call appears as a trace
5. Click trace to inspect nodes, LLM calls, errors, token usage

## Free Tier Limits

- **Unlimited traces** — no quota
- **7-day retention** — older traces auto-deleted
- **No advanced features** — feedback annotations, custom metrics require paid tier

## Files Modified

- `requirements.txt` — added `langsmith>=0.1.0`
- `src/utils/config.py` — loads LANGSMITH_* vars; validates if tracing enabled
- `api.py` — logs tracing status on startup ("LangSmith tracing ENABLED" or "disabled")
- `src/agent/pipeline.py` — docstring documents automatic tracing
- `CLAUDE.md` — comprehensive monitoring guide (setup, what's traced, free tier)
- `README.md` — setup instructions, external services table
- `PRD.md` — Technology Stack table, "Observability" row

## Deployment Checklist

**Local development:**
```bash
# Set in .env
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_...

# Verify on startup
uvicorn api:app --reload
# Should log: "LangSmith tracing ENABLED"
```

**Render production:**
1. Add env vars in Render dashboard (Settings → Environment)
2. Redeploy service
3. Check logs for "LangSmith tracing ENABLED"
4. View traces at smith.langchain.com dashboard

## How to Apply

When debugging slow API responses or unexpected LLM behavior, enable LangSmith tracing to see:
- Which node took longest (retrieve vs. embed vs. reason)
- What Claude actually generated vs. expected
- Token usage and costs
- Any exceptions in the pipeline

**Why:** Provides visibility into the black box of LLM reasoning without adding code instrumentation.
