---
name: Fraud Ring Detection — Project State
description: All phases 1-3b complete; current implementations (LangSmith, NL query improvements)
type: project
originSessionId: 61524b88-846a-4d26-9715-c826e91fc72a
---
## Rollout Status (as of 2026-04-25)

| Phase | Scope | Status |
| ----- | ----- | ------ |
| Phase 1 | Neo4j graph + rule engine + HITL queue | ✓ Complete |
| Phase 2 | GraphSAGE + HINormer + XGB/RF/LGBM ensemble | ✓ Complete |
| Phase 3 | LangGraph pipeline + NL query + feedback loop | ✓ Complete |
| Phase 3b | Streamlit UI + FastAPI on Render | ✓ Complete |
| Phase 4 | Real-time Kinesis ingest + cross-carrier exchange | Not started |

**Graph scale:** 14,292 nodes · 28,690 edges · 941 properties · 24 node types · 28 edge types

## Recent Implementations (2026-04-24/25)

### LangSmith Monitoring
- Integrated LangSmith for LLM pipeline observability
- Automatic tracing when `LANGCHAIN_TRACING_V2=true`
- Traces LangGraph nodes, LLM calls, state transitions to smith.langchain.com
- Free tier: unlimited traces, 7-day retention
- Implementation: config.py, api.py, requirements.txt, full docs in CLAUDE.md/README.md/PRD.md

### Improved NL Query Examples
- **Problem:** Original few-shot examples referenced data patterns that don't exist (witnesses in 3+ claims)
- **Solution:** Updated to match actual graph data patterns via Cypher exploration
- **Result:** All example queries now return results (lawyers in 100-150 claims, fraud rings, shared bank accounts, high-risk claims)
- **Files:** src/tools/nl_query.py, ui/streamlit_app.py

## Next Work

Phase 4 streaming (real-time Kinesis + cross-carrier exchange) and SageMaker ML serving.
