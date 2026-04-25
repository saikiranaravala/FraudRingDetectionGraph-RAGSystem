"""
Fraud Ring Detection — FastAPI Web Service

Exposes the Phase 3 GraphRAG pipeline as a REST API.

Endpoints
─────────
  GET  /health                          Liveness probe
  POST /explain/{claim_id}              Full GraphRAG investigation brief
  POST /query                           Natural language graph query
  POST /feedback/{claim_id}             Record investigator decision
  GET  /stats                           Feedback history + vector store stats

Start locally:
  uvicorn api:app --reload --port 8000

Environment variables (all read from .env):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
  OPENROUTER_API_KEY, OPENROUTER_MODEL
  VECTOR_STORE_BACKEND
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# Log LangSmith tracing status on startup
import os as _os
if _os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
    log.info("LangSmith tracing ENABLED — traces will be sent to LangSmith dashboard")
else:
    log.info("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")

app = FastAPI(
    title="Fraud Ring Detection API",
    description="GraphRAG + LangGraph fraud investigation pipeline",
    version="3.0.0",
)


# ── Lazy singletons (initialised on first request) ────────────────────

_retriever    = None
_vector_store = None
_embedder     = None
_llm_client   = None
_nl_engine    = None
_feedback_store = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.services.graph_retriever import GraphRetriever
        _retriever = GraphRetriever()
    return _retriever


def _get_pipeline_deps():
    global _vector_store, _embedder, _llm_client
    if _vector_store is None:
        from src.services.vector_store import get_vector_store
        from src.utils.embedder import FraudEmbedder
        from openai import OpenAI
        from src.utils import config as C
        import os

        _vector_store = get_vector_store(load_existing=True)
        _embedder     = FraudEmbedder()

        # Pass Google API key via header if using Gemma (for user's own quota)
        headers = {}
        google_key = os.getenv("GOOGLE_API_KEY")
        if "gemma" in C.LLM_MODEL.lower() and google_key:
            log.info("Using Gemma with user's Google API key")
            headers["HTTP-Referer"] = "https://fraud-ring-detection.local"
            headers["X-Title"] = "Fraud Ring Detection"
            # OpenRouter uses x-api-key header for provider keys
            headers["x-api-key"] = google_key

        _llm_client = OpenAI(
            api_key=C.OPENROUTER_API_KEY,
            base_url=C.OPENROUTER_BASE_URL,
            default_headers=headers if headers else None
        )
        log.info("LLM client initialized (model: %s)", C.LLM_MODEL)
    return _vector_store, _embedder, _llm_client


def _get_nl_engine():
    global _nl_engine
    if _nl_engine is None:
        from src.tools.nl_query import NLQueryEngine
        retriever = _get_retriever()
        _, _, llm = _get_pipeline_deps()
        _nl_engine = NLQueryEngine(retriever=retriever, llm_client=llm)
    return _nl_engine


def _get_feedback_store():
    global _feedback_store
    if _feedback_store is None:
        from src.services.feedback import FeedbackStore
        _feedback_store = FeedbackStore(retriever=_get_retriever())
    return _feedback_store


# ── Request / Response models ─────────────────────────────────────────

class ExplainResponse(BaseModel):
    claim_id:          str
    fraud_score:       float
    override_triggers: list[str]
    analogous_rings:   list[dict]
    reasoning_trace:   str
    error:             Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class QueryResponse(BaseModel):
    question: str
    answer:   str


class FeedbackRequest(BaseModel):
    decision:          Literal["Approve", "Dismiss", "Escalate"]
    investigator_id:   str = Field(..., min_length=1, max_length=64)
    feedback_to_model: Literal["Correct", "FP", "FN", "Uncertain"] = "Correct"
    override_reason:   str = ""
    confidence:        float = Field(default=1.0, ge=0.0, le=1.0)


class FeedbackResponse(BaseModel):
    claim_id:   str
    decision:   str
    recorded:   bool


class StatsResponse(BaseModel):
    total_reviews:     int
    last_review:       Optional[str]
    last_retrain:      Optional[str]
    f1_history:        list[dict]
    vector_store_size: int


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint. Returns status OK if API is running.

    Returns:
        dict: Status dictionary with "status": "ok"
    """
    return {"status": "ok"}


@app.post("/explain/{claim_id}", response_model=ExplainResponse)
def explain(claim_id: str):
    """
    Run the full GraphRAG pipeline for a claim.
    Returns the Claude investigation brief + override triggers + analogous rings.
    """
    try:
        from src.agent.pipeline import run_pipeline
        retriever = _get_retriever()
        vs, emb, llm = _get_pipeline_deps()

        result = run_pipeline(
            claim_id=claim_id,
            retriever=retriever,
            vector_store=vs,
            embedder=emb,
            llm_client=llm,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("explain failed for %s", claim_id)
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])

    return ExplainResponse(
        claim_id=claim_id,
        fraud_score=result.get("fraud_score", 0.0),
        override_triggers=result.get("override_triggers", []),
        analogous_rings=result.get("analogous_rings", []),
        reasoning_trace=result.get("reasoning_trace", ""),
    )


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    """
    Convert a natural language question to Cypher, execute it, and return a summary.
    All generated Cypher is validated as read-only before execution.
    """
    try:
        engine = _get_nl_engine()
        answer = engine.query(body.question)
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("query failed: %s", body.question)
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(question=body.question, answer=answer)


@app.post("/feedback/{claim_id}", response_model=FeedbackResponse)
def feedback(claim_id: str, body: FeedbackRequest):
    """
    Record an investigator decision for a claim.
    Writes a HumanReview node to Neo4j linked to the claim's InvestigationCase.
    """
    try:
        store = _get_feedback_store()
        store.record(
            claim_id=claim_id,
            decision=body.decision,
            investigator_id=body.investigator_id,
            feedback_to_model=body.feedback_to_model,
            override_reason=body.override_reason,
            confidence=body.confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("feedback failed for %s", claim_id)
        raise HTTPException(status_code=500, detail=str(e))

    return FeedbackResponse(claim_id=claim_id, decision=body.decision, recorded=True)


@app.get("/stats", response_model=StatsResponse)
def stats():
    """Return feedback history and vector store entry count."""
    try:
        store = _get_feedback_store()
        meta  = store._meta

        vs_size = 0
        try:
            vs, _, _ = _get_pipeline_deps()
            vs_size = len(vs)
        except Exception:
            pass

        return StatsResponse(
            total_reviews=meta.get("total_reviews", 0),
            last_review=meta.get("last_review"),
            last_retrain=meta.get("last_train_date"),
            f1_history=meta.get("f1_history", []),
            vector_store_size=vs_size,
        )
    except Exception as e:
        log.exception("stats failed")
        raise HTTPException(status_code=500, detail=str(e))
