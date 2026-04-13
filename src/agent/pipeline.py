"""
GraphRAG Pipeline — LangGraph StateGraph

Three-node agentic pipeline per the PRD §4.3 (Layer 3):

  retrieve_subgraph
      Pull claim's direct neighbourhood from Neo4j (customer, lawyer,
      repair shop, witnesses, medical report, ring membership, network scores).
      ↓
  find_analogous_rings
      Embed the subgraph context and search the FraudRing vector knowledge
      base for the top-K historically similar rings.
      ↓
  generate_reasoning
      Send subgraph + analogous rings to Claude with a structured prompt.
      Returns a citable investigation brief the investigator can interrogate.

The compiled graph can be invoked with just a claim_id:

    pipeline = build_pipeline()
    result   = pipeline.invoke({"claim_id": "CLM-521585"})
    print(result["reasoning_trace"])

State keys
──────────
  claim_id          str       — input
  fraud_score       float     — from Claim.final_suspicion_score
  subgraph          dict      — raw Neo4j context
  subgraph_text     str       — human-readable subgraph for LLM
  analogous_rings   list      — [{id, score, metadata}, ...]
  override_triggers list[str] — OVR codes that fired
  reasoning_trace   str       — final Claude investigation brief
  error             str|None  — populated if any node fails
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from openai import OpenAI

from src.utils import config as C
from src.utils.embedder import FraudEmbedder, subgraph_to_text
from src.services.graph_retriever import GraphRetriever
from src.services.vector_store import VectorStore, get_vector_store

log = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END, START
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    log.warning(
        "langgraph not installed — pipeline will run in sequential mode. "
        "Install with: pip install langgraph"
    )


# ── Pipeline state ────────────────────────────────────────────────────
class PipelineState(TypedDict, total=False):
    claim_id:         str
    fraud_score:      float
    subgraph:         Dict[str, Any]
    subgraph_text:    str
    analogous_rings:  List[Dict]
    override_triggers: List[str]
    reasoning_trace:  str
    error:            Optional[str]


# ── Override trigger detection ────────────────────────────────────────
def _detect_overrides(claim_props: Dict, subgraph: Dict) -> List[str]:
    """Check Mandatory Override criteria (PRD §7) from live claim data."""
    triggers: list[str] = []
    yn = lambda v: str(v).strip().lower() in ("true", "1", "y", "yes")

    amount = 0.0
    try:
        amount = float(claim_props.get("total_claim_amount", 0))
    except (TypeError, ValueError):
        pass

    # OVR-001: ring suspicion score ≥ 0.90
    score = float(claim_props.get("final_suspicion_score",
                  claim_props.get("gnn_suspicion_score", 0)) or 0)
    if score >= 0.90:
        triggers.append(f"OVR-001 — ring suspicion score {score:.2f} ≥ 0.90")

    # OVR-002: high financial exposure
    if amount > 75_000:
        triggers.append(f"OVR-002 — claim amount ${amount:,.0f} > $75K")

    # OVR-003: cross-jurisdiction (lawyer + claims across states)
    lawyer = subgraph.get("lawyer", {})
    if lawyer and yn(lawyer.get("closed_loop_network_flag")):
        triggers.append("OVR-003 — closed-loop network lawyer detected")

    # OVR-005: attorney is primary connecting edge
    if lawyer:
        triggers.append(
            f"OVR-005 — licensed attorney {lawyer.get('full_name', '?')} "
            "is primary connecting edge"
        )

    # OVR-006: vulnerable claimant (proxy: language_assistance or elderly flag)
    customer = subgraph.get("customer", {})
    age = customer.get("age", 0)
    try:
        age = int(float(age))
    except (TypeError, ValueError):
        age = 0
    if age >= 65:
        triggers.append(f"OVR-006 — claimant age {age} ≥ 65 (potential vulnerable claimant)")

    # OVR-008: prior SIU referral on any node
    if yn(customer.get("siu_referral_flag")):
        triggers.append("OVR-008 — customer has prior SIU referral")
    shop = subgraph.get("repair_shop", {})
    if shop and int(shop.get("siu_referral_count", 0) or 0) > 0:
        triggers.append(
            f"OVR-008 — repair shop has {shop.get('siu_referral_count')} SIU referrals"
        )

    # Ring membership
    ring = subgraph.get("fraud_ring", {})
    if ring:
        triggers.append(
            f"RING-MEMBER — {ring.get('ring_id', '?')} "
            f"({ring.get('status', '?')}, score={ring.get('ring_score', 0)})"
        )

    return triggers


# ── Node functions ────────────────────────────────────────────────────
def make_retrieve_node(retriever: GraphRetriever):
    def retrieve_subgraph(state: PipelineState) -> PipelineState:
        claim_id = state["claim_id"]
        try:
            subgraph = retriever.get_claim_subgraph(claim_id)
            claim = subgraph.get("claim", {})
            fraud_score = float(
                claim.get("final_suspicion_score",
                claim.get("gnn_suspicion_score",
                claim.get("rag_confidence_score", 0.0))) or 0.0
            )
            overrides = _detect_overrides(claim, subgraph)
            return {
                **state,
                "subgraph":         subgraph,
                "subgraph_text":    subgraph_to_text(subgraph),
                "fraud_score":      fraud_score,
                "override_triggers": overrides,
                "error":            None,
            }
        except Exception as e:
            log.error("retrieve_subgraph failed for %s: %s", claim_id, e)
            return {**state, "subgraph": {}, "subgraph_text": "", "error": str(e)}
    return retrieve_subgraph


def make_search_node(embedder: FraudEmbedder, vector_store: VectorStore):
    def find_analogous_rings(state: PipelineState) -> PipelineState:
        subgraph = state.get("subgraph", {})
        if not subgraph:
            return {**state, "analogous_rings": []}
        try:
            query_vec = embedder.embed_subgraph(subgraph)
            results   = vector_store.search(query_vec, top_k=C.TOP_K_ANALOGOUS)
            return {**state, "analogous_rings": results}
        except Exception as e:
            log.warning("find_analogous_rings failed: %s", e)
            return {**state, "analogous_rings": []}
    return find_analogous_rings


def make_reasoning_node(llm_client: OpenAI):
    def generate_reasoning(state: PipelineState) -> PipelineState:
        claim_id      = state["claim_id"]
        fraud_score   = state.get("fraud_score", 0.0)
        subgraph_text = state.get("subgraph_text", "No context available.")
        analogous     = state.get("analogous_rings", [])
        overrides     = state.get("override_triggers", [])

        # Format analogous rings section
        if analogous:
            ana_lines = []
            for match in analogous:
                meta = match.get("metadata", {})
                ana_lines.append(
                    f"  • Ring {match['id']} (similarity={match['score']:.3f}): "
                    f"status={meta.get('status', '?')}, "
                    f"score={meta.get('ring_score', '?')}, "
                    f"members={meta.get('member_count', '?')}, "
                    f"closed_loop={meta.get('closed_loop_detected', '?')}, "
                    f"shared_lawyer={meta.get('shared_lawyer_flag', '?')}"
                )
            ana_text = "Analogous historical rings retrieved from vector KB:\n" + "\n".join(ana_lines)
        else:
            ana_text = "No analogous rings found in vector knowledge base."

        override_text = (
            "MANDATORY OVERRIDE TRIGGERED:\n" +
            "\n".join(f"  ⚠  {t}" for t in overrides)
            if overrides else
            "No mandatory override criteria triggered."
        )

        user_message = f"""Analyze this insurance claim for fraud ring patterns.

CLAIM ID: {claim_id}
GNN FRAUD SCORE: {fraud_score:.4f}

SUBGRAPH CONTEXT (from Neo4j):
{subgraph_text}

{ana_text}

{override_text}

Generate a structured investigation brief."""

        try:
            response = llm_client.chat.completions.create(
                model=C.CLAUDE_MODEL,
                max_tokens=C.MAX_TOKENS,
                messages=[
                    {"role": "system", "content": C.REASONING_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            trace = response.choices[0].message.content
        except Exception as e:
            log.error("LLM reasoning failed for %s: %s", claim_id, e)
            trace = (
                f"[Reasoning trace unavailable — LLM API error: {e}]\n\n"
                f"Fraud score: {fraud_score:.4f}\n"
                f"Subgraph:\n{subgraph_text}\n"
                f"Overrides: {', '.join(overrides) or 'none'}"
            )

        return {**state, "reasoning_trace": trace}
    return generate_reasoning


# ── Pipeline builder ──────────────────────────────────────────────────
def build_pipeline(
    retriever:    Optional[GraphRetriever] = None,
    vector_store: Optional[VectorStore]   = None,
    embedder:     Optional[FraudEmbedder] = None,
    llm_client:   Optional[OpenAI]        = None,
):
    """
    Build and compile the GraphRAG LangGraph pipeline.

    All dependencies are created lazily if not provided.

    Returns a compiled LangGraph app (if langgraph is installed) or
    a simple callable that runs the nodes sequentially as a fallback.
    """
    if not C.OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY must be set in .env to use the GraphRAG pipeline."
        )

    retriever    = retriever    or GraphRetriever()
    vector_store = vector_store or get_vector_store(load_existing=True)
    embedder     = embedder     or FraudEmbedder()
    llm_client   = llm_client   or OpenAI(
        api_key=C.OPENROUTER_API_KEY,
        base_url=C.OPENROUTER_BASE_URL,
    )

    retrieve_fn = make_retrieve_node(retriever)
    search_fn   = make_search_node(embedder, vector_store)
    reason_fn   = make_reasoning_node(llm_client)

    if HAS_LANGGRAPH:
        graph = StateGraph(PipelineState)
        graph.add_node("retrieve_subgraph",    retrieve_fn)
        graph.add_node("find_analogous_rings", search_fn)
        graph.add_node("generate_reasoning",   reason_fn)

        graph.add_edge(START,                   "retrieve_subgraph")
        graph.add_edge("retrieve_subgraph",    "find_analogous_rings")
        graph.add_edge("find_analogous_rings", "generate_reasoning")
        graph.add_edge("generate_reasoning",    END)

        return graph.compile()
    else:
        # Fallback: plain function
        def sequential_pipeline(state: dict) -> dict:
            s = PipelineState(**state)
            s = retrieve_fn(s)
            s = search_fn(s)
            s = reason_fn(s)
            return dict(s)
        return sequential_pipeline


# ── Convenience runner ────────────────────────────────────────────────
def run_pipeline(
    claim_id: str,
    retriever:    Optional[GraphRetriever] = None,
    vector_store: Optional[VectorStore]   = None,
    embedder:     Optional[FraudEmbedder] = None,
    llm_client:   Optional[OpenAI]        = None,
) -> Dict[str, Any]:
    """
    Run the full GraphRAG pipeline for a single claim.

    Returns the final pipeline state dict:
      {
        "claim_id":          str,
        "fraud_score":       float,
        "subgraph":          dict,
        "analogous_rings":   list,
        "override_triggers": list,
        "reasoning_trace":   str,
      }
    """
    pipeline = build_pipeline(retriever, vector_store, embedder, llm_client)
    return pipeline.invoke({"claim_id": claim_id, "error": None})
