"""
Phase 3 — GraphRAG + LangGraph CLI

Commands
────────
  index    Embed all FraudRing nodes into the vector knowledge base
  explain  Run full GraphRAG pipeline for a claim (retrieve → search → reason)
  query    Natural language query against the fraud graph
  feedback Record an investigator decision for a claim
  retrain  Trigger feedback-loop retraining with F1 delta measurement
  stats    Print feedback history and vector store stats

Examples
────────
  python run_phase3.py index
  python run_phase3.py explain CLM-521585
  python run_phase3.py explain CLM-521585 --verbose
  python run_phase3.py query
  python run_phase3.py query --question "Which fraud rings have a lawyer in 3+ claims?"
  python run_phase3.py feedback CLM-521585 --decision Approve --investigator INV-001
  python run_phase3.py feedback CLM-521585 --decision Dismiss  --investigator INV-001 --feedback FP
  python run_phase3.py retrain
  python run_phase3.py stats
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase3")


# ── Subcommand implementations ─────────────────────────────────────────

def cmd_index(args: argparse.Namespace) -> None:
    """Embed all FraudRing nodes and save the vector knowledge base."""
    from phase3_rag.graph_retriever import GraphRetriever
    from phase3_rag.embedder import FraudEmbedder
    from phase3_rag.vector_store import get_vector_store

    print("\n  Building fraud ring vector knowledge base …\n")

    retriever = GraphRetriever()
    rings = retriever.get_all_fraud_rings()
    retriever.close()

    if not rings:
        print("  ❌  No FraudRing nodes found in Neo4j.")
        sys.exit(1)

    print(f"  Found {len(rings)} fraud rings — embedding …")

    embedder = FraudEmbedder()
    vs = get_vector_store(load_existing=False)

    from tqdm import tqdm
    for ring in tqdm(rings, desc="  Embedding rings"):
        ring_id = ring.get("ring_id", ring.get("id", "UNKNOWN"))
        vec = embedder.embed_ring(ring)
        meta = {
            "ring_id":             ring_id,
            "status":              ring.get("status", ""),
            "ring_score":          ring.get("ring_score", 0),
            "member_count":        ring.get("member_count", 0),
            "total_claim_amount":  ring.get("total_claim_amount", 0),
            "closed_loop_detected": ring.get("closed_loop_detected", False),
            "shared_lawyer_flag":  ring.get("shared_lawyer_flag", False),
        }
        vs.add(ring_id, vec, meta)

    vs.save()
    print(f"\n  ✅  Indexed {len(vs)} rings into vector store.")


def cmd_explain(args: argparse.Namespace) -> None:
    """Run the full GraphRAG pipeline for a single claim."""
    from phase3_rag.pipeline import run_pipeline

    claim_id = args.claim_id
    print(f"\n  Running GraphRAG pipeline for {claim_id} …\n")

    result = run_pipeline(claim_id)

    if result.get("error"):
        print(f"  ❌  Pipeline error: {result['error']}")
        sys.exit(1)

    fraud_score = result.get("fraud_score", 0.0)
    overrides   = result.get("override_triggers", [])
    analogous   = result.get("analogous_rings", [])
    trace       = result.get("reasoning_trace", "(no trace)")

    print(f"  Claim ID    : {claim_id}")
    print(f"  Fraud Score : {fraud_score:.4f}")

    if overrides:
        print(f"\n  Override triggers ({len(overrides)}):")
        for t in overrides:
            print(f"    ⚠  {t}")

    if analogous:
        print(f"\n  Analogous rings (top {len(analogous)}):")
        for m in analogous:
            meta = m.get("metadata", {})
            print(
                f"    • {m['id']}  sim={m['score']:.3f}  "
                f"status={meta.get('status', '?')}  "
                f"score={meta.get('ring_score', '?')}"
            )

    print(f"\n{'─'*60}")
    print("  Investigation Brief")
    print(f"{'─'*60}")
    print(trace)
    print(f"{'─'*60}\n")

    if args.verbose:
        subgraph = result.get("subgraph", {})
        print("\n  Raw subgraph keys:")
        for k, v in subgraph.items():
            if v:
                print(f"    {k}: {str(v)[:120]}")


def cmd_query(args: argparse.Namespace) -> None:
    """Run a natural language query against the fraud graph."""
    from phase3_rag.nl_query import NLQueryEngine

    engine = NLQueryEngine()

    if args.question:
        answer = engine.query(args.question, verbose=args.verbose)
        print(f"\n{answer}\n")
    else:
        engine.interactive()


def cmd_feedback(args: argparse.Namespace) -> None:
    """Record an investigator decision for a claim."""
    from phase3_rag.feedback import FeedbackStore

    store = FeedbackStore()
    store.record(
        claim_id=args.claim_id,
        decision=args.decision,
        investigator_id=args.investigator,
        feedback_to_model=args.feedback,
        override_reason=args.override_reason,
        confidence=args.confidence,
    )
    store.print_stats()


def cmd_retrain(args: argparse.Namespace) -> None:
    """Trigger feedback-loop retraining and print F1 delta report."""
    from phase3_rag.feedback import FeedbackStore

    store = FeedbackStore()

    # Optionally just evaluate without retraining
    if args.evaluate_only:
        print("\n  Computing F1 metrics on current feedback (no retraining) …\n")
        metrics = store.compute_f1_delta()
        if "error" in metrics:
            print(f"  ❌  {metrics['error']}")
            sys.exit(1)
        print(f"  Reviews      : {metrics['n_reviews']}")
        print(f"  Fraud        : {metrics['n_fraud']}")
        print(f"  Legit        : {metrics['n_legit']}")
        print(f"  Agreement    : {metrics.get('agreement_rate', '?')}")
        print(f"  Model F1     : {metrics.get('model_f1', '?')}")
        print(f"  Model AUC    : {metrics.get('model_auc', '?')}")
        print(f"  Baseline F1  : {metrics.get('baseline_f1', '?')}")
        if "report" in metrics:
            print(f"\n{metrics['report']}")
        return

    report = store.trigger_retrain(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        min_reviews=args.min_reviews,
    )

    print(f"\n  Retrain report:")
    print(f"    Status     : {report.get('status')}")
    print(f"    Reviews    : {report.get('n_reviews')}")
    print(f"    F1 before  : {report.get('f1_before')}")
    print(f"    F1 after   : {report.get('f1_after')}")
    print(f"    F1 delta   : {report.get('f1_delta')}")
    print(f"    AUC after  : {report.get('auc_after')}")
    print(f"    Promoted   : {report.get('promoted')}")

    if report.get("reason"):
        print(f"    Reason     : {report['reason']}")
    if report.get("error"):
        print(f"    Error      : {report['error']}")
        sys.exit(1)


def cmd_stats(args: argparse.Namespace) -> None:
    """Print feedback history and vector store statistics."""
    from phase3_rag.feedback import FeedbackStore
    from phase3_rag.vector_store import get_vector_store

    store = FeedbackStore()
    store.print_stats()

    try:
        vs = get_vector_store(load_existing=True)
        print(f"\n  Vector store entries : {len(vs)}")
    except Exception as e:
        print(f"\n  Vector store         : unavailable ({e})")


# ── Argument parser ────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_phase3.py",
        description="Phase 3 — GraphRAG + LangGraph fraud investigation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── index ────────────────────────────────────────────────────────
    sub.add_parser(
        "index",
        help="Embed all FraudRing nodes into the vector knowledge base",
    )

    # ── explain ──────────────────────────────────────────────────────
    p_explain = sub.add_parser(
        "explain",
        help="Run full GraphRAG pipeline for a claim (retrieve → embed → reason)",
    )
    p_explain.add_argument("claim_id", help="Claim ID, e.g. CLM-521585")
    p_explain.add_argument(
        "--verbose", "-v", action="store_true",
        help="Also print raw subgraph keys",
    )

    # ── query ────────────────────────────────────────────────────────
    p_query = sub.add_parser(
        "query",
        help="Natural language query against the fraud graph",
    )
    p_query.add_argument(
        "--question", "-q", default="",
        help="One-shot question (skips interactive REPL)",
    )
    p_query.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print generated Cypher before results",
    )

    # ── feedback ─────────────────────────────────────────────────────
    p_fb = sub.add_parser(
        "feedback",
        help="Record an investigator decision for a claim",
    )
    p_fb.add_argument("claim_id", help="Claim ID, e.g. CLM-521585")
    p_fb.add_argument(
        "--decision", "-d", required=True,
        choices=["Approve", "Dismiss", "Escalate"],
        help="Investigator decision",
    )
    p_fb.add_argument(
        "--investigator", "-i", required=True,
        help="Investigator ID, e.g. INV-001",
    )
    p_fb.add_argument(
        "--feedback", "-f", default="Correct",
        choices=["Correct", "FP", "FN", "Uncertain"],
        help="Fine-grained model feedback signal (default: Correct)",
    )
    p_fb.add_argument(
        "--override-reason", default="",
        help="Optional override reason code (e.g. OVR-001)",
    )
    p_fb.add_argument(
        "--confidence", type=float, default=1.0,
        help="Decision confidence 0.0–1.0 (default: 1.0)",
    )

    # ── retrain ──────────────────────────────────────────────────────
    p_rt = sub.add_parser(
        "retrain",
        help="Trigger feedback-loop retraining with F1 delta measurement",
    )
    p_rt.add_argument(
        "--evaluate-only", action="store_true",
        help="Compute F1 metrics without triggering retraining",
    )
    p_rt.add_argument(
        "--data-dir", default="./data",
        help="Path to CSV data directory (default: ./data)",
    )
    p_rt.add_argument(
        "--models-dir", default=None,
        help="Path to models directory (default: from config)",
    )
    p_rt.add_argument(
        "--min-reviews", type=int, default=20,
        help="Minimum labelled reviews before retraining (default: 20)",
    )

    # ── stats ────────────────────────────────────────────────────────
    sub.add_parser(
        "stats",
        help="Print feedback history and vector store statistics",
    )

    return parser


# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "index":    cmd_index,
        "explain":  cmd_explain,
        "query":    cmd_query,
        "feedback": cmd_feedback,
        "retrain":  cmd_retrain,
        "stats":    cmd_stats,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(0)
    except EnvironmentError as e:
        print(f"\n  ❌  Configuration error: {e}")
        print("  Make sure NEO4J_URI, NEO4J_PASSWORD, and OPENROUTER_API_KEY are set in .env")
        sys.exit(1)
    except Exception as e:
        log.exception("Unexpected error in %s: %s", args.command, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
