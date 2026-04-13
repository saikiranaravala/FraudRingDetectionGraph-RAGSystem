"""
Fraud Ring Detection — Unified CLI

Usage
─────
  # Graph loading (Phase 1)
  python src/main.py load-graph [--dry-run] [--batch N]

  # GNN scoring (Phase 2 — run locally, not on Render)
  python src/main.py gnn train
  python src/main.py gnn score [--dry-run]
  python src/main.py gnn explain [--top-n N] [--output FILE]
  python src/main.py gnn evaluate

  # GraphRAG pipeline (Phase 3)
  python src/main.py rag index
  python src/main.py rag explain CLM-521585 [--verbose]
  python src/main.py rag query [--question "..."]
  python src/main.py rag feedback CLM-521585 --decision Approve --investigator INV-001
  python src/main.py rag retrain [--evaluate-only] [--min-reviews N]
  python src/main.py rag stats
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root is on sys.path when invoked as `python src/main.py`
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── load-graph ────────────────────────────────────────────────────────

def cmd_load_graph(args: argparse.Namespace) -> None:
    from src.tools.load_graph import main as load_main
    sys.argv = ["load_graph.py"]
    if args.dry_run:
        sys.argv.append("--dry-run")
    if args.batch:
        sys.argv += ["--batch", str(args.batch)]
    if args.uri:
        sys.argv += ["--uri", args.uri]
    if args.password:
        sys.argv += ["--password", args.password]
    load_main()


# ── gnn ───────────────────────────────────────────────────────────────

def cmd_gnn_train(args: argparse.Namespace) -> None:
    try:
        import torch
        from src.services.gnn.train import run_training
    except ImportError as e:
        print(f"\n  [ERROR]  GNN dependencies not installed: {e}")
        print("  Run: pip install -r requirements_phase2.txt")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training GNN on {device} …\n")
    run_training(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        device=device,
        epochs=args.epochs,
    )


def cmd_gnn_score(args: argparse.Namespace) -> None:
    try:
        import torch
        from src.services.gnn.scorer import run_scoring
    except ImportError as e:
        print(f"\n  [ERROR]  GNN dependencies not installed: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_scoring(data_dir=args.data_dir, models_dir=args.models_dir,
                device=device, dry_run=args.dry_run)


def cmd_gnn_explain(args: argparse.Namespace) -> None:
    try:
        import torch
        from src.services.gnn.explainer import FraudExplainer, format_trace
        from src.services.gnn.train import load_artefacts
        from src.services.gnn.data_loader import build_hetero_data
    except ImportError as e:
        print(f"\n  [ERROR]  GNN dependencies not installed: {e}")
        sys.exit(1)

    import json
    device = torch.device("cpu")
    data, meta = build_hetero_data(data_dir=args.data_dir)
    model, _, _, _ = load_artefacts(models_dir=args.models_dir, device=device)
    explainer = FraudExplainer(model=model, data=data, meta=meta, device=device)
    traces = explainer.explain_top_flagged(n=args.top_n)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(traces, f, indent=2)
        print(f"  Saved {len(traces)} traces to {args.output}")
    else:
        from src.services.gnn.explainer import format_trace
        for t in traces:
            print(format_trace(t))


def cmd_gnn_evaluate(args: argparse.Namespace) -> None:
    try:
        import torch
        from src.services.gnn.train import run_training
    except ImportError as e:
        print(f"\n  [ERROR]  GNN dependencies not installed: {e}")
        sys.exit(1)

    device = torch.device("cpu")
    run_training(data_dir=args.data_dir, models_dir=args.models_dir,
                 device=device, evaluate_only=True)


# ── rag ───────────────────────────────────────────────────────────────

def cmd_rag_index(args: argparse.Namespace) -> None:
    from src.services.graph_retriever import GraphRetriever
    from src.utils.embedder import FraudEmbedder
    from src.services.vector_store import get_vector_store
    from tqdm import tqdm

    print("\n  Building fraud ring vector knowledge base …\n")
    retriever = GraphRetriever()
    rings = retriever.get_all_fraud_rings()
    retriever.close()

    if not rings:
        print("  [ERROR]  No FraudRing nodes found in Neo4j.")
        sys.exit(1)

    print(f"  Found {len(rings)} fraud rings — embedding …")
    embedder = FraudEmbedder()
    vs = get_vector_store(load_existing=False)

    for ring in tqdm(rings, desc="  Embedding rings"):
        ring_id = ring.get("ring_id", ring.get("id", "UNKNOWN"))
        vec = embedder.embed_ring(ring)
        vs.add(ring_id, vec, {
            "ring_id":             ring_id,
            "status":              ring.get("status", ""),
            "ring_score":          ring.get("ring_score", 0),
            "member_count":        ring.get("member_count", 0),
            "total_claim_amount":  ring.get("total_claim_amount", 0),
            "closed_loop_detected": ring.get("closed_loop_detected", False),
            "shared_lawyer_flag":  ring.get("shared_lawyer_flag", False),
        })

    vs.save()
    print(f"\n  [OK]  Indexed {len(vs)} rings into vector store.")


def cmd_rag_explain(args: argparse.Namespace) -> None:
    from src.agent.pipeline import run_pipeline

    claim_id = args.claim_id
    print(f"\n  Running GraphRAG pipeline for {claim_id} …\n")
    result = run_pipeline(claim_id)

    if result.get("error"):
        print(f"  [ERROR]  Pipeline error: {result['error']}")
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
            print(f"    [!]  {t}")

    if analogous:
        print(f"\n  Analogous rings (top {len(analogous)}):")
        for m in analogous:
            meta = m.get("metadata", {})
            print(f"    • {m['id']}  sim={m['score']:.3f}  "
                  f"status={meta.get('status', '?')}  score={meta.get('ring_score', '?')}")

    print(f"\n{'─'*60}\n  Investigation Brief\n{'─'*60}")
    print(trace)
    print(f"{'─'*60}\n")

    if args.verbose:
        subgraph = result.get("subgraph", {})
        print("\n  Raw subgraph keys:")
        for k, v in subgraph.items():
            if v:
                print(f"    {k}: {str(v)[:120]}")


def cmd_rag_query(args: argparse.Namespace) -> None:
    from src.tools.nl_query import NLQueryEngine
    engine = NLQueryEngine()
    if args.question:
        print(f"\n{engine.query(args.question, verbose=args.verbose)}\n")
    else:
        engine.interactive()


def cmd_rag_feedback(args: argparse.Namespace) -> None:
    from src.services.feedback import FeedbackStore
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


def cmd_rag_retrain(args: argparse.Namespace) -> None:
    from src.services.feedback import FeedbackStore
    store = FeedbackStore()

    if args.evaluate_only:
        print("\n  Computing F1 metrics (no retraining) …\n")
        metrics = store.compute_f1_delta()
        if "error" in metrics:
            print(f"  [ERROR]  {metrics['error']}")
            sys.exit(1)
        for k, v in metrics.items():
            if k != "report":
                print(f"    {k:20}: {v}")
        if "report" in metrics:
            print(f"\n{metrics['report']}")
        return

    report = store.trigger_retrain(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        min_reviews=args.min_reviews,
    )
    for k, v in report.items():
        print(f"  {k:15}: {v}")
    if report.get("error"):
        sys.exit(1)


def cmd_rag_stats(args: argparse.Namespace) -> None:
    from src.services.feedback import FeedbackStore
    from src.services.vector_store import get_vector_store
    FeedbackStore().print_stats()
    try:
        vs = get_vector_store(load_existing=True)
        print(f"\n  Vector store entries : {len(vs)}")
    except Exception as e:
        print(f"\n  Vector store         : unavailable ({e})")


# ── Argument parser ───────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="src/main.py",
        description="Fraud Ring Detection — unified CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── load-graph ───────────────────────────────────────────────────
    lg = sub.add_parser("load-graph", help="Load graph data into Neo4j")
    lg.add_argument("--dry-run", action="store_true")
    lg.add_argument("--batch",   type=int, default=None)
    lg.add_argument("--uri",     default=None)
    lg.add_argument("--password", default=None)

    # ── gnn ──────────────────────────────────────────────────────────
    gnn = sub.add_parser("gnn", help="GNN scoring commands (run locally)")
    gnn_sub = gnn.add_subparsers(dest="gnn_cmd", required=True)

    def _gnn_common(p):
        p.add_argument("--data-dir",   default="./data")
        p.add_argument("--models-dir", default="./models")
        return p

    g_train = _gnn_common(gnn_sub.add_parser("train",    help="Train GNN + ensemble"))
    g_train.add_argument("--epochs", type=int, default=None)
    g_score = gnn_sub.add_parser("score", help="Score all claims + write to Neo4j")
    g_score.add_argument("--data-dir",   default="./data")
    g_score.add_argument("--models-dir", default="./models")
    g_score.add_argument("--dry-run", action="store_true")
    g_explain = gnn_sub.add_parser("explain", help="Explain top flagged claims")
    g_explain.add_argument("--data-dir",   default="./data")
    g_explain.add_argument("--models-dir", default="./models")
    g_explain.add_argument("--top-n",  type=int, default=20)
    g_explain.add_argument("--output", default=None)
    _gnn_common(gnn_sub.add_parser("evaluate", help="Re-evaluate saved model"))

    # ── rag ──────────────────────────────────────────────────────────
    rag = sub.add_parser("rag", help="GraphRAG pipeline commands")
    rag_sub = rag.add_subparsers(dest="rag_cmd", required=True)

    rag_sub.add_parser("index", help="Embed all FraudRings into vector store")

    r_explain = rag_sub.add_parser("explain", help="Run GraphRAG pipeline for a claim")
    r_explain.add_argument("claim_id")
    r_explain.add_argument("--verbose", "-v", action="store_true")

    r_query = rag_sub.add_parser("query", help="Natural language graph query")
    r_query.add_argument("--question", "-q", default="")
    r_query.add_argument("--verbose",  "-v", action="store_true")

    r_fb = rag_sub.add_parser("feedback", help="Record investigator decision")
    r_fb.add_argument("claim_id")
    r_fb.add_argument("--decision",    "-d", required=True,
                      choices=["Approve", "Dismiss", "Escalate"])
    r_fb.add_argument("--investigator", "-i", required=True)
    r_fb.add_argument("--feedback",    "-f", default="Correct",
                      choices=["Correct", "FP", "FN", "Uncertain"])
    r_fb.add_argument("--override-reason", default="")
    r_fb.add_argument("--confidence",  type=float, default=1.0)

    r_rt = rag_sub.add_parser("retrain", help="Trigger feedback-loop retraining")
    r_rt.add_argument("--evaluate-only", action="store_true")
    r_rt.add_argument("--data-dir",   default="./data")
    r_rt.add_argument("--models-dir", default=None)
    r_rt.add_argument("--min-reviews", type=int, default=20)

    rag_sub.add_parser("stats", help="Print feedback + vector store stats")

    return p


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    try:
        if args.command == "load-graph":
            cmd_load_graph(args)
        elif args.command == "gnn":
            dispatch = {
                "train":    cmd_gnn_train,
                "score":    cmd_gnn_score,
                "explain":  cmd_gnn_explain,
                "evaluate": cmd_gnn_evaluate,
            }
            dispatch[args.gnn_cmd](args)
        elif args.command == "rag":
            dispatch = {
                "index":    cmd_rag_index,
                "explain":  cmd_rag_explain,
                "query":    cmd_rag_query,
                "feedback": cmd_rag_feedback,
                "retrain":  cmd_rag_retrain,
                "stats":    cmd_rag_stats,
            }
            dispatch[args.rag_cmd](args)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(0)
    except EnvironmentError as e:
        print(f"\n  [ERROR]  Configuration error: {e}")
        print("  Ensure NEO4J_URI, NEO4J_PASSWORD, OPENROUTER_API_KEY are set in .env")
        sys.exit(1)
    except Exception as e:
        log.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
