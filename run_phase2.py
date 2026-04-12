#!/usr/bin/env python3
"""
Phase 2 — GNN Scoring CLI
==========================
Entry point for all Phase 2 operations.

Commands
────────
  train     Train FraudGNN + ensemble, save artefacts to models/
  score     Load trained model, score all Claims, write to Neo4j
  explain   Generate reasoning traces for top-N flagged claims
  evaluate  Re-evaluate a saved model on the test split

Examples
────────
  python run_phase2.py train
  python run_phase2.py train --epochs 200 --lr 0.0005
  python run_phase2.py score --dry-run
  python run_phase2.py score
  python run_phase2.py explain --top-n 20 --threshold 0.6
  python run_phase2.py evaluate

Prerequisites
─────────────
  pip install torch torch-geometric imbalanced-learn xgboost lightgbm
  (see requirements_phase2.txt for full instructions)
"""

import argparse
import logging
import os
import sys

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2")

# ── Load .env ─────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Argument parser ───────────────────────────────────────────────────
def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Fraud Ring GNN Scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train GNN + ensemble")
    p_train.add_argument("--data-dir",   default="./data",   help="CSV data directory")
    p_train.add_argument("--models-dir", default="./models", help="Artefact output directory")
    p_train.add_argument("--epochs",     type=int,   default=None, help="Override max epochs")
    p_train.add_argument("--lr",         type=float, default=None, help="Override learning rate")
    p_train.add_argument("--hidden",     type=int,   default=None, help="Override hidden_channels")
    p_train.add_argument("--patience",   type=int,   default=None, help="Override early stopping patience")
    p_train.add_argument("--seed",       type=int,   default=None, help="Override random seed")
    p_train.add_argument("--cpu",        action="store_true",       help="Force CPU even if GPU available")

    # ── score ──────────────────────────────────────────────────────────
    p_score = sub.add_parser("score", help="Score all claims and write to Neo4j")
    p_score.add_argument("--data-dir",   default="./data",   help="CSV data directory")
    p_score.add_argument("--models-dir", default="./models", help="Artefact directory")
    p_score.add_argument("--threshold",  type=float, default=0.5,  help="Decision threshold for tier assignment")
    p_score.add_argument("--dry-run",    action="store_true",       help="Print scores but skip Neo4j writes")
    p_score.add_argument("--cpu",        action="store_true",       help="Force CPU")

    # ── explain ────────────────────────────────────────────────────────
    p_explain = sub.add_parser("explain", help="Generate reasoning traces for flagged claims")
    p_explain.add_argument("--data-dir",   default="./data",   help="CSV data directory")
    p_explain.add_argument("--models-dir", default="./models", help="Artefact directory")
    p_explain.add_argument("--top-n",      type=int,   default=20,  help="Number of top claims to explain")
    p_explain.add_argument("--threshold",  type=float, default=0.5, help="Minimum fraud score to explain")
    p_explain.add_argument("--output",     default=None,             help="Save traces to JSON file")
    p_explain.add_argument("--cpu",        action="store_true",      help="Force CPU")

    # ── evaluate ───────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Re-evaluate saved model on test split")
    p_eval.add_argument("--data-dir",   default="./data",   help="CSV data directory")
    p_eval.add_argument("--models-dir", default="./models", help="Artefact directory")
    p_eval.add_argument("--threshold",  type=float, default=0.5, help="Decision threshold")
    p_eval.add_argument("--cpu",        action="store_true",      help="Force CPU")

    return parser


# ── Command handlers ──────────────────────────────────────────────────
def cmd_train(args):
    import torch
    from phase2_gnn import config as C
    from phase2_gnn.train import run_training

    cfg = dict(C.TRAIN_CONFIG)  # copy defaults
    if args.epochs   is not None: cfg["epochs"]          = args.epochs
    if args.lr       is not None: cfg["lr"]              = args.lr
    if args.hidden   is not None: cfg["hidden_channels"] = args.hidden
    if args.patience is not None: cfg["patience"]        = args.patience
    if args.seed     is not None: cfg["random_seed"]     = args.seed

    device = torch.device("cpu") if args.cpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ensemble, metrics = run_training(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        device=device,
        cfg=cfg,
    )

    print(f"\n{'='*55}")
    print("TRAINING COMPLETE")
    print(f"{'='*55}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}  (target ≥ 0.91)")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    if metrics["auc_roc"] >= 0.91:
        print("\n  ✅ Phase 2 exit criterion MET: AUC-ROC ≥ 0.91")
    else:
        print(f"\n  ⚠  AUC-ROC {metrics['auc_roc']:.4f} below 0.91 target. "
              "Consider more epochs or feature engineering.")


def cmd_score(args):
    import torch
    from phase2_gnn.scorer import run_scoring

    device = torch.device("cpu") if args.cpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_scoring(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        dry_run=args.dry_run,
        device=device,
    )


def cmd_explain(args):
    import json
    import torch
    from phase2_gnn.data_loader import build_hetero_data
    from phase2_gnn.train import load_artefacts
    from phase2_gnn.explainer import FraudExplainer

    device = torch.device("cpu") if args.cpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading graph …")
    data, meta = build_hetero_data(data_dir=args.data_dir, verbose=False)

    print("Loading model …")
    from phase2_gnn import config as C
    model, ensemble = load_artefacts(
        data, meta, models_dir=args.models_dir, device=device
    )

    explainer = FraudExplainer(model, data, meta, device=device)

    print(f"\nExplaining top {args.top_n} claims with score ≥ {args.threshold} …\n")
    traces = explainer.explain_top_flagged(n=args.top_n, threshold=args.threshold)

    for trace in traces:
        print(FraudExplainer.format_trace(trace))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(traces, f, indent=2)
        print(f"\n  Traces saved to {args.output}")

    print(f"\n  Explained {len(traces)} claims.")


def cmd_evaluate(args):
    import torch
    from phase2_gnn.data_loader import load_graph_splits
    from phase2_gnn.train import evaluate, load_artefacts

    device = torch.device("cpu") if args.cpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading graph splits …")
    data, train_mask, val_mask, test_mask, meta = load_graph_splits(
        data_dir=args.data_dir, verbose=True
    )

    print("Loading model …")
    model, ensemble = load_artefacts(
        data, meta, models_dir=args.models_dir, device=device
    )

    metrics = evaluate(model, ensemble, data, test_mask, device, threshold=args.threshold)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = make_parser()
    args = parser.parse_args()

    # Validate PyG is installed before any command
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        print(
            "\n❌ PyTorch Geometric is not installed.\n"
            "   Install it following the instructions in requirements_phase2.txt:\n"
            "   1. pip install torch  (CPU or GPU build)\n"
            "   2. pip install torch-geometric\n"
            "   3. pip install torch-scatter torch-sparse "
            "-f https://data.pyg.org/whl/torch-<version>+cpu.html\n",
            file=sys.stderr,
        )
        sys.exit(1)

    dispatch = {
        "train":    cmd_train,
        "score":    cmd_score,
        "explain":  cmd_explain,
        "evaluate": cmd_evaluate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
