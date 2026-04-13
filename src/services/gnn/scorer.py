"""
Scorer — batch inference + Neo4j write-back.

Runs the trained FraudGNN + ensemble over the full graph and writes
gnn_suspicion_score back to every Claim and NetworkFeature node in Neo4j.

Neo4j properties written
─────────────────────────
  Claim.gnn_suspicion_score        float  [0, 1]
  Claim.ensemble_suspicion_score   float  [0, 1]
  Claim.final_suspicion_score      float  [0, 1]   (blended GNN + ensemble)
  Claim.adjuster_priority_tier     str    updated if score exceeds threshold
  NetworkFeature.ring_suspicion_score  float  (updated for Claim-linked NFs)

Score tiers (PRD §6 / §7)
──────────────────────────
  ≥ 0.90  → CRITICAL / Mandatory Override (OVR-001)
  ≥ 0.70  → HIGH PRIORITY queue
  ≥ 0.50  → STANDARD queue
  <  0.50 → Low suspicion

Usage
─────
    python run_phase2.py score
    python run_phase2.py score --threshold 0.6 --dry-run
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from . import config as C
from .data_loader import build_hetero_data
from .model import FraudGNN, build_model
from .train import extract_embeddings, load_artefacts

# ── Score tier mapping ────────────────────────────────────────────────
def score_to_tier(score: float) -> str:
    if score >= 0.90:
        return "Critical"
    if score >= 0.70:
        return "High Priority"
    if score >= 0.50:
        return "Standard"
    return "Standard"


# ── Full inference pass ───────────────────────────────────────────────
def run_inference(
    model: FraudGNN,
    ensemble: dict,
    data,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run GNN + ensemble inference on all Claim nodes.

    Returns
    -------
    gnn_scores      : np.ndarray shape (N_claims,) — GNN sigmoid probabilities
    ensemble_scores : np.ndarray shape (N_claims,) — ensemble mean probabilities
    final_scores    : np.ndarray shape (N_claims,) — 0.5*GNN + 0.5*ensemble
    """
    device = device or torch.device("cpu")

    # GNN probabilities
    model.eval()
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    with torch.no_grad():
        gnn_scores = model.predict_proba(x_dict, ei_dict).cpu().numpy()

    # Ensemble probabilities
    embs = extract_embeddings(model, data, device)
    ensemble_models = {k: v for k, v in ensemble.items() if k not in ("val_auc",)}

    if ensemble_models:
        embs_np = np.asarray(embs, dtype=np.float32)
        # LightGBM 4.x auto-assigns internal feature names during fit and warns
        # when a plain numpy array (no names) is passed at inference.
        # Scores are correct regardless — suppress the cosmetic warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            ens_probas = np.mean(
                [m.predict_proba(embs_np)[:, 1] for m in ensemble_models.values()],
                axis=0,
            )
    else:
        ens_probas = gnn_scores.copy()

    final_scores = 0.5 * gnn_scores + 0.5 * ens_probas
    return gnn_scores, ens_probas, final_scores


# ── Neo4j write-back ──────────────────────────────────────────────────
def _get_driver():
    """Create Neo4j driver from environment variables."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError("neo4j Python driver required: pip install neo4j") from exc

    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw   = os.getenv("NEO4J_PASSWORD")

    if not uri or not pw:
        raise EnvironmentError(
            "NEO4J_URI and NEO4J_PASSWORD must be set in .env or environment."
        )

    trust_all = os.getenv("NEO4J_TRUST_ALL_CERTS", "true").lower() == "true"
    _UPGRADE = {"neo4j+s": "neo4j+ssc", "bolt+s": "bolt+ssc"}
    if trust_all:
        for strict, relaxed in _UPGRADE.items():
            if uri.startswith(strict + "://"):
                uri = relaxed + uri[len(strict):]
                break

    driver = GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    return driver


def write_scores_to_neo4j(
    claim_ids: List[str],
    gnn_scores: np.ndarray,
    ensemble_scores: np.ndarray,
    final_scores: np.ndarray,
    batch_size: int = 200,
    dry_run: bool = False,
) -> int:
    """
    Write suspicion scores back to Neo4j Claim nodes.

    Args:
        claim_ids       : list of Claim :ID strings (same order as score arrays)
        gnn_scores      : GNN probabilities
        ensemble_scores : ensemble probabilities
        final_scores    : blended final probabilities
        batch_size      : UNWIND batch size
        dry_run         : if True, print what would be written but skip DB writes

    Returns:
        Number of claims updated.
    """
    # Build payload
    rows = []
    for cid, gnn, ens, final in zip(
        claim_ids, gnn_scores, ensemble_scores, final_scores
    ):
        rows.append({
            "claim_id":                str(cid),
            "gnn_suspicion_score":     float(round(gnn, 4)),
            "ensemble_suspicion_score": float(round(ens, 4)),
            "final_suspicion_score":   float(round(final, 4)),
            "adjuster_priority_tier":  score_to_tier(float(final)),
        })

    if dry_run:
        critical = sum(1 for r in rows if r["final_suspicion_score"] >= 0.90)
        high     = sum(1 for r in rows if 0.70 <= r["final_suspicion_score"] < 0.90)
        standard = sum(1 for r in rows if r["final_suspicion_score"] < 0.70)
        print(f"\n  [DRY RUN] Would update {len(rows)} Claim nodes:")
        print(f"    Critical (≥0.90): {critical}")
        print(f"    High (≥0.70):     {high}")
        print(f"    Standard (<0.70): {standard}")
        print("  Top 5 scores:")
        top5 = sorted(rows, key=lambda r: r["final_suspicion_score"], reverse=True)[:5]
        for r in top5:
            print(f"    {r['claim_id']}: {r['final_suspicion_score']:.4f}  [{r['adjuster_priority_tier']}]")
        return len(rows)

    query = """
    UNWIND $rows AS row
    MATCH (c:Claim {_neo4j_id: row.claim_id})
    SET c.gnn_suspicion_score        = row.gnn_suspicion_score,
        c.ensemble_suspicion_score   = row.ensemble_suspicion_score,
        c.final_suspicion_score      = row.final_suspicion_score,
        c.adjuster_priority_tier     = row.adjuster_priority_tier
    """

    driver = _get_driver()
    total_written = 0
    try:
        with driver.session() as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i: i + batch_size]
                session.run(query, rows=batch)
                total_written += len(batch)
                print(f"  Written {total_written}/{len(rows)} claims …", end="\r")
    finally:
        driver.close()

    print(f"\n  ✅ Updated {total_written} Claim nodes in Neo4j.")
    return total_written


def write_investigation_scores(
    claim_ids: List[str],
    final_scores: np.ndarray,
    meta: dict,
    dry_run: bool = False,
) -> int:
    """
    Update InvestigationCase.ai_fraud_score for cases linked to scored claims.

    Links via InvestigationCase.linked_claim_id property.
    """
    if dry_run:
        print(f"\n  [DRY RUN] Would update InvestigationCase.ai_fraud_score "
              f"for {len(claim_ids)} linked cases.")
        return len(claim_ids)

    rows = [
        {"claim_id": str(cid), "ai_fraud_score": float(round(sc, 4))}
        for cid, sc in zip(claim_ids, final_scores)
    ]
    query = """
    UNWIND $rows AS row
    MATCH (ic:InvestigationCase {linked_claim_id: row.claim_id})
    SET ic.ai_fraud_score = row.ai_fraud_score
    """
    driver = _get_driver()
    written = 0
    try:
        with driver.session() as session:
            for i in range(0, len(rows), 200):
                batch = rows[i: i + 200]
                session.run(query, rows=batch)
                written += len(batch)
    finally:
        driver.close()
    print(f"  ✅ Updated {written} InvestigationCase ai_fraud_score values.")
    return written


# ── Full scoring pipeline ─────────────────────────────────────────────
def run_scoring(
    data_dir: str = C.DATA_DIR,
    models_dir: str = C.MODELS_DIR,
    dry_run: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """
    Load trained artefacts, score all Claims, write back to Neo4j.

    Returns dict of {claim_id: final_score} for downstream use.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading graph …")
    data, meta = build_hetero_data(data_dir=data_dir, verbose=False)

    print("Loading trained model …")
    model, ensemble = load_artefacts(data, meta, models_dir=models_dir, device=device)

    print("Running inference …")
    gnn_scores, ens_scores, final_scores = run_inference(model, ensemble, data, device)

    # Recover claim IDs in index order
    claim_idx_to_id = {
        v: k for k, v in meta["node_index"]["Claim"].items()
    }
    n_claims = len(gnn_scores)
    claim_ids = [claim_idx_to_id.get(i, f"UNKNOWN-{i}") for i in range(n_claims)]

    # Write to Neo4j (or dry-run print)
    write_scores_to_neo4j(
        claim_ids, gnn_scores, ens_scores, final_scores,
        dry_run=dry_run,
    )

    write_investigation_scores(
        claim_ids, final_scores, meta, dry_run=dry_run
    )

    # Score summary
    print(f"\n{'='*55}")
    print("SCORING SUMMARY")
    print(f"{'='*55}")
    print(f"  Total claims scored : {n_claims}")
    print(f"  Critical (≥0.90)    : {(final_scores >= 0.90).sum()}")
    print(f"  High     (≥0.70)    : {((final_scores >= 0.70) & (final_scores < 0.90)).sum()}")
    print(f"  Standard (<0.70)    : {(final_scores < 0.70).sum()}")
    print(f"  Mean score          : {final_scores.mean():.4f}")
    print(f"  Max score           : {final_scores.max():.4f}")

    return dict(zip(claim_ids, final_scores.tolist()))
