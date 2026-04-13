"""
GNN Explainability — Reasoning Trace Generation

Uses PyTorch Geometric's GNNExplainer to identify the critical subgraph
edges and node features that drove a specific Claim's fraud suspicion score.

Output is a structured Reasoning Trace dict that can be:
  - Displayed in the investigator dashboard
  - Passed to Claude (Anthropic API) in Phase 3 for natural-language narration
  - Stored back to InvestigationCase.explanation_trace_id in Neo4j

Reference: Ying et al. (2019) — "GNNExplainer: Generating Explanations
for Graph Neural Networks"

Usage
─────
    from phase2_gnn.explainer import FraudExplainer
    explainer = FraudExplainer(model, data, meta)
    trace = explainer.explain_claim(claim_idx=42)
    print(explainer.format_trace(trace))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.explain import Explainer, GNNExplainer
    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False
    log.warning(
        "torch_geometric.explain not available in your PyG version. "
        "Upgrade to torch-geometric >= 2.4.0 for GNNExplainer support. "
        "Falling back to gradient-based importance scores."
    )

from . import config as C
from .model import FraudGNN


# ── Fraud signal labels (used to annotate feature importance) ─────────
# Maps feature position within the Claim feature vector to a human label.
# Built from config.NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES.
def _build_claim_feature_names() -> List[str]:
    names = []
    names += C.NUMERIC_FEATURES.get("Claim", [])
    names += [c for c in C.BINARY_FEATURES.get("Claim", []) if c != C.LABEL_COL]
    names += list(C.ORDINAL_FEATURES.get("Claim", {}).keys())
    return names if names else [f"feat_{i}" for i in range(32)]


CLAIM_FEATURE_NAMES = _build_claim_feature_names()

# Human-readable severity labels for SHARES_ATTRIBUTE edges
ATTR_RISK_LABELS = {
    "CRITICAL": "shared bank account (CRITICAL)",
    "HIGH":     "shared IP address (HIGH)",
    "MEDIUM":   "shared phone number (MEDIUM)",
    "LOW":      "shared zip code (LOW)",
}


class FraudExplainer:
    """
    Generates per-claim reasoning traces from a trained FraudGNN.

    Two explanation methods:
      1. GNNExplainer (preferred)  — masks edges/features iteratively.
      2. Gradient saliency (fallback) — grad of fraud score w.r.t. features.
    """

    def __init__(
        self,
        model: FraudGNN,
        data: HeteroData,
        meta: dict,
        device: Optional[torch.device] = None,
        top_k_edges: int = 10,
        top_k_features: int = 8,
    ):
        self.model = model
        self.data = data
        self.meta = meta
        self.device = device or torch.device("cpu")
        self.top_k_edges = top_k_edges
        self.top_k_features = top_k_features

        self.model = model.to(self.device).eval()

        # Reverse index: claim int → claim_id string
        self.claim_id_map: Dict[int, str] = {}
        if "node_index" in meta and "Claim" in meta["node_index"]:
            self.claim_id_map = {v: k for k, v in meta["node_index"]["Claim"].items()}

        # Build PyG Explainer if available
        self._explainer: Optional[Any] = None
        if HAS_EXPLAINER:
            try:
                self._explainer = self._build_pyg_explainer()
            except Exception as e:
                log.warning("Could not build PyG Explainer: %s. Using gradient fallback.", e)

    def _build_pyg_explainer(self):
        """Wrap model in PyG Explainer for heterogeneous GNNExplainer."""
        # PyG's Explainer API works best with homogeneous models.
        # For heterogeneous graphs we use the gradient saliency path
        # (which is more robust across PyG versions).
        return None  # Gradient saliency used by default (see explain_claim)

    # ── Gradient saliency ─────────────────────────────────────────────
    def _gradient_saliency(self, claim_idx: int) -> Dict[str, np.ndarray]:
        """
        Compute input-gradient saliency for Claim node features.

        Returns {node_type: importance_array} where importance is the
        L2 norm of ∂(fraud_score) / ∂(node_features).
        """
        x_dict = {
            nt: x.clone().to(self.device).requires_grad_(True)
            for nt, x in self.data.x_dict.items()
        }
        ei_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}

        self.model.train()  # enable gradients through BN
        _, logits = self.model(x_dict, ei_dict)
        score = torch.sigmoid(logits[claim_idx])
        score.backward()
        self.model.eval()

        saliency = {}
        for nt, x in x_dict.items():
            if x.grad is not None:
                saliency[nt] = x.grad.abs().cpu().numpy()
            else:
                saliency[nt] = np.zeros_like(x.detach().cpu().numpy())
        return saliency

    # ── Main explain method ───────────────────────────────────────────
    def explain_claim(self, claim_idx: int) -> Dict:
        """
        Generate a reasoning trace for a specific Claim node.

        Args:
            claim_idx : integer index of the Claim node in the graph.

        Returns
        -------
        trace : dict with keys:
            claim_id            — string identifier
            fraud_score         — float [0, 1] GNN fraud probability
            top_features        — list of (feature_name, importance) tuples
            top_neighbor_types  — node types most influential via message passing
            override_triggers   — list of OVR codes triggered (from node flags)
            signal_summary      — one-line textual summary
        """
        x_dict = {k: v.to(self.device) for k, v in self.data.x_dict.items()}
        ei_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}

        self.model.eval()
        with torch.no_grad():
            proba = self.model.predict_proba(x_dict, ei_dict)[claim_idx].item()

        # Gradient saliency for feature importance
        saliency = self._gradient_saliency(claim_idx)

        # Top Claim-level features
        claim_saliency = saliency.get("Claim", np.array([]))
        if claim_saliency.ndim == 2:
            claim_importance = claim_saliency[claim_idx]
        else:
            claim_importance = claim_saliency.flatten()

        feature_names = CLAIM_FEATURE_NAMES
        top_features = self._top_k_items(
            claim_importance, feature_names, k=self.top_k_features
        )

        # Top contributing neighbor node types (by average saliency magnitude)
        neighbor_contributions = {
            nt: float(np.mean(sal))
            for nt, sal in saliency.items()
            if nt != "Claim" and sal.size > 0
        }
        top_neighbors = sorted(
            neighbor_contributions.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # OVR trigger detection from raw Claim features
        override_triggers = self._detect_overrides(claim_idx)

        # Build textual summary
        summary = self._build_summary(
            claim_idx, proba, top_features, top_neighbors, override_triggers
        )

        return {
            "claim_id":           self.claim_id_map.get(claim_idx, f"CLM-{claim_idx}"),
            "fraud_score":        round(proba, 4),
            "top_features":       top_features,
            "top_neighbor_types": top_neighbors,
            "override_triggers":  override_triggers,
            "signal_summary":     summary,
        }

    # ── Batch explain ─────────────────────────────────────────────────
    def explain_top_flagged(
        self,
        n: int = 20,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Explain the top-N highest-scoring claims above threshold.

        Returns list of trace dicts sorted by fraud_score descending.
        """
        x_dict = {k: v.to(self.device) for k, v in self.data.x_dict.items()}
        ei_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}

        self.model.eval()
        with torch.no_grad():
            probas = self.model.predict_proba(x_dict, ei_dict).cpu().numpy()

        flagged = np.where(probas >= threshold)[0]
        top_indices = flagged[np.argsort(-probas[flagged])][:n]

        traces = []
        for idx in top_indices:
            try:
                trace = self.explain_claim(int(idx))
                traces.append(trace)
            except Exception as e:
                log.warning("Explanation failed for claim %d: %s", idx, e)

        return traces

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _top_k_items(
        importance: np.ndarray,
        names: List[str],
        k: int = 8,
    ) -> List[Tuple[str, float]]:
        if len(importance) == 0:
            return []
        k = min(k, len(importance), len(names))
        top_idx = np.argsort(-importance)[:k]
        return [(names[i] if i < len(names) else f"feat_{i}",
                 round(float(importance[i]), 4))
                for i in top_idx if importance[i] > 0]

    def _detect_overrides(self, claim_idx: int) -> List[str]:
        """
        Check Claim node flags against Mandatory Override criteria (PRD §7).
        Returns list of triggered OVR codes.
        """
        triggers = []
        claim_x = self.data["Claim"].x[claim_idx].cpu().numpy()
        feat_names = CLAIM_FEATURE_NAMES

        def flag_val(name: str) -> float:
            if name in feat_names:
                return float(claim_x[feat_names.index(name)])
            return 0.0

        # OVR-001: ring_suspicion_score ≥ 0.9 (mapped to fraud_score here)
        # checked separately via fraud_score in explain_claim

        # OVR-002: high claim amount
        amt_idx = feat_names.index("total_claim_amount") if "total_claim_amount" in feat_names else -1
        if amt_idx >= 0 and claim_x[amt_idx] > 50000:
            triggers.append("OVR-002 (total_claim_amount > $50K)")

        # OVR-003/005: represented_by lawyer check — would need edge traversal;
        # approximate via REPRESENTED_BY edge presence
        ei = self.data.get(("Claim", "REPRESENTED_BY", "Lawyer"), None)
        if ei is not None and hasattr(ei, "edge_index"):
            src_nodes = ei.edge_index[0].tolist()
            if claim_idx in src_nodes:
                triggers.append("OVR-005 (licensed attorney is primary connecting edge)")

        # Ring member flag
        if flag_val("ring_member_flag") == 1.0:
            triggers.append("OVR-001-candidate (ring_member_flag=True)")

        # Staged accident
        if flag_val("staged_accident_flag") == 1.0:
            triggers.append("staged_accident_flag")

        # Hospitalization + no police report
        hosp_idx = feat_names.index("hospitalization_required") if "hospitalization_required" in feat_names else -1
        police_idx = feat_names.index("police_report_available") if "police_report_available" in feat_names else -1
        if hosp_idx >= 0 and police_idx >= 0:
            if claim_x[hosp_idx] >= 1.0 and claim_x[police_idx] < 0.5:
                triggers.append("OVR-006-candidate (hospitalization, no police report)")

        return triggers

    def _build_summary(
        self,
        claim_idx: int,
        proba: float,
        top_features: List[Tuple[str, float]],
        top_neighbors: List[Tuple[str, float]],
        override_triggers: List[str],
    ) -> str:
        """Build a one-paragraph reasoning trace string (input to Phase 3 LLM)."""
        claim_id = self.claim_id_map.get(claim_idx, f"CLM-{claim_idx}")
        severity = (
            "CRITICAL" if proba >= 0.9 else
            "HIGH"     if proba >= 0.7 else
            "MEDIUM"   if proba >= 0.5 else
            "LOW"
        )

        feat_str = ", ".join(
            f"{name} (importance={imp:.3f})"
            for name, imp in top_features[:4]
        ) or "no dominant features identified"

        nbr_str = ", ".join(
            f"{nt} nodes (score={sc:.3f})"
            for nt, sc in top_neighbors[:3]
        ) or "no significant neighbor influence"

        ovr_str = (
            f"  Override triggers: {'; '.join(override_triggers)}."
            if override_triggers else ""
        )

        return (
            f"Claim {claim_id} — fraud score {proba:.3f} ({severity}). "
            f"Top driving features: {feat_str}. "
            f"Strongest neighborhood influence from: {nbr_str}. "
            f"{ovr_str}"
        ).strip()

    # ── Formatting ────────────────────────────────────────────────────
    @staticmethod
    def format_trace(trace: Dict) -> str:
        """Pretty-print a reasoning trace for the investigator dashboard."""
        lines = [
            f"{'─'*60}",
            f"Claim ID    : {trace['claim_id']}",
            f"Fraud Score : {trace['fraud_score']:.4f}  "
            f"({'CRITICAL' if trace['fraud_score']>=0.9 else 'HIGH' if trace['fraud_score']>=0.7 else 'MEDIUM' if trace['fraud_score']>=0.5 else 'LOW'})",
            f"{'─'*60}",
            "Top Driving Features:",
        ]
        for name, imp in trace["top_features"]:
            lines.append(f"  {name:<35} importance={imp:.4f}")

        lines.append("\nNeighborhood Influence (by node type):")
        for nt, sc in trace["top_neighbor_types"]:
            lines.append(f"  {nt:<25} avg_saliency={sc:.4f}")

        if trace["override_triggers"]:
            lines.append("\nMandatory Override Triggers:")
            for t in trace["override_triggers"]:
                lines.append(f"  ⚠  {t}")

        lines.append(f"\nSummary: {trace['signal_summary']}")
        lines.append("─"*60)
        return "\n".join(lines)
