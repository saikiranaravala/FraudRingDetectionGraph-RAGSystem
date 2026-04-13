"""
FraudGNN — Heterogeneous Graph Neural Network for fraud ring detection.

Architecture (two-layer heterogeneous GNN):

  Layer 1 — GraphSAGE (HeteroConv wrapper)
    Inductive learning: handles new nodes without full retraining.
    Each edge type gets its own SAGEConv operator.
    All node types first projected to hidden_channels via type-specific
    linear layers so every conv receives the same-width inputs.

  Layer 2 — HGTConv (Heterogeneous Graph Transformer)
    Equivalent to HINormer: applies type-specific attention weights
    per node-type pair, capturing heterogeneous structural patterns.
    Achieved highest F-scores on insurance heterogeneous datasets.

  Output
    Classification head on Claim node embeddings → sigmoid probability.

References
──────────
  GraphSAGE  : Hamilton et al. (2017), "Inductive Representation Learning
               on Large Graphs"
  HINormer   : Fang et al. (2023), "HINormer: Representation Learning on
               Heterogeneous Information Networks with Graph Transformer"
               — implemented here via PyG's HGTConv which shares the
               same heterogeneous transformer attention architecture.
  GNNExplainer: Ying et al. (2019) — see explainer.py
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HeteroConv, SAGEConv, HGTConv
    from torch_geometric.data import HeteroData
except ImportError as exc:
    raise ImportError(
        "PyTorch Geometric is required. "
        "See requirements_phase2.txt for install instructions."
    ) from exc


class FraudGNN(nn.Module):
    """
    Heterogeneous GNN for fraud ring detection.

    Parameters
    ----------
    in_channels_dict : {node_type: feature_dim}
        Input feature dimension per node type.
    metadata : (node_types, edge_types)
        Graph schema from HeteroData.metadata().
    hidden_channels : int
        Width of all internal representations (default 128).
    out_channels : int
        Width of the final Claim embedding returned by encode() (default 64).
    num_heads : int
        Number of attention heads in the HGTConv layer (default 4).
    dropout : float
        Dropout probability applied after each conv layer (default 0.3).
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        metadata: Tuple[List[str], List[Tuple]],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_p = dropout

        node_types, edge_types = metadata

        # ── Input projections ────────────────────────────────────────
        # Each node type gets its own linear projection to hidden_channels.
        # This decouples input feature dimensions across types.
        self.input_proj = nn.ModuleDict()
        for nt in node_types:
            in_dim = in_channels_dict.get(nt, 1)
            self.input_proj[nt] = nn.Linear(in_dim, hidden_channels)

        # ── Layer 1: GraphSAGE (inductive, handles new nodes) ────────
        # SAGEConv per edge type, all sharing the same (hidden→hidden) dim.
        sage_dict = {}
        for et in edge_types:
            sage_dict[et] = SAGEConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                normalize=True,
            )
        self.sage_conv = HeteroConv(sage_dict, aggr="sum")

        # ── Layer 1 BatchNorm ────────────────────────────────────────
        self.bn1 = nn.ModuleDict({
            nt: nn.BatchNorm1d(hidden_channels) for nt in node_types
        })

        # ── Layer 2: HGTConv (HINormer-equivalent) ───────────────────
        # Type-specific transformer attention per (src_type, edge_type, dst_type).
        # hidden_channels must be divisible by num_heads.
        assert hidden_channels % num_heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by "
            f"num_heads ({num_heads})"
        )
        self.hgt_conv = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads,
        )

        # ── Layer 2 BatchNorm ────────────────────────────────────────
        self.bn2 = nn.ModuleDict({
            nt: nn.BatchNorm1d(hidden_channels) for nt in node_types
        })

        # ── Classification head (Claim nodes only) ───────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1),
        )

        self.dropout = nn.Dropout(dropout)

    # ── Forward passes ────────────────────────────────────────────────
    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute node embeddings for all types.

        Returns
        -------
        x_dict : {node_type: tensor(N, hidden_channels)}
        """
        # 1. Project each node type to hidden_channels
        h = {
            nt: F.relu(self.input_proj[nt](x))
            for nt, x in x_dict.items()
            if nt in self.input_proj
        }

        # 2. GraphSAGE — inductive neighborhood aggregation
        h = self.sage_conv(h, edge_index_dict)
        h = {
            nt: self.dropout(F.relu(self.bn1[nt](emb)))
            for nt, emb in h.items()
            if nt in self.bn1
        }

        # 3. HGTConv — heterogeneous transformer attention (HINormer equiv.)
        h = self.hgt_conv(h, edge_index_dict)
        h = {
            nt: self.dropout(F.relu(self.bn2[nt](emb)))
            for nt, emb in h.items()
            if nt in self.bn2
        }

        return h

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Full forward pass.

        Returns
        -------
        embeddings : {node_type: tensor}   — full node embeddings
        logits     : tensor(N_claims,)     — pre-sigmoid fraud logits
        """
        embeddings = self.encode(x_dict, edge_index_dict)
        logits = self.classifier(embeddings["Claim"]).squeeze(-1)
        return embeddings, logits

    def predict_proba(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Return sigmoid probabilities for Claim nodes."""
        _, logits = self.forward(x_dict, edge_index_dict)
        return torch.sigmoid(logits)

    def get_claim_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Return Claim node embeddings (for ensemble / SMOTE input)."""
        embs = self.encode(x_dict, edge_index_dict)
        return embs["Claim"]


# ── Model factory ─────────────────────────────────────────────────────
def build_model(
    in_channels_dict: Dict[str, int],
    metadata: Tuple,
    cfg: Optional[dict] = None,
) -> FraudGNN:
    """
    Instantiate FraudGNN from a config dict.

    Args:
        in_channels_dict : feature dims per node type (from data_loader meta).
        metadata         : HeteroData.metadata() tuple.
        cfg              : TRAIN_CONFIG dict (or None for defaults).
    """
    from . import config as C
    cfg = cfg or C.TRAIN_CONFIG
    return FraudGNN(
        in_channels_dict=in_channels_dict,
        metadata=metadata,
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
    )


# ── Class-weight helper ───────────────────────────────────────────────
def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    pos_weight = (#negatives) / (#positives)

    A ratio of 10:1 legitimate:fraud → pos_weight = 10 amplifies the
    loss signal for the rare positive class.
    """
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)
