"""
Data loader: reads CSVs from data/ and builds a PyTorch Geometric
HeteroData object ready for heterogeneous GNN training.

Graph construction
──────────────────
Nodes:  11 types (Claim, Customer, Witness, …) — see config.NODE_TYPES
Edges:  13 typed relations — see config.EDGE_TYPES

Labels: Claim.fraud_reported ('Y' / 'N') → float32 tensor {0, 1}

Train / val / test masks are created on Claim nodes via
stratified split (config.TRAIN_CONFIG ratios).

Usage
─────
    from phase2_gnn.data_loader import build_hetero_data, load_graph_splits
    data, meta = build_hetero_data()
    train_data, val_data, test_data, meta = load_graph_splits()
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from . import config as C
from .feature_utils import (
    build_feature_matrix,
    extract_labels,
    feature_dim,
    fit_scaler,
    scale_matrix,
)

try:
    from torch_geometric.data import HeteroData
except ImportError as exc:
    raise ImportError(
        "PyTorch Geometric is required.  "
        "Install it following the instructions in requirements_phase2.txt."
    ) from exc

log = logging.getLogger(__name__)

# ── Node index management ─────────────────────────────────────────────
NodeIndex = Dict[str, Dict[str, int]]  # label → {raw_id → int_idx}


def _load_node_df(label: str, data_dir: str) -> pd.DataFrame:
    fname = C.NODE_CSV_MAP[label]
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Node CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _build_node_index(df: pd.DataFrame, id_col: str = C.NODE_ID_COL) -> Dict[str, int]:
    """Map string IDs in a node CSV to zero-based integer indices."""
    return {str(row[id_col]): idx for idx, row in df.iterrows()}


def _load_edge_csv(edge_type: Tuple[str, str, str], data_dir: str) -> Optional[pd.DataFrame]:
    fname = C.EDGE_CSV_MAP.get(edge_type)
    if not fname:
        return None
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        log.warning("Edge CSV missing: %s", path)
        return None
    return pd.read_csv(path, low_memory=False)


def _edge_index_tensor(
    df: pd.DataFrame,
    src_index: Dict[str, int],
    dst_index: Dict[str, int],
    filter_col: Optional[str] = None,
    filter_val: Optional[str] = None,
) -> torch.Tensor:
    """
    Build a (2, E) int64 edge index tensor from a CSV edge DataFrame.

    Args:
        df:          Edge DataFrame with ':START_ID' and ':END_ID' columns.
        src_index:   {raw_id → int} for source node type.
        dst_index:   {raw_id → int} for destination node type.
        filter_col:  Optional column to filter on before building index.
        filter_val:  Value to keep when filter_col is specified.

    Returns:
        LongTensor of shape (2, E). Edges whose endpoints are absent
        from the index are silently dropped.
    """
    if df is None or df.empty:
        return torch.zeros((2, 0), dtype=torch.long)

    rows = df.copy()
    if filter_col and filter_val:
        rows = rows[rows[filter_col].astype(str) == filter_val]

    srcs, dsts = [], []
    for _, row in rows.iterrows():
        s = str(row[":START_ID"])
        d = str(row[":END_ID"])
        if s in src_index and d in dst_index:
            srcs.append(src_index[s])
            dsts.append(dst_index[d])

    if not srcs:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor([srcs, dsts], dtype=torch.long)


# ── Main builder ──────────────────────────────────────────────────────
def build_hetero_data(
    data_dir: str = C.DATA_DIR,
    verbose: bool = True,
) -> Tuple[HeteroData, dict]:
    """
    Load all node CSVs and edge CSVs and build a PyG HeteroData.

    Returns
    -------
    data : HeteroData
        Node feature tensors (x), Claim labels (y), edge indices.
    meta : dict
        {
          'node_index':  NodeIndex,       # label → {id → int}
          'claim_df':    pd.DataFrame,    # original Claim CSV (for IDs)
          'labels':      np.ndarray,      # Claim fraud labels (float32)
          'in_channels': {label: int},    # feature dim per node type
        }
    """
    if verbose:
        print("Loading graph from CSVs …")

    data = HeteroData()
    node_index: NodeIndex = {}
    in_channels: Dict[str, int] = {}
    claim_df: Optional[pd.DataFrame] = None

    # ── Node features ─────────────────────────────────────────────────
    for label in C.NODE_TYPES:
        try:
            df = _load_node_df(label, data_dir)
        except FileNotFoundError as e:
            log.warning(str(e))
            continue

        if label == "Claim":
            claim_df = df.copy()

        idx = _build_node_index(df)
        node_index[label] = idx

        num_cols = C.NUMERIC_FEATURES.get(label, [])
        bin_cols = C.BINARY_FEATURES.get(label, [])
        ord_cols = C.ORDINAL_FEATURES.get(label, {})

        # Exclude the label column from Claim features
        if label == "Claim":
            bin_cols = [c for c in bin_cols if c != C.LABEL_COL]

        mat = build_feature_matrix(df, num_cols, bin_cols, ord_cols or None)
        in_channels[label] = mat.shape[1]

        data[label].x = torch.from_numpy(mat)
        data[label].num_nodes = len(df)

        if verbose:
            print(f"  {label:22s}  {len(df):>5} nodes  {mat.shape[1]:>3} features")

    # ── Claim labels ──────────────────────────────────────────────────
    if claim_df is None:
        raise RuntimeError("nodes_Claim.csv could not be loaded.")

    labels = extract_labels(claim_df, C.LABEL_COL)
    data["Claim"].y = torch.from_numpy(labels)

    if verbose:
        pos = int(labels.sum())
        print(f"\n  Claim labels: {pos} fraud / {len(labels)-pos} legitimate "
              f"({100*pos/len(labels):.1f}% positive)\n")

    # ── Edges ─────────────────────────────────────────────────────────
    for edge_type in C.EDGE_TYPES:
        src_label, rel, dst_label = edge_type
        if src_label not in node_index or dst_label not in node_index:
            continue

        edf = _load_edge_csv(edge_type, data_dir)

        # Polymorphic DESCRIBES_ENTITY: filter by entity_type column
        filter_col, filter_val = None, None
        if rel == "DESCRIBES_CLAIM":
            filter_col, filter_val = "entity_type", "Claim"
        elif rel == "DESCRIBES_CUSTOMER":
            filter_col, filter_val = "entity_type", "Customer"

        ei = _edge_index_tensor(
            edf,
            node_index[src_label],
            node_index[dst_label],
            filter_col=filter_col,
            filter_val=filter_val,
        )
        data[src_label, rel, dst_label].edge_index = ei

        if verbose:
            print(f"  {src_label} -[{rel}]-> {dst_label}: {ei.shape[1]} edges")

    # Add reverse edges so message passing flows in both directions
    # (important for Claim nodes to receive information from neighbors)
    rev_types_added = set()
    for src_label, rel, dst_label in C.EDGE_TYPES:
        if dst_label == src_label:
            continue  # self-loops already bidirectional
        rev_rel = f"rev_{rel}"
        key = (dst_label, rev_rel, src_label)
        if key in rev_types_added:
            continue
        fwd_key = (src_label, rel, dst_label)
        if hasattr(data[src_label, rel, dst_label], "edge_index"):
            fwd_ei = data[src_label, rel, dst_label].edge_index
            if fwd_ei.shape[1] > 0:
                data[dst_label, rev_rel, src_label].edge_index = fwd_ei.flip(0)
                rev_types_added.add(key)

    if verbose:
        print(f"\n  Reverse edges added: {len(rev_types_added)}")

    meta = {
        "node_index": node_index,
        "claim_df":   claim_df,
        "labels":     labels,
        "in_channels": in_channels,
    }
    return data, meta


# ── Train / val / test split on Claim nodes ───────────────────────────
def load_graph_splits(
    data_dir: str = C.DATA_DIR,
    verbose: bool = True,
) -> Tuple[HeteroData, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build the full HeteroData and return stratified train/val/test masks
    for Claim nodes.

    Returns
    -------
    data         : HeteroData (full graph — all nodes, all edges)
    train_mask   : bool tensor of shape (N_claims,)
    val_mask     : bool tensor of shape (N_claims,)
    test_mask    : bool tensor of shape (N_claims,)
    meta         : dict from build_hetero_data()
    """
    cfg = C.TRAIN_CONFIG
    seed = cfg["random_seed"]

    data, meta = build_hetero_data(data_dir=data_dir, verbose=verbose)
    labels = meta["labels"]
    n = len(labels)
    indices = np.arange(n)

    train_ratio = cfg["train_ratio"]
    val_ratio   = cfg["val_ratio"]

    # Stratified split: train vs (val + test)
    idx_train, idx_temp = train_test_split(
        indices,
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=seed,
    )
    # Split val+test into val and test
    val_frac_of_temp = val_ratio / (1 - train_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1 - val_frac_of_temp,
        stratify=labels[idx_temp],
        random_state=seed,
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val]     = True
    test_mask[idx_test]   = True

    if verbose:
        print(f"\n  Split — train: {train_mask.sum()} | "
              f"val: {val_mask.sum()} | test: {test_mask.sum()}")

    return data, train_mask, val_mask, test_mask, meta
