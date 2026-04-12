"""
Feature extraction utilities.

Converts raw DataFrame columns (from CSVs) into float32 numpy arrays
ready for PyTorch Geometric node feature tensors.

Handles:
  - Numeric columns (int / float) — NaN → 0
  - Binary columns with mixed string representations (Y/N, Yes/No, True/False, 1/0)
  - Ordinal categoricals with explicit encoding maps
  - StandardScaler fitting on train split, applied to val/test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

# ── Binary encoding ───────────────────────────────────────────────────
_TRUTHY = {"y", "yes", "true", "1", "t"}

def bool_encode(val) -> float:
    """Convert Y/N / Yes/No / True/False / 1/0 / bool to float 0.0 or 1.0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    if isinstance(val, (bool, np.bool_)):
        return 1.0 if val else 0.0
    if isinstance(val, (int, np.integer)):
        return float(bool(val))
    return 1.0 if str(val).strip().lower() in _TRUTHY else 0.0


def ordinal_encode(val, mapping: dict) -> float:
    """Map a raw value to its ordinal int using the provided mapping."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    # Try direct key first, then stripped string
    if val in mapping:
        return float(mapping[val])
    key_str = str(val).strip()
    return float(mapping.get(key_str, 0))


# ── Feature matrix builder ────────────────────────────────────────────
def build_feature_matrix(
    df: pd.DataFrame,
    numeric_cols: List[str],
    binary_cols: List[str],
    ordinal_cols: Optional[Dict[str, dict]] = None,
) -> np.ndarray:
    """
    Build a float32 (N, F) feature matrix from a node DataFrame.

    Missing columns are silently filled with zeros so the feature
    dimension stays constant regardless of which CSV we are processing.

    Args:
        df:           DataFrame of node properties (one row per node).
        numeric_cols: Columns to coerce to float (NaN → 0).
        binary_cols:  Columns to encode as 0/1 via bool_encode().
        ordinal_cols: {col_name: {raw_val: int, ...}} — ordinal encoding map.

    Returns:
        np.ndarray of shape (len(df), F), dtype float32.
    """
    parts: list[np.ndarray] = []

    for col in numeric_cols:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        else:
            v = np.zeros(len(df), dtype=np.float32)
        parts.append(v.reshape(-1, 1))

    for col in binary_cols:
        if col in df.columns:
            v = df[col].apply(bool_encode).to_numpy(dtype=np.float32)
        else:
            v = np.zeros(len(df), dtype=np.float32)
        parts.append(v.reshape(-1, 1))

    if ordinal_cols:
        for col, mapping in ordinal_cols.items():
            if col in df.columns:
                v = df[col].apply(lambda x, m=mapping: ordinal_encode(x, m)).to_numpy(dtype=np.float32)
            else:
                v = np.zeros(len(df), dtype=np.float32)
            parts.append(v.reshape(-1, 1))

    if not parts:
        # Node type has no configured features — return a single all-zero column
        # so it can still participate in message passing.
        return np.zeros((len(df), 1), dtype=np.float32)

    return np.hstack(parts).astype(np.float32)


# ── Scaler fitting / transform ────────────────────────────────────────
def fit_scaler(
    train_matrix: np.ndarray,
) -> StandardScaler:
    """Fit a StandardScaler on the training node features."""
    scaler = StandardScaler()
    scaler.fit(train_matrix)
    return scaler


def scale_matrix(matrix: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler, handling all-zero columns gracefully."""
    return scaler.transform(matrix).astype(np.float32)


# ── Label extraction ─────────────────────────────────────────────────
def extract_labels(df: pd.DataFrame, label_col: str = "fraud_reported") -> np.ndarray:
    """
    Extract binary fraud labels from a Claim DataFrame.

    Returns float32 array of shape (N,) with 1.0 = fraud, 0.0 = legitimate.
    Rows where label_col is missing are treated as 0.
    """
    if label_col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return df[label_col].apply(bool_encode).to_numpy(dtype=np.float32)


# ── Feature dimension registry ────────────────────────────────────────
def feature_dim(
    numeric_cols: List[str],
    binary_cols: List[str],
    ordinal_cols: Optional[Dict[str, dict]] = None,
) -> int:
    """Return the number of feature columns this config will produce."""
    n_ordinal = len(ordinal_cols) if ordinal_cols else 0
    total = len(numeric_cols) + len(binary_cols) + n_ordinal
    return max(total, 1)  # always at least 1 (the zero-column fallback)
