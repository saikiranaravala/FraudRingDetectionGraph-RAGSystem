"""
Phase 2 training pipeline.

Steps
─────
1. Load full graph (HeteroData) with stratified Claim splits.
2. Train FraudGNN with:
   - BCEWithLogitsLoss + pos_weight  (handles class imbalance in the GNN)
   - Early stopping on val AUC-ROC
3. Extract GNN embeddings for Claim nodes (train split).
4. Apply SMOTE + ADASYN to augment minority class in embedding space.
   ⚠️  Oversampling is TRAINING TIME ONLY — never applied at inference.
5. Train ensemble (XGBoost + RandomForest + LightGBM) on augmented embeddings.
6. Evaluate on test split: AUC-ROC, F1, precision, recall.
7. Save model artefacts to models/.

Usage
─────
    python run_phase2.py train
    python run_phase2.py train --epochs 200 --lr 5e-4 --data-dir ./data
"""

from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, classification_report,
)
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger(__name__)

# ── Optional imports (graceful degradation) ───────────────────────────
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    log.warning("imbalanced-learn not installed — SMOTE/ADASYN disabled. "
                "Install with: pip install imbalanced-learn")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from torch_geometric.data import HeteroData
except ImportError as exc:
    raise ImportError("PyTorch Geometric required. See requirements_phase2.txt.") from exc

from . import config as C
from .data_loader import load_graph_splits
from .model import FraudGNN, build_model, compute_pos_weight


# ── Training loop ─────────────────────────────────────────────────────
def train_gnn(
    model: FraudGNN,
    data: HeteroData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    cfg: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Train FraudGNN with BCEWithLogitsLoss + early stopping on val AUC.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_auc', 'best_epoch'
    """
    cfg = cfg or C.TRAIN_CONFIG
    device = device or torch.device("cpu")
    model = model.to(device)

    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    labels = data["Claim"].y.to(device)

    pos_weight = compute_pos_weight(labels[train_mask]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "best_epoch": 0}

    print(f"\n{'='*55}")
    print("TRAINING FraudGNN")
    print(f"{'='*55}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {cfg['epochs']}  |  Patience: {cfg['patience']}")
    print(f"  LR:         {cfg['lr']}  |  pos_weight: {pos_weight.item():.2f}")
    print(f"  Train/Val:  {train_mask.sum()} / {val_mask.sum()} claims")
    print(f"{'='*55}\n")

    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train step ──────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        _, logits = model(x_dict, edge_index_dict)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = loss.item()
        history["train_loss"].append(train_loss)

        # ── Validation step ─────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            _, val_logits = model(x_dict, edge_index_dict)
            val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
            val_labels = labels[val_mask].cpu().numpy()

        if val_labels.sum() > 0:
            val_auc = roc_auc_score(val_labels, val_probs)
        else:
            val_auc = 0.5
        history["val_auc"].append(val_auc)

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>4}  loss={train_loss:.4f}  "
                  f"val_auc={val_auc:.4f}  best={best_val_auc:.4f}  "
                  f"[{elapsed:.0f}s]")

        if patience_counter >= cfg["patience"]:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg['patience']} epochs)")
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  Restored best weights from epoch {history['best_epoch']} "
              f"(val AUC={best_val_auc:.4f})")

    return history


# ── Embedding extraction + SMOTE ─────────────────────────────────────
def extract_embeddings(
    model: FraudGNN,
    data: HeteroData,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run GNN encode pass and return Claim embeddings as numpy array."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    with torch.no_grad():
        embs = model.get_claim_embeddings(x_dict, ei_dict)
    return embs.cpu().numpy()


def apply_smote_adasyn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment minority class (fraud) using SMOTE followed by ADASYN.

    ⚠️  TRAINING TIME ONLY — never call this function on val/test data.

    Strategy:
      1. SMOTE  — oversample to 50% of majority class count
      2. ADASYN — add extra samples near the decision boundary
    """
    if not HAS_IMBALANCED:
        log.warning("imbalanced-learn not installed — skipping SMOTE/ADASYN. "
                    "Training ensemble on original embeddings.")
        return X_train, y_train

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos

    if n_pos == 0:
        log.warning("No positive samples in training set — skipping SMOTE.")
        return X_train, y_train

    print(f"\n  SMOTE/ADASYN: {n_pos} fraud / {n_neg} legitimate (before)")

    # SMOTE: bring minority up to ~50% of majority
    smote_target = max(n_pos, int(n_neg * 0.5))
    try:
        smote = SMOTE(
            sampling_strategy={1: smote_target},
            k_neighbors=min(5, n_pos - 1) if n_pos > 1 else 1,
            random_state=random_state,
        )
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
    except Exception as e:
        log.warning("SMOTE failed (%s) — using original data.", e)
        X_sm, y_sm = X_train, y_train

    # ADASYN: add boundary-focused samples
    try:
        adasyn = ADASYN(
            sampling_strategy=0.8,
            n_neighbors=min(5, int(y_sm.sum()) - 1) if int(y_sm.sum()) > 1 else 1,
            random_state=random_state,
        )
        X_aug, y_aug = adasyn.fit_resample(X_sm, y_sm)
    except Exception as e:
        log.warning("ADASYN failed (%s) — using SMOTE output only.", e)
        X_aug, y_aug = X_sm, y_sm

    n_pos_after = int(y_aug.sum())
    print(f"  SMOTE/ADASYN: {n_pos_after} fraud / {len(y_aug)-n_pos_after} "
          f"legitimate (after)")
    return X_aug, y_aug


# ── Ensemble training ─────────────────────────────────────────────────
def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Optional[dict] = None,
) -> dict:
    """
    Train XGBoost + RandomForest + LightGBM on GNN embeddings.

    All three classifiers receive the SMOTE-augmented training embeddings.
    Final score = mean of the three probability outputs.

    Returns
    -------
    ensemble : dict with keys 'xgb', 'rf', 'lgbm', 'val_auc'
    """
    cfg = cfg or C.ENSEMBLE_CONFIG
    models = {}

    print(f"\n  Training ensemble on {X_train.shape[0]} samples "
          f"({int(y_train.sum())} fraud) …")

    # XGBoost
    if HAS_XGB:
        xgb_params = {k: v for k, v in cfg["xgb"].items()
                      if k != "use_label_encoder"}
        clf_xgb = xgb.XGBClassifier(
            **xgb_params,
            scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
            random_state=42,
            verbosity=0,
        )
        clf_xgb.fit(X_train, y_train)
        models["xgb"] = clf_xgb
        xgb_auc = roc_auc_score(y_val, clf_xgb.predict_proba(X_val)[:, 1])
        print(f"    XGBoost  val AUC = {xgb_auc:.4f}")
    else:
        log.warning("xgboost not installed — skipping XGBoost.")

    # Random Forest
    clf_rf = RandomForestClassifier(**cfg["rf"], random_state=42)
    clf_rf.fit(X_train, y_train)
    models["rf"] = clf_rf
    rf_auc = roc_auc_score(y_val, clf_rf.predict_proba(X_val)[:, 1])
    print(f"    RF       val AUC = {rf_auc:.4f}")

    # LightGBM — train with plain numpy so no feature names are stored
    if HAS_LGBM:
        X_train_np = np.asarray(X_train, dtype=np.float32)
        X_val_np   = np.asarray(X_val,   dtype=np.float32)
        clf_lgbm = lgb.LGBMClassifier(**cfg["lgbm"], random_state=42)
        clf_lgbm.fit(
            X_train_np, y_train,
            eval_set=[(X_val_np, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        models["lgbm"] = clf_lgbm
        lgbm_auc = roc_auc_score(y_val, clf_lgbm.predict_proba(X_val_np)[:, 1])
        print(f"    LightGBM val AUC = {lgbm_auc:.4f}")
    else:
        log.warning("lightgbm not installed — skipping LightGBM.")

    # Ensemble AUC (mean of available classifiers)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names",
                                category=UserWarning)
        proba_list = [
            m.predict_proba(np.asarray(X_val, dtype=np.float32))[:, 1]
            for m in models.values()
        ]
    ensemble_proba = np.mean(proba_list, axis=0)
    ensemble_auc = roc_auc_score(y_val, ensemble_proba)
    print(f"    Ensemble val AUC = {ensemble_auc:.4f}")

    models["val_auc"] = ensemble_auc
    return models


# ── Evaluation ────────────────────────────────────────────────────────
def evaluate(
    model: FraudGNN,
    ensemble: dict,
    data: HeteroData,
    test_mask: torch.Tensor,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate ensemble on the held-out test split.

    Returns
    -------
    metrics : dict with AUC-ROC, F1, precision, recall, avg_precision
    """
    device = device or torch.device("cpu")

    # GNN probabilities
    model.eval()
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    with torch.no_grad():
        gnn_proba = model.predict_proba(x_dict, ei_dict).cpu().numpy()

    # Ensemble probabilities on test embeddings
    test_embs = extract_embeddings(model, data, device)
    idx_test = test_mask.numpy().nonzero()[0]
    X_test = test_embs[idx_test]
    y_test = data["Claim"].y.numpy()[idx_test]

    ensemble_models = {k: v for k, v in ensemble.items() if k != "val_auc"}
    if ensemble_models:
        X_test_np = np.asarray(X_test, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names",
                                    category=UserWarning)
            ens_probas = np.mean(
                [m.predict_proba(X_test_np)[:, 1] for m in ensemble_models.values()],
                axis=0,
            )
        # Blend GNN and ensemble (equal weight)
        final_proba = 0.5 * gnn_proba[idx_test] + 0.5 * ens_probas
    else:
        final_proba = gnn_proba[idx_test]

    preds = (final_proba >= threshold).astype(int)

    metrics = {
        "auc_roc":    roc_auc_score(y_test, final_proba),
        "avg_prec":   average_precision_score(y_test, final_proba),
        "f1":         f1_score(y_test, preds, zero_division=0),
        "precision":  precision_score(y_test, preds, zero_division=0),
        "recall":     recall_score(y_test, preds, zero_division=0),
        "n_test":     len(y_test),
        "n_positive": int(y_test.sum()),
    }

    print(f"\n{'='*55}")
    print("TEST SET RESULTS")
    print(f"{'='*55}")
    print(f"  AUC-ROC         : {metrics['auc_roc']:.4f}  (target ≥ 0.91)")
    print(f"  Avg Precision   : {metrics['avg_prec']:.4f}")
    print(f"  F1 Score        : {metrics['f1']:.4f}")
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  Test size       : {metrics['n_test']}  ({metrics['n_positive']} fraud)")
    print(f"\n{classification_report(y_test, preds, target_names=['Legit', 'Fraud'])}")

    return metrics


# ── Serialization ─────────────────────────────────────────────────────
def save_artefacts(
    model: FraudGNN,
    ensemble: dict,
    history: dict,
    metrics: dict,
    meta: dict,
    models_dir: str = C.MODELS_DIR,
) -> str:
    """Save model weights, ensemble, scaler, and metadata to models/."""
    import joblib, json
    os.makedirs(models_dir, exist_ok=True)

    # GNN weights
    gnn_path = os.path.join(models_dir, "fraud_gnn.pt")
    torch.save(model.state_dict(), gnn_path)

    # Ensemble
    ens_path = os.path.join(models_dir, "ensemble.joblib")
    ensemble_save = {k: v for k, v in ensemble.items() if k != "val_auc"}
    joblib.dump(ensemble_save, ens_path)

    # Metadata (in_channels, metadata, metrics, history)
    meta_save = {
        "in_channels":  meta["in_channels"],
        "train_config": C.TRAIN_CONFIG,
        "history":      history,
        "metrics":      metrics,
    }
    meta_path = os.path.join(models_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_save, f, indent=2, default=str)

    print(f"\n  Artefacts saved to {models_dir}/")
    print(f"    {os.path.basename(gnn_path)}")
    print(f"    {os.path.basename(ens_path)}")
    print(f"    {os.path.basename(meta_path)}")
    return models_dir


def load_artefacts(
    data: HeteroData,
    meta: dict,
    models_dir: str = C.MODELS_DIR,
    device: Optional[torch.device] = None,
) -> Tuple[FraudGNN, dict]:
    """Load saved GNN + ensemble from models/."""
    import joblib
    device = device or torch.device("cpu")

    model = build_model(meta["in_channels"], data.metadata())
    gnn_path = os.path.join(models_dir, "fraud_gnn.pt")
    model.load_state_dict(torch.load(gnn_path, map_location=device))
    model = model.to(device).eval()

    ens_path = os.path.join(models_dir, "ensemble.joblib")
    ensemble = joblib.load(ens_path) if os.path.exists(ens_path) else {}

    return model, ensemble


# ── Full pipeline ─────────────────────────────────────────────────────
def run_training(
    data_dir: str = C.DATA_DIR,
    models_dir: str = C.MODELS_DIR,
    device: Optional[torch.device] = None,
    cfg: Optional[dict] = None,
) -> Tuple[FraudGNN, dict, dict]:
    """
    End-to-end training pipeline.

    Returns (model, ensemble, metrics)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg or C.TRAIN_CONFIG

    # 1. Load data
    data, train_mask, val_mask, test_mask, meta = load_graph_splits(
        data_dir=data_dir, verbose=True
    )

    # 2. Build and train GNN
    model = build_model(meta["in_channels"], data.metadata(), cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  FraudGNN parameters: {n_params:,}")

    history = train_gnn(model, data, train_mask, val_mask, cfg, device)

    # 3. Extract embeddings (training nodes only)
    all_embs = extract_embeddings(model, data, device)
    idx_train = train_mask.numpy().nonzero()[0]
    idx_val   = val_mask.numpy().nonzero()[0]
    labels    = data["Claim"].y.numpy()

    X_train_raw = all_embs[idx_train]
    y_train     = labels[idx_train]
    X_val       = all_embs[idx_val]
    y_val       = labels[idx_val]

    # 4. SMOTE + ADASYN on training embeddings (training time only)
    X_train_aug, y_train_aug = apply_smote_adasyn(
        X_train_raw, y_train, random_state=cfg["random_seed"]
    )

    # 5. Train ensemble
    ensemble = train_ensemble(X_train_aug, y_train_aug, X_val, y_val)

    # 6. Evaluate on test set
    metrics = evaluate(model, ensemble, data, test_mask, device)

    # 7. Save artefacts
    save_artefacts(model, ensemble, history, metrics, meta, models_dir)

    return model, ensemble, metrics
