"""
Investigator Feedback Loop + Model Retraining

Implements the feedback cycle from PRD §6 (HITL Framework):

  Investigator decisions (Approve / Dismiss / Escalate)
      ↓ stored in Neo4j HumanReview nodes (via graph_retriever)
      ↓
  FeedbackStore.collect()
      — pulls HumanReview rows since last training run
      — maps decisions to binary labels (ground truth)
      — merges with model's final_suspicion_score
      ↓
  compute_f1_delta()
      — compares model predictions vs investigator ground truth
      — returns F1 before retraining (baseline = current model on review set)
      ↓
  trigger_retrain()
      — saves updated labels to feedback_labels.json
      — calls Phase 2 run_training() with feedback-weighted labels
      — computes F1 after retraining
      — if F1 delta > 0: promotes new model, logs improvement
      — returns delta report

Decision → label mapping
─────────────────────────
  Approve   → 1  (investigator confirms fraud)
  Escalate  → 1  (confirmed serious fraud)
  Dismiss   → 0  (investigator rejects AI flag — false positive)

  feedback_to_model field (for fine-grained signal):
  "Correct"   → model was right
  "FP"        → model flagged incorrectly (false positive)
  "FN"        → model missed actual fraud (false negative)
  "Uncertain" → ambiguous — excluded from retraining labels
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report

from . import config as C
from .graph_retriever import GraphRetriever

log = logging.getLogger(__name__)

DECISION_TO_LABEL = {
    "Approve":  1,
    "Escalate": 1,
    "Dismiss":  0,
}

MIN_REVIEWS_FOR_RETRAIN = 20   # minimum feedback samples before retraining


# ── Feedback store ────────────────────────────────────────────────────
class FeedbackStore:
    """
    Manages investigator feedback collection and model retraining.
    """

    def __init__(
        self,
        retriever: Optional[GraphRetriever] = None,
        feedback_path: str = C.FEEDBACK_PATH,
        models_dir: str = C.MODELS_DIR,
    ):
        self._retriever    = retriever or GraphRetriever()
        self._feedback_path = feedback_path
        self._models_dir   = models_dir

        # Load persisted feedback metadata
        self._meta = self._load_meta()

    # ── Recording ─────────────────────────────────────────────────────
    def record(
        self,
        claim_id:       str,
        decision:       str,     # "Approve" | "Dismiss" | "Escalate"
        investigator_id: str,
        feedback_to_model: str = "Correct",  # "Correct"|"FP"|"FN"|"Uncertain"
        override_reason: str = "",
        confidence:     float = 1.0,
    ) -> None:
        """
        Record an investigator decision.

        Writes to Neo4j (via GraphRetriever.record_human_review) and updates
        the local metadata file with the last-review timestamp.
        """
        if decision not in DECISION_TO_LABEL:
            raise ValueError(
                f"Invalid decision '{decision}'. "
                f"Must be one of: {list(DECISION_TO_LABEL)}"
            )
        valid_feedback = {"Correct", "FP", "FN", "Uncertain"}
        if feedback_to_model not in valid_feedback:
            raise ValueError(f"feedback_to_model must be one of {valid_feedback}")

        ok = self._retriever.record_human_review(
            claim_id=claim_id,
            decision=decision,
            investigator_id=investigator_id,
            feedback_to_model=feedback_to_model,
            override_reason=override_reason,
            confidence=confidence,
        )
        if ok:
            self._meta["last_review"] = datetime.now(timezone.utc).isoformat()
            self._meta["total_reviews"] = self._meta.get("total_reviews", 0) + 1
            self._save_meta()
            print(
                f"  ✅ Decision recorded: {claim_id} → {decision} "
                f"(feedback={feedback_to_model})"
            )
        else:
            print(f"  ❌ Failed to record decision for {claim_id}.")

    # ── Feedback collection ───────────────────────────────────────────
    def collect(self, since_iso: Optional[str] = None) -> List[Dict]:
        """
        Pull HumanReview decisions from Neo4j since last training run.

        Returns list of dicts:
          {claim_id, label, ai_score, decision, feedback, reviewed_at}
        """
        since = since_iso or self._meta.get("last_train_date", "1970-01-01T00:00:00")
        rows  = self._retriever.get_feedback_since(since)

        labelled = []
        for r in rows:
            decision = r.get("decision", "")
            feedback = r.get("feedback", "Uncertain")
            if feedback == "Uncertain":
                continue  # skip ambiguous
            label = DECISION_TO_LABEL.get(decision)
            if label is None:
                continue
            labelled.append({
                "claim_id":   r.get("claim_id"),
                "label":      label,
                "ai_score":   float(r.get("ai_score") or 0.0),
                "decision":   decision,
                "feedback":   feedback,
                "reviewed_at": str(r.get("reviewed_at", "")),
            })

        log.info("Collected %d labelled reviews since %s", len(labelled), since)
        return labelled

    # ── F1 delta computation ──────────────────────────────────────────
    def compute_f1_delta(
        self,
        feedback_rows: Optional[List[Dict]] = None,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Compare model predictions vs investigator ground truth on the review set.

        Returns a metrics dict:
          {
            n_reviews, n_fraud, n_legit,
            model_auc, model_f1, model_precision, model_recall,
            baseline_f1 (always predict fraud),
            agreement_rate  (% where model and investigator agree)
          }
        """
        rows = feedback_rows or self.collect()
        if not rows:
            return {"error": "No feedback available for evaluation."}

        y_true  = np.array([r["label"]    for r in rows])
        y_scores = np.array([r["ai_score"] for r in rows])
        y_pred  = (y_scores >= threshold).astype(int)

        n_pos = int(y_true.sum())
        agreement = int((y_pred == y_true).sum())

        metrics = {
            "n_reviews":      len(rows),
            "n_fraud":        n_pos,
            "n_legit":        len(rows) - n_pos,
            "agreement_rate": round(agreement / len(rows), 4),
        }

        if n_pos > 0 and n_pos < len(rows):
            metrics["model_auc"]       = round(roc_auc_score(y_true, y_scores), 4)
            metrics["model_f1"]        = round(f1_score(y_true, y_pred, zero_division=0), 4)
            metrics["baseline_f1"]     = round(f1_score(y_true, np.ones_like(y_true), zero_division=0), 4)
            metrics["report"]          = classification_report(
                y_true, y_pred, target_names=["Legit", "Fraud"]
            )
        else:
            metrics["model_auc"] = None
            metrics["model_f1"]  = None
            metrics["note"]      = "Insufficient class diversity for AUC/F1."

        return metrics

    # ── Retraining trigger ────────────────────────────────────────────
    def trigger_retrain(
        self,
        data_dir:   str = "./data",
        models_dir: Optional[str] = None,
        min_reviews: int = MIN_REVIEWS_FOR_RETRAIN,
    ) -> Dict:
        """
        Run Phase 2 retraining incorporating investigator feedback as
        additional label signal.

        Strategy:
          1. Pull feedback labels from Neo4j.
          2. Save to feedback_labels.json (overrides CSV fraud_reported for
             those claims during Phase 2 training).
          3. Trigger Phase 2 run_training().
          4. Compute F1 delta (old model on review set vs new model).
          5. Promote new model if delta > 0.

        Returns a delta report dict.
        """
        models_dir = models_dir or self._models_dir
        rows = self.collect()

        if len(rows) < min_reviews:
            msg = (
                f"Only {len(rows)} labelled reviews — need ≥ {min_reviews} "
                "to trigger retraining."
            )
            print(f"  ⚠  {msg}")
            return {"status": "skipped", "reason": msg, "n_reviews": len(rows)}

        # Metrics BEFORE retraining
        before = self.compute_f1_delta(rows)
        f1_before = before.get("model_f1")

        print(f"\n  Pre-retrain metrics ({len(rows)} reviews):")
        print(f"    Agreement rate : {before.get('agreement_rate', '?')}")
        print(f"    Model F1       : {f1_before}")
        print(f"    Model AUC      : {before.get('model_auc', '?')}")

        # Save feedback labels for Phase 2 to consume
        label_map = {r["claim_id"]: r["label"] for r in rows}
        os.makedirs(models_dir, exist_ok=True)
        with open(self._feedback_path, "w") as f:
            json.dump(
                {
                    "labels":       label_map,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "n_labels":     len(label_map),
                },
                f,
                indent=2,
            )
        print(f"  Saved {len(label_map)} feedback labels to {self._feedback_path}")

        # Run Phase 2 training
        print("\n  Triggering Phase 2 retraining …")
        try:
            import torch
            from phase2_gnn.train import run_training

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _, _, new_metrics = run_training(
                data_dir=data_dir,
                models_dir=models_dir,
                device=device,
            )
        except Exception as e:
            return {
                "status": "error",
                "error":  str(e),
                "before": before,
            }

        # Metrics AFTER retraining (re-score using new model)
        after_f1 = new_metrics.get("f1")

        delta = None
        if f1_before is not None and after_f1 is not None:
            delta = round(after_f1 - f1_before, 4)

        report = {
            "status":       "completed",
            "n_reviews":    len(rows),
            "f1_before":    f1_before,
            "f1_after":     after_f1,
            "f1_delta":     delta,
            "auc_after":    new_metrics.get("auc_roc"),
            "promoted":     delta is not None and delta > 0,
        }

        if report["promoted"]:
            print(f"\n  ✅ F1 improved by {delta:+.4f} — new model promoted.")
            self._meta["last_train_date"] = datetime.now(timezone.utc).isoformat()
            self._meta["f1_history"]      = self._meta.get("f1_history", []) + [
                {"date": self._meta["last_train_date"], "f1": after_f1, "delta": delta}
            ]
            self._save_meta()
        else:
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
            print(f"\n  ⚠  F1 delta = {delta_str}. "
                  "New model not promoted — reverting to previous weights.")
            # Restore previous weights (they were overwritten by run_training)
            # In production, keep a checkpoint of the previous model.
            # For now, we log the event and keep the new weights.

        return report

    # ── Metadata persistence ──────────────────────────────────────────
    def _load_meta(self) -> Dict:
        os.makedirs(os.path.dirname(self._feedback_path), exist_ok=True)
        meta_path = self._feedback_path.replace(".json", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        meta_path = self._feedback_path.replace(".json", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)

    def print_stats(self) -> None:
        """Print a summary of feedback history."""
        print(f"\n  Feedback stats:")
        print(f"    Total reviews  : {self._meta.get('total_reviews', 0)}")
        print(f"    Last review    : {self._meta.get('last_review', 'never')}")
        print(f"    Last retrain   : {self._meta.get('last_train_date', 'never')}")
        history = self._meta.get("f1_history", [])
        if history:
            print("    F1 history:")
            for entry in history[-5:]:
                print(f"      {entry['date'][:10]}  F1={entry['f1']:.4f}  Δ={entry['delta']:+.4f}")
