# Feedback & Retraining Loop

Continuous learning cycle from investigator decisions to model promotion.

```mermaid
flowchart TD
    INV["Investigator Decision\nApprove · Dismiss · Escalate\nconfidence · feedback_signal"]

    INV --> HR["HumanReview Node\nwritten to Neo4j\nreview_id · claim_id · investigator_id\ndecision · confidence\nfeedback_to_model: Correct/FP/FN/Uncertain\noverride_ai_recommendation: bool\ntimestamp"]

    HR --> FS["FeedbackStore.collect()\nsrc/services/feedback.py\npull HumanReview nodes\nsince last_training_run"]

    FS --> LABEL["Map decisions → binary labels\nApprove  → label = 1\nEscalate → label = 1\nDismiss  → label = 0\nUncertain → excluded"]

    LABEL --> COUNT{">= min_reviews\n(default 20)?"}

    COUNT -->|"No"| WAIT["Accumulate\nmore reviews\nlog count"]

    COUNT -->|"Yes"| F1DELTA["compute_f1_delta()\ncompare model predictions\nvs investigator ground truth\nF1 before / F1 after estimate"]

    F1DELTA --> TRIGGER["trigger_retrain()\nsrc/services/feedback.py\nsave feedback_labels.json\nsave feedback_labels_meta.json"]

    TRIGGER --> LOAD["Reload CSVs\nsrc/services/gnn/data_loader.py\napply feedback label overrides\nfeedback_labels.json\ntakes priority over CSV labels"]

    LOAD --> SMOTE["SMOTE + ADASYN\nsrc/services/gnn/train.py\napply_smote_adasyn()\ntraining split ONLY\nnever val / test / inference"]

    SMOTE --> TRAIN["run_training()\nBCEWithLogitsLoss\npos_weight = N_neg / N_pos\nSAGEConv + HGTConv\n+ XGB + RF + LGBM ensemble"]

    TRAIN --> EVAL["evaluate_model()\nF1 · AUC-ROC · Precision · Recall\non held-out val + test splits"]

    EVAL --> DELTA{"F1 delta\n> 0?"}

    DELTA -->|"Yes — improved"| PROMOTE["Promote new model\noverwrite models/fraud_gnn.pt\noverwrite models/ensemble.pkl\noverwrite models/scaler.pkl\nlog promotion event"]

    DELTA -->|"No — regression"| KEEP["Keep current model\nlog delta for audit\nalert compliance officer"]

    PROMOTE --> RESCORE["Re-score all Claims\npython src/main.py gnn score\nupdate Neo4j properties\ngnn_suspicion_score\nensemble_suspicion_score\nfinal_suspicion_score\nadjuster_priority_tier"]

    RESCORE --> LOOP(["Next cycle begins\nmonitor new reviews"])
```

## Feedback Label File Format

```mermaid
flowchart LR
    subgraph FILES["models/ — Feedback Artefacts"]
        direction TB
        F1["feedback_labels.json\n{\n  'CLM-001': 1,\n  'CLM-002': 0,\n  'CLM-003': 1\n}\nclaim_id → binary label\noverrides CSV labels"]

        F2["feedback_labels_meta.json\n{\n  'last_training_run': '2026-04-13',\n  'review_count': 47,\n  'f1_before': 0.821,\n  'f1_after': 0.849,\n  'delta': +0.028,\n  'promoted': true\n}"]
    end
```

## Retraining Triggers

```mermaid
flowchart LR
    subgraph AUTO["Automatic (CLI)"]
        A1["python src/main.py rag retrain\n-- checks >= 20 reviews\n-- runs full cycle"]
        A2["python src/main.py rag retrain\n--min-reviews 10\n-- lower threshold for testing"]
        A3["python src/main.py rag retrain\n--evaluate-only\n-- F1 metrics, no retraining"]
    end

    subgraph MANUAL["Manual (Override)"]
        M1["python src/main.py gnn train\n-- full retrain from scratch\n-- does not check review count"]
        M2["python src/main.py gnn evaluate\n-- re-evaluate saved model\n-- no retraining"]
    end

    subgraph GUARD["Safety Guards"]
        G1["Uncertain labels excluded\nfrom feedback_labels.json"]
        G2["F1 gate — no promotion\nif model regresses"]
        G3["SMOTE never applied\nto val/test/production data"]
        G4["Compliance log written\nfor every retrain attempt"]
    end
```

## F1 History — Stats Tab

```mermaid
flowchart LR
    subgraph STATS["Streamlit Stats Tab"]
        direction TB
        ST1["Total Reviews: 47"]
        ST2["Pending Retraining: 7\n(since last run)"]
        ST3["Vector Store Size: 20 rings"]
        ST4["F1 History Table\n| Date       | F1 Before | F1 After | Delta | Promoted |\n|------------|-----------|----------|-------|----------|\n| 2026-04-13 | 0.821     | 0.849    | +0.028 | Yes     |\n| 2026-03-28 | 0.798     | 0.821    | +0.023 | Yes     |"]
    end

    STATS -->|"GET /stats"| API["FastAPI\n/stats endpoint\nreads feedback_labels_meta.json\n+ Neo4j review count\n+ Pinecone vector count"]
