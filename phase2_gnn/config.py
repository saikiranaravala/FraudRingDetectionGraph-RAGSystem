"""
Phase 2 configuration: node types, feature column definitions,
edge types, and training hyperparameters.

Feature columns reference exact CSV column names from data/*.csv.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Graph topology ────────────────────────────────────────────────────
# Node types included in the GNN. These map directly to Neo4j labels
# and nodes_<Label>.csv filenames.
NODE_TYPES = [
    "Claim",
    "Customer",
    "Witness",
    "NetworkFeature",
    "FraudRing",
    "FamilyUnit",
    "InvestigationCase",
    "FinancialTransaction",
    "MedicalReport",
    "Lawyer",
    "RepairShop",
]

# Edge types as (src_label, rel_type, dst_label) tuples.
# Must map to a CSV file in data/ via EDGE_CSV_MAP below.
EDGE_TYPES = [
    ("Customer",             "FILED_CLAIM",           "Claim"),
    ("Customer",             "SHARES_ATTRIBUTE",      "Customer"),
    ("FraudRing",            "RING_CONTAINS_CLAIM",   "Claim"),
    ("FraudRing",            "RING_INVOLVES_CUSTOMER","Customer"),
    ("Claim",                "HAS_WITNESS",           "Witness"),
    ("Claim",                "HAS_MEDICAL_REPORT",    "MedicalReport"),
    ("Claim",                "REPRESENTED_BY",        "Lawyer"),
    ("Claim",                "REPAIRED_AT",           "RepairShop"),
    ("InvestigationCase",    "INVESTIGATES_CLAIM",    "Claim"),
    ("FinancialTransaction", "PAYMENT_FOR_CLAIM",     "Claim"),
    # NetworkFeature→entity edges are split by entity_type in data_loader
    ("NetworkFeature",       "DESCRIBES_CLAIM",       "Claim"),
    ("NetworkFeature",       "DESCRIBES_CUSTOMER",    "Customer"),
    # Family membership
    ("Customer",             "BELONGS_TO_FAMILY",     "FamilyUnit"),
]

# CSV files for each edge type. For polymorphic edges (DESCRIBES_ENTITY,
# BELONGS_TO_FAMILY) the data_loader applies additional filtering.
EDGE_CSV_MAP = {
    ("Customer",             "FILED_CLAIM",           "Claim"):
        "edges_CUSTOMER_TO_CLAIM.csv",
    ("Customer",             "SHARES_ATTRIBUTE",      "Customer"):
        "edges_SHARES_ATTRIBUTE.csv",
    ("FraudRing",            "RING_CONTAINS_CLAIM",   "Claim"):
        "edges_FRAUDRING_TO_CLAIM.csv",
    ("FraudRing",            "RING_INVOLVES_CUSTOMER","Customer"):
        "edges_FRAUDRING_TO_CUSTOMER.csv",
    ("Claim",                "HAS_WITNESS",           "Witness"):
        "edges_CLAIM_TO_WITNESS.csv",
    ("Claim",                "HAS_MEDICAL_REPORT",    "MedicalReport"):
        "edges_CLAIM_TO_MEDICALREPORT.csv",
    ("Claim",                "REPRESENTED_BY",        "Lawyer"):
        "edges_CLAIM_TO_LAWYER.csv",
    ("Claim",                "REPAIRED_AT",           "RepairShop"):
        "edges_CLAIM_TO_REPAIRSHOP.csv",
    ("InvestigationCase",    "INVESTIGATES_CLAIM",    "Claim"):
        "edges_INVESTIGATIONCASE_TO_CLAIM.csv",
    ("FinancialTransaction", "PAYMENT_FOR_CLAIM",     "Claim"):
        "edges_TRANSACTION_TO_CLAIM.csv",
    # Both DESCRIBES_* resolved from same file, filtered by entity_type col
    ("NetworkFeature",       "DESCRIBES_CLAIM",       "Claim"):
        "edges_NETWORKFEATURE_TO_ENTITY.csv",
    ("NetworkFeature",       "DESCRIBES_CUSTOMER",    "Customer"):
        "edges_NETWORKFEATURE_TO_ENTITY.csv",
    ("Customer",             "BELONGS_TO_FAMILY",     "FamilyUnit"):
        "rel_Customer_BELONGS_TO_FAMILY_FamilyUnit.csv",
}

# Node CSV filenames
NODE_CSV_MAP = {
    "Claim":                "nodes_Claim.csv",
    "Customer":             "nodes_Customer.csv",
    "Witness":              "nodes_Witness.csv",
    "NetworkFeature":       "nodes_NetworkFeature.csv",
    "FraudRing":            "nodes_FraudRing.csv",
    "FamilyUnit":           "nodes_FamilyUnit.csv",
    "InvestigationCase":    "nodes_InvestigationCase.csv",
    "FinancialTransaction": "nodes_FinancialTransaction.csv",
    "MedicalReport":        "nodes_MedicalReport.csv",
    "Lawyer":               "nodes_Lawyer.csv",
    "RepairShop":           "nodes_RepairShop.csv",
}

# ID column in each node CSV (used to build the int→str index)
NODE_ID_COL = ":ID"

# ── Feature definitions ───────────────────────────────────────────────
# Exact column names from the CSVs.
# LABEL_COL is excluded from features and used as training target.
LABEL_COL = "fraud_reported"   # 'Y' / 'N' on Claim nodes

# Numeric (continuous / count) features per node type
NUMERIC_FEATURES = {
    "Claim": [
        "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
        "days_to_report", "days_open", "incident_hour_of_the_day",
        "number_of_vehicles_involved", "witnesses",
        "rag_confidence_score", "llm_judge_confidence",
        "total_legal_rubric_score", "evidence_quality_score",
        "liability_probability_score", "claim_velocity_score",
        "network_repeat_score", "shared_entity_count",
        "geo_latitude", "geo_longitude",
        "reserves_set", "salvage_value", "subrogation_recovered",
        "witness_statement_count",
    ],
    "Customer": [
        "age", "months_as_customer", "tenure_years", "fraud_history_count",
        "prior_claims_count", "credit_score_estimate", "risk_score",
        "nps_score", "churn_risk_score", "linked_entity_count",
        "years_at_address", "payment_history_score",
    ],
    "Witness": [
        "reliability_score", "prior_testimony_count",
        "cross_claim_appearance_count", "same_name_claims_count",
        "same_phone_claims_count", "same_address_claims_count",
        "distance_from_scene_ft", "age",
    ],
    "NetworkFeature": [
        "degree_centrality", "betweenness_centrality", "pagerank_score",
        "ring_suspicion_score", "neighbor_fraud_rate", "hop_2_fraud_count",
        "shared_entity_count", "claim_velocity_score", "network_repeat_score",
        "community_id",
    ],
    "FraudRing": [
        "ring_score", "member_count", "claim_count",
        "total_claim_amount", "avg_claim_amount", "fraud_claim_pct",
        "avg_legal_rubric_score", "estimated_total_loss", "recovery_amount_usd",
    ],
    "FamilyUnit": [
        "member_count", "total_claims", "total_claim_value",
        "avg_risk_score", "fraud_member_count", "combined_annual_premium",
        "active_policies", "fraud_member_pct", "claims_per_member",
        "avg_claim_value", "shared_contact_overlap_score",
        "external_connections_count", "cross_family_claims_count",
        "ring_suspicion_score",
    ],
    "InvestigationCase": [
        "ai_fraud_score", "rag_confidence_score", "rubric_score",
        "linked_entities_count", "claim_amount", "estimated_fraud_loss",
        "recovery_amount_usd", "days_open",
    ],
    "FinancialTransaction": [
        "amount_usd", "days_to_payment",
    ],
    "MedicalReport": [
        "treatment_cost", "billed_amount", "insurance_paid_amount",
        "billing_pattern_score", "medical_necessity_score",
        "clinic_repeat_claim_rate", "days_injury_to_treatment",
        "impairment_rating_pct", "physical_therapy_sessions", "inpatient_days",
    ],
    "Lawyer": [
        "shared_clients_count", "referral_network_size",
        "avg_claims_per_client", "bar_discipline_score",
        "years_in_practice", "win_rate_pct", "avg_settlement_usd",
        "active_cases_count", "malpractice_claims",
    ],
    "RepairShop": [
        "estimate_variance_pct", "parts_markup_pct",
        "inflated_estimate_rate_pct", "same_lawyer_referrals",
        "shared_vehicle_count", "avg_repair_cost_usd",
        "siu_referral_count", "supplements_filed_rate_pct",
    ],
}

# Binary (Y/N, Yes/No, True/False, 1/0) features per node type
BINARY_FEATURES = {
    "Claim": [
        "ring_member_flag", "staged_accident_flag", "claim_padding_flag",
        "closed_loop_flag", "manual_override_flag",
        "cross_role_participation_flag", "duplicate_claim_flag",
        "litigation_flag", "arbitration_flag", "role_switching_flag",
        "siu_referral",
    ],
    "Customer": [
        "fraud_flag", "ip_flagged", "shared_phone_flag",
        "shared_address_flag", "shared_bank_flag",
        "synthetic_identity_flag", "role_switching_flag",
        "siu_referral_flag", "watchlist_flag",
        "high_risk_hobby_flag", "identity_theft_flag", "deceased_flag",
    ],
    "Witness": [
        "professional_witness_flag", "coached_statement_flag",
        "criminal_record_flag", "compensation_received_flag",
        "credibility_flag", "contacted_by_attorney", "willing_to_testify",
    ],
    "NetworkFeature": [
        "closed_loop_flag", "cross_role_flag",
    ],
    "FraudRing": [
        "shared_lawyer_flag", "shared_shop_flag",
        "shared_witness_flag", "shared_doctor_flag",
        "closed_loop_detected", "law_enforcement_notified",
        "nicb_case_filed", "recovery_initiated",
    ],
    "FamilyUnit": [
        "fraud_ring_flag",
    ],
    "InvestigationCase": [
        "ring_member", "manual_override_triggered",
        "closed_loop_detected", "recovery_initiated",
    ],
    "FinancialTransaction": [
        "fraud_flag", "velocity_flag", "round_amount_flag",
        "is_shared_account_flag", "suspicious_timing_flag",
    ],
    "MedicalReport": [
        "upcoding_flag", "unbundling_flag", "phantom_billing_flag",
        "surgery_required", "er_visit", "admitted_as_inpatient",
        "causation_confirmed", "imaging_ordered", "pre_existing_condition",
    ],
    "Lawyer": [
        "known_to_siu", "prior_fraud_involvement_flag",
        "closed_loop_network_flag",
    ],
    "RepairShop": [
        "fraud_flag", "inflated_estimate_flag",
    ],
}

# Ordinal categorical features and their encoding maps
ORDINAL_FEATURES = {
    "Claim": {
        "adjuster_priority_tier": {
            "Standard": 0, "High Priority": 1, "Critical": 2, "Urgent": 3
        },
        "incident_severity": {
            "Trivial Damage": 0, "Minor Damage": 1,
            "Major Damage": 2,   "Total Loss": 3
        },
        "police_report_available": {"NO": 0, "?": 0, "YES": 1},
        "hospitalization_required": {"No": 0, "Yes": 1},
    },
    "Customer": {
        "risk_band":        {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
        "customer_tier":    {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3},
        "credit_tier":      {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3},
        "annual_income_band": {
            "< 25000": 0, "25000-50000": 1, "50000-75000": 2,
            "75000-100000": 3, "> 100000": 4,
        },
    },
    "FraudRing": {
        "status": {"Under Watch": 0, "Suspected": 1, "Confirmed": 2},
    },
    "FamilyUnit": {
        "family_risk_band": {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
    },
    "InvestigationCase": {
        "status":   {"Open": 0, "In Progress": 1, "Closed": 2},
        "priority": {"Low": 0, "Medium": 1, "High": 2, "Critical": 3},
    },
}

# ── Training hyperparameters ──────────────────────────────────────────
TRAIN_CONFIG = {
    "hidden_channels":  128,
    "out_channels":     64,
    "num_heads":        4,          # HGTConv attention heads
    "dropout":          0.3,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "epochs":           150,
    "patience":         20,         # early stopping patience (val AUC)
    "train_ratio":      0.70,
    "val_ratio":        0.15,
    # test_ratio = 1 - train - val = 0.15
    "random_seed":      42,
    "batch_size":       None,       # None = full-batch (graph fits in memory)
}

# Ensemble config
ENSEMBLE_CONFIG = {
    "xgb": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "auc",
        "use_label_encoder": False,
    },
    "rf": {
        "n_estimators": 200,
        "max_depth": 8,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "lgbm": {
        "n_estimators": 300,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "verbose": -1,
    },
}
