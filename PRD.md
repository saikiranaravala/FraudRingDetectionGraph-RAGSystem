# Fraud Ring Detection: Graph-RAG System

## PRD v3.0 — Auto Insurance · Production Architecture

> **Confidential · Internal Use Only**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Strategic Objectives](#2-strategic-objectives)
3. [Why Graph-Based Approaches](#3-why-graph-based-approaches)
4. [System Architecture](#4-system-architecture)
5. [Graph Data Model](#5-graph-data-model)
6. [Human-in-the-Loop Framework](#6-human-in-the-loop-framework)
7. [Mandatory Override Criteria](#7-mandatory-override-criteria)
8. [Technology Stack](#8-technology-stack)
9. [Success Metrics](#9-success-metrics)
10. [Cross-Carrier Collaboration](#10-cross-carrier-collaboration)
11. [Compliance & Ethics](#11-compliance--ethics)
12. [Phased Rollout](#12-phased-rollout)

---

## 1. Executive Summary

Insurance fraud costs the industry an estimated **$300 billion annually** — roughly 10% of all insurance spending globally. Fraud is set to worsen in 2025, driven by a 30% rate among people under 45 who don't view insurance fraud as a crime, combined with a sharp rise in exaggerated claims, often exacerbated by attorneys who prioritize maximizing settlements.

Most existing defenses remain **reactive and investigator-intuition dependent**.

This PRD defines a best-in-class, production-grade **Fraud Ring Detection system** for auto insurance — one that combines Graph Neural Networks, GraphRAG, a Streamlit investigator UI, real-time streaming, and a rigorous Human-in-the-Loop framework to detect coordinated fraud rings **before claims are paid**.

### Key Numbers

| Metric | Value |
| ------ | ----- |
| Annual industry fraud loss | ~$300B (10% of all insurance spend) |
| GNN recall gain vs. baseline | +19.7% (RL-GNN fusion) |
| False positive reduction | 33% vs. baseline models |
| Inference latency | 42ms per batch (near real-time) |
| Target AUC-ROC at launch | >= 0.91 |
| Target AUC-ROC at month 6 | >= 0.95 |

---

## 2. Strategic Objectives

1. **Identify coordinated fraud rings** by analyzing indirect, multi-hop relationships across claimants, lawyers, witnesses, and repair shops — connections invisible to traditional rule-based systems.

2. **Shift from reactive to pre-payment detection.** Real-time monitoring with machine learning and graph analytics flags and investigates suspicious claims before payments are made, reducing losses significantly.

3. **Build explainable AI, not a black box.** Every suspicion score must be traceable to specific data points so investigators can perform a meaningful sanity check — not just accept or reject a number.

4. **Enforce Human-in-the-Loop accountability at every decision gate.** No automated action is taken without a named human signature.

5. **Learn continuously.** Investigator feedback on every reviewed lead becomes training signal for model improvement.

---

## 3. Why Graph-Based Approaches

Traditional ML treats each claim in isolation. Increasingly sophisticated fraudsters exploit this by working together and leveraging false identities. The significant opportunity lies in **looking beyond individual data points to the connections that link them** — connections that oftentimes hold the best clues.

### Research Findings

- **GNNs** capture and interpret intricate patterns by considering both node features and connections between nodes — processing complex relationships and dependencies within a dataset.
- **Multi-channel heterogeneous graph structure learning** captures diverse graph-based features from different claims to model complex relationships for increasing detection accuracy.
- **RL-GNN fusion models** demonstrate a 19.7% gain in recall and a 33% reduction in false positives vs. baseline GNN models, with near real-time inference averaging 42ms per batch.
- **GraphRAG** (Microsoft Research) weaves knowledge graphs directly into retrieval pipelines, enabling LLMs to connect relationships — not just retrieve facts — making it especially powerful for fraud detection's complex inference demands.

### Top Fraud Ring Signals

| Signal | Severity | Detection Method |
| ------ | -------- | ---------------- |
| Shared phone / address across unrelated claimants | CRITICAL | Graph traversal — `SHARES_ATTRIBUTE` edges |
| Same attorney + repair shop across >= 3 claims in 30 days | CRITICAL | Pattern query — closed-loop detection |
| Shared bank account / payment routing hash | CRITICAL | `shared_bank_flag` + `SHARES_ATTRIBUTE` |
| Professional witness (3+ claims, different incidents) | HIGH | `professional_witness_flag` + `same_name_claims_count` |
| Geo-cluster incidents within defined radius | HIGH | `geo_latitude/longitude` + `claim_cluster_id` |
| Cross-jurisdiction attorney as primary link node | HIGH | OVR-005 mandatory override trigger |
| Staged accident pattern | HIGH | `staged_accident_flag` + `closed_loop_flag` |
| Synthetic identity / role-switching | MEDIUM | `synthetic_identity_flag` + `role_switching_flag` |

---

## 4. System Architecture

### 4.1 Deployment Topology

```text
GitHub Repository
    |
    ├──► Render.com (free plan)
    │        api.py — FastAPI backend
    │        build.sh — CPU torch + requirements.txt
    │        render.yaml — service definition
    │        URL: https://your-service.onrender.com
    │
    └──► Streamlit Community Cloud (free)
             ui/streamlit_app.py — investigator UI
             ui/requirements.txt — streamlit, requests, pandas
             URL: https://your-app.streamlit.app
```

### 4.2 Data Ingestion Layer

#### Batch pipeline

- Historical claims data, vendor registry, prior fraud ring indices, court records, attorney licensing databases
- Spark-based ETL via **AWS Glue + EMR** → Neo4j via JDBC connector
- Covers 3+ years of historical data for model training

#### Real-time streaming (Phase 4)

- Live FNOL events, claim submissions, repair shop invoices
- **Amazon Kinesis → Neo4j Kafka Connector**
- Graph updated within seconds — enables pre-payment intervention

#### External enrichment

- Cross-carrier consortium feeds
- Geospatial incident data
- DMV records, attorney bar association APIs
- Every node enriched at ingestion time

#### Identity resolution

- Hashed phone numbers, address normalization, VIN fingerprinting
- IP clustering, bank account hashing
- Durable entity identifiers that survive data entry variation

---

### 4.3 Detection Engine — Three-Layer Model

#### Layer 1 — Rule & Heuristic Signals

> **Speed:** <100ms · Always-on · No ML required

Triggered immediately on claim ingestion. Deterministic rules run on every claim entering the graph.

**Flags triggered by:**

- Same attorney + shop appearing across >= 3 claims in 30 days
- Duplicate phone/address across unrelated claimants
- Incident geo-cluster within defined radius
- Repair estimate outlier vs. vehicle damage photos

Outputs a **signal score** per entity that feeds as input features into Layer 2. Rules updated quarterly as ring tactics evolve.

---

#### Layer 2 — Graph Neural Network Scoring

> **Speed:** 42ms per batch · Scalable to 500K+ transactions

**Primary architecture: GraphSAGE** (Graph Sample and Aggregate)

- Inductive learning — handles new nodes without full retraining
- Scales well to production graph sizes

**Augmented with: HINormer** (Heterogeneous Information Network Transformer)

- Handles heterogeneous node types natively
- Achieved highest F-scores on small and large-scale claims datasets

**How it works:**

1. GNN computes a **suspicion embedding** for every entity
2. Scores propagate through the network via connected edges
3. A single new fraudulent claim elevates scores for all connected entities
4. Ring detection emerges via score diffusion across the graph

**Class imbalance handling:**

- SMOTE + ADASYN applied at **training time only** (never inference)
- VAE/GAN synthetic minority generation for rare ring pattern types
- Hybrid ensemble: XGBoost + Random Forest + LightGBM

---

#### Layer 3 — GraphRAG Reasoning

> **Triggered:** On HITL flag · LLM-powered explanation & retrieval

When a ring is flagged for human review, the **LangGraph agentic pipeline**:

1. Retrieves the subgraph of connected entities from Neo4j
2. Fetches analogous historical ring cases from Pinecone (384-dim cosine search)
3. Generates a **natural-language reasoning trace** via Claude (OpenRouter)

**Example reasoning trace output:**
> *"This ring shares structural patterns with Ring #FR-2023-0047, in which the same law firm appeared across claims in non-contiguous jurisdictions. Three witnesses appear in both rings. Repair estimates at SHOP-004 are 34% above baseline for this incident type."*

Rather than a binary fraud/no-fraud label, the system produces **citable, structured natural language** the investigator can interrogate — asking follow-up questions directly against the live graph via the Streamlit UI.

**Explainability:** Gradient saliency (d_fraud_score / d_features) attached to every GNN prediction — identifies the critical subgraph edges driving each score.

---

### 4.4 Investigator UI — Streamlit Frontend

A four-tab web application deployed to Streamlit Community Cloud:

| Tab | Purpose |
| --- | ------- |
| **Investigate Claim** | Enter claim ID → fraud score banner + override triggers + analogous rings + Claude brief |
| **Graph Query** | Natural language question → Cypher → Neo4j → NL answer |
| **Record Feedback** | Submit Approve / Dismiss / Escalate with confidence score and override reason |
| **Stats & History** | Total reviews, vector store entries, F1 retraining history |

The UI calls the Render.com FastAPI backend over HTTPS. No ML or DB code runs in the frontend — it is purely a thin HTTP client.

---

### 4.5 Imbalanced Data Strategy

Fraud cases are rare relative to legitimate claims. Strategy:

| Approach | Tool | When Applied |
| -------- | ---- | ------------ |
| Oversampling | SMOTE + ADASYN | Training time only |
| Synthetic generation | VAE / GAN / Diffusion Models | Training time only |
| Ensemble classification | XGBoost + Random Forest + LightGBM | Inference |
| Threshold tuning | Per-class calibration | Post-training |

> **Warning:** Synthetic minority oversampling must **never** be applied at inference time.

---

## 5. Graph Data Model

Scale: 14,292 nodes · 28,690 edges · 941 properties · 24 node types · 28 edge types

### 5.1 Node Reference

| Node Type | Count | Properties | Key Fraud Signals |
| --------- | ----- | ---------- | ----------------- |
| `Customer` | 1,000 | 109 | `ssn_hash`, `device_id`, `ip_address_last_login`, `bank_account_hash`, `shared_phone_flag`, `shared_bank_flag`, `synthetic_identity_flag`, `role_switching_flag`, `ip_flagged`, `fraud_history_count` |
| `Claim` | 1,000 | 111 | `geo_latitude`, `geo_longitude`, `claim_cluster_id`, `ring_member_flag`, `staged_accident_flag`, `claim_padding_flag`, `closed_loop_flag`, `rag_confidence_score`, `llm_judge_verdict`, `manual_override_flag`, `manual_override_reasons` |
| `FraudRing` | 20 | 30 | `ring_score`, `status` (Confirmed/Suspected/Under Watch), `member_count`, `total_claim_amount`, `detection_method`, `closed_loop_detected`, `law_enforcement_notified`, `nicb_case_filed` |
| `Witness` | 1,487 | 42 | `professional_witness_flag`, `same_name_claims_count`, `same_phone_claims_count`, `coached_statement_flag`, `reliability_score`, `prior_testimony_count`, `criminal_record_flag` |
| `InvestigationCase` | 876 | 33 | `ai_fraud_score`, `explanation_trace_id`, `rag_evidence_nodes`, `manual_override_triggered`, `evidence_collected`, `recovery_initiated`, `nicb_case_filed` |
| `HumanReview` | 570 | 27 | `decision`, `decision_confidence`, `override_ai_recommendation`, `disagreement_flag`, `override_reason_code`, `feedback_to_model` (Correct/FP/FN/Uncertain) |
| `NetworkFeature` | 2,000 | 17 | `degree_centrality`, `betweenness_centrality`, `pagerank_score`, `community_id`, `ring_suspicion_score`, `neighbor_fraud_rate`, `hop_2_fraud_count` |
| `Event` | 1,000 | 23 | Single incident event linking Claim + Witness + MedicalReport. `event_type`, `geo_coordinates`, `severity`, `weather`, `ring_member`, `cluster_id` |
| `FinancialTransaction` | 1,516 | 25 | `payment_channel`, `is_shared_account_flag`, `routing_number_hash`, `velocity_flag`, `round_amount_flag`, `suspicious_timing_flag` |
| `MedicalReport` | 992 | 48 | `icd10_code`, `upcoding_flag`, `unbundling_flag`, `phantom_billing_flag`, `billing_pattern_score`, `medical_necessity_score`, `provider_npi` |
| `Policy` | 1,000 | 36 | `coverage_csl`, `deductible_amount`, `annual_premium`, `umbrella_limit`, `endorsements`, `renewal_count` |
| `Vehicle` | 1,000 | 48 | `vin_hash`, `telematics_risk_score`, `auction_sourced_flag`, `previous_owners_count`, `odometer_rollback_flag`, `open_recalls` |
| `Lawyer` | 7 | 54 | `shared_clients_count`, `referral_network_size`, `avg_claims_per_client`, `bar_discipline_score`, `closed_loop_network_flag`, `known_to_siu` |
| `RepairShop` | 7 | 55 | `estimate_variance_pct`, `parts_markup_pct`, `inflated_estimate_rate_pct`, `same_lawyer_referrals`, `shared_vehicle_count`, `fraud_flag` |
| `PoliceOfficer` | 589 | 32 | `badge_number`, `rank`, `department`, `accident_reconstruction_certified`, `cases_handled_ytd` |
| `Doctor` | 6 | 47 | `npi_number`, `malpractice_claims_count`, `insurance_fraud_flag`, `prior_siu_referral`, `avg_treatment_cost_per_claim` |
| `Hospital` | 6 | 46 | `trauma_level`, `insurance_fraud_investigations_ytd`, `malpractice_suits_ytd`, `patient_satisfaction_score` |
| `FamilyUnit` | 233 | 21 | `fraud_member_count`, `fraud_ring_flag`, `ring_suspicion_score`, `shared_contact_overlap_score`, `claims_per_member` |
| `Agent` | 945 | 28 | `fraud_referral_count_ytd`, `retention_rate_pct`, `claims_ratio_pct`, `performance_tier` |
| `Agency` | 6 | 28 | `fraud_referrals_ytd`, `regulatory_complaints_ytd`, `audit_result` |
| `Contractor` | 5 | 48 | `osha_citations_total`, `work_zone_violations_count`, `negligence_findings_count`, `prior_safety_incidents` |
| `DrinkingVenue` | 5 | 44 | `dram_shop_lawsuits_count`, `prior_violations_count`, `dui_incidents_linked`, `claims_linked_count`, `tips_certified_staff_pct` |
| `LiabilityPattern` | 8 | 14 | `recovery_likelihood`, `avg_recovery_pct`, `legal_complexity`, `statute_of_limitations_years` |
| `VehicleMake` | 14 | 13 | `fraud_rate_pct`, `avg_claim_amount`, `total_loss_rate_pct`, `active_recalls_2015` |

### 5.2 Key Edge Types

| Edge Type | Count | Key Properties |
| --------- | ----- | -------------- |
| `CUSTOMER_RELATIONSHIPS` | 7,974 | `relationship_type`, `confidence_score`, `weight`, `first_seen_date`, `last_seen_date`, `interaction_count` |
| `SHARES_ATTRIBUTE` | 417 | `attribute_type`, `attribute_risk_score`, `fraud_signal`, `global_frequency`, `connection_strength` |
| `RING_CONTAINS_CLAIM` | 150 | `role_in_ring`, `weight`, `connection_strength`, `event_time` |
| `RING_INVOLVES_CUSTOMER` | 150 | `member_role`, `fraud_signal`, `weight`, `interaction_count` |
| `HAS_WITNESS` | 1,487 | `statement_strength`, `credibility_score`, `professional_witness`, `coached_flag` |
| `REPRESENTED_BY` | 941 | `engagement_date`, `case_type`, `settlement_amount`, `litigation_initiated` |
| `REPAIRED_AT` | 1,000 | `repair_cost`, `repair_days`, `supplement_filed`, `estimate_submitted` |
| `HAS_MEDICAL_REPORT` | 992 | `injury_type`, `treatment_cost`, `billing_flag`, `causation_link` |
| `EVENT_TO_ENTITIES` | 3,479 | `weight`, `connection_strength`, `event_time` |

**All edges carry:** `weight`, `connection_strength`, `first_seen_date`, `last_seen_date`, `event_time`, `interaction_count`, `shared_count`

### 5.3 Fraud Ring Detection Patterns (Stress-Tested)

```text
Witnesses in 3+ claims:        263 witnesses  (top: Jason Evans = 9 claims)
Shared bank account customers: 5+  (bank_account_hash match)
Ring member claims:            150  across 20 rings
Closed loop claims:            76
Manual override triggers:      776  claims
Fraud-ring family units:       50
FraudRing nodes confirmed:     20
Staged accident flags:         25  claims
Synthetic identity flags:      8   customers
Role-switching customers:      19
```

---

## 6. Human-in-the-Loop Framework

> **Core principle:** The system is an investigative intelligence tool, not a decision-making system. No automated action is taken without a named investigator signature.

### 6.1 HITL Decision Flow

```text
FNOL Event
    |
    v
Graph Updated (Kinesis → Neo4j, <1 sec)
    |
    v
Layer 1: Rule Engine (<100ms)
    |
    v
Layer 2: GNN Scoring (42ms/batch)
    |
    v
Triage Tier Assigned
    ├── Standard       → Queue for review (<30 min SLA)
    ├── High Priority  → Priority queue (<4 hr SLA)
    └── Mandatory Override → Payment HOLD + Senior review
    |
    v
Layer 3: GraphRAG Fires
    └── LangGraph: subgraph retrieval → analogous cases → reasoning trace
    |
    v
Streamlit Investigator UI
    ├── Investigate Claim tab  → view score, override triggers, Claude brief
    ├── Graph Query tab        → ask questions in plain English
    ├── Record Feedback tab    → Approve / Dismiss / Escalate
    └── Stats tab              → review queue metrics, F1 history
    |
    v
Weekly Retraining Cycle (feedback → F1 delta measurement)
```

### 6.2 Investigator Responsibilities

- You are the **final authority**. If the AI surfaces a connection your experience tells you is coincidental, you have both the power and the professional responsibility to override it.
- Your daily Approve / Dismiss / Escalate decisions — with documented reasoning — are the **training data for the next model version**.
- Do not click through without reasoning. Every decision must include a rationale.

### 6.3 Project Leader Responsibilities

- Enforce that no automated action is taken without a named investigator signature.
- Monitor feedback quality metrics — track feedback_to_model distribution (Correct / FP / FN / Uncertain).
- Invest in the skill transition from "data hunters" to "AI validators."
- Run quarterly bias audits on model output distributions.

---

## 7. Mandatory Override Criteria

These conditions **halt all automated progression**. A named senior investigator must review and sign off before any investigative action, payment hold, or vendor contact proceeds.

| Code | Trigger | Condition | Rationale |
| ---- | ------- | --------- | --------- |
| OVR-001 | Score threshold | Ring suspicion score >= 0.90 | Highest scores carry greatest consequences — fraud exposure AND wrongful accusation risk |
| OVR-002 | Financial exposure | Projected ring exposure > $75K–$100K | Dollar thresholds are legally defensible and easy to audit |
| OVR-003 | Cross-jurisdiction | Same attorney, witness, or shop across claims in different regulatory jurisdictions | Legal complexity demands senior review before any contact |
| OVR-004 | Ring contradiction | Entity from a previously dismissed ring reappears in current ring | Contradiction must be resolved: improved signal vs. known false positive |
| OVR-005 | Attorney link | Licensed attorney is the primary connecting edge in the ring | Requires mandatory review AND legal team consultation before contact |
| OVR-006 | Vulnerable claimant | Elderly, recent bereavement, cognitive impairment noted, language assistance required | Mandatory review before any investigative action or contact |
| OVR-007 | Reputational exposure | Public figure, prominent local business, or prior media coverage in ring | Requires escalation before any action |
| OVR-008 | Prior SIU referral | Any node in the ring has a prior SIU referral on record (regardless of outcome) | Automatically elevates the full ring to mandatory review |

### Claim-Level Auto-Triggers

Claims are also flagged for manual override when any of the following conditions are met:

```python
manual_override_flag = True if any([
    total_claim_amount > 50000,                          # High value
    rag_confidence_score < 0.6 and fraud_reported == 'Y', # Low AI confidence + fraud flag
    hospitalization_required == 'Yes'
        and police_report_available in ('NO', '?'),       # Injury, no police report
    claim_cluster_id != '',                               # Ring cluster member
]) else False
```

---

## 8. Technology Stack

| Layer | Tool | Rationale |
| ----- | ---- | --------- |
| **Graph DB** | Neo4j Aura (GDB + GDS) | Native graph storage, Cypher queries, built-in GDS algorithms (PageRank, community detection, shortest path) |
| **GNN Framework** | PyTorch Geometric + GraphSAGE / HINormer | Best-in-class heterogeneous graph support. GraphSAGE: inductive learning (new nodes without retraining). HINormer: highest F-scores on insurance datasets |
| **Explainability** | Gradient saliency (d_score / d_features) | Identifies critical subgraph features driving each score → investigator Reasoning Trace panel |
| **Streaming Ingest** | Amazon Kinesis + Neo4j Kafka Connector | Sub-second claim-to-graph latency. Enables pre-payment interception (Phase 4) |
| **Batch ETL** | AWS Glue + EMR | Scalable bulk historical load. 3+ years of claims enrichment, vendor registry joins, court record cross-references |
| **Orchestration** | LangGraph | Multi-step agentic RAG pipeline: graph traversal → vector retrieval → LLM reasoning in a single auditable chain |
| **Embeddings** | fastembed (ONNX) + BAAI/bge-small-en-v1.5 | 384-dim, no PyTorch required. ~100 MB RAM vs. ~380 MB (sentence-transformers). Fits Render free plan. |
| **Vector KB** | Pinecone | Fast cosine similarity retrieval for analogous historical ring cases. 384-dim index, free starter tier. Persists across Render deploys. |
| **LLM Reasoning** | Claude or Gemma via OpenRouter (openai-compatible API) | Claude: best quality, pay-per-token. Gemma: free (with Google API key). Explanation generation, NL querying, reasoning trace. Investigators ask graph questions in plain English. |
| **Oversampling** | SMOTE + ADASYN | Class imbalance correction at training time only. Never applied to val/test/inference. |
| **Frontend** | Streamlit | 4-tab investigator UI: Investigate / Query / Feedback / Stats. Deployed on Streamlit Community Cloud (free). |
| **Web API** | FastAPI + uvicorn | REST API wrapping the GraphRAG pipeline. Deployed on Render.com (free plan, 512 MB RAM, fastembed fits). |
| **ML Serving** | Amazon SageMaker | Model hosting, weekly retraining pipeline, A/B testing between model versions (Phase 4) |

---

## 9. Success Metrics

| Metric | Launch Target | Month 6 Target | Why It Matters |
| ------ | ------------- | -------------- | -------------- |
| Human Agreement Rate | > 65% | > 80% | % of AI-flagged rings confirmed by investigators. Primary signal quality measure. |
| False Positive Rate | < 35% | < 20% | % of flagged rings dismissed. High FPR burns investigator trust and bandwidth. |
| Pre-payment Interception Rate | Baseline TBD | Maximize | Highest-value metric for loss ratio impact. % of confirmed rings caught before payment exits. |
| Mean Time to Decision (Standard) | < 30 min | < 20 min | Investigator queue-entry to Approve/Dismiss SLA. |
| Mean Time to Decision (Override) | < 4 hours | < 2 hours | Senior review SLA. Balances rigor with pre-payment speed. |
| F1-Score Delta per Retraining | Positive | Consistent growth | Measures feedback loop effectiveness. Stagnating F1 signals feedback quality issue. |
| Model AUC-ROC | >= 0.91 | >= 0.95 | Production benchmark. Literature: 0.961 AUC achieved in comparable GNN fraud tasks. |
| Investigator Confidence Score | Baseline survey | Trend upward | Quarterly survey. Leading indicator of tool adoption and trust. |

---

## 10. Cross-Carrier Collaboration

> **This is the highest-leverage capability available in 2025.** A single carrier sees only its own claims — fraud rings exploit this blind spot deliberately by spreading activity across carriers.

### Design Principles

- **Design from day one** to participate in a trusted carrier intelligence exchange
- Share **anonymized entity fingerprints** only: hashed attorney bar numbers, shop EINs, phone number hashes, VIN fingerprints — never raw PII
- **Incoming** alerts from consortium partners enrich the local graph in real time
- **Outgoing** confirmed ring data (post-investigator approval only) is contributed back to the exchange

### Network Effect

As more carriers participate, the graph becomes exponentially more powerful. A ring that exploits a single-carrier blind spot becomes visible the moment any participating carrier flags one of its members.

---

## 11. Compliance & Ethics

### No Automated Adverse Action

The system never triggers a coverage denial, claim rejection, or legal referral without human sign-off. All automated outputs are labeled **investigative leads**, not determinations.

### Bias Monitoring

GNN models trained on historical fraud data may encode geographic or demographic biases present in past investigative patterns.

- **Quarterly bias audits** on model outputs by zip code, claimant demographics, and attorney demographics
- Track and report: are certain demographic groups flagged at disproportionate rates?
- Audit results reviewed by compliance officer before each model version is promoted to production

### Data Minimization

- Only data fields **necessary for fraud detection** are ingested into the graph
- PII handling follows state insurance regulatory requirements and applicable data protection law
- Raw PII (SSNs, full bank accounts) are **never stored** — only hashed representations

### Audit Trail

Every AI output, investigator decision, override, and feedback signal is logged **immutably** with timestamp and user ID. This log is the chain of accountability for any regulatory inquiry.

### Right to Explanation

If the system's outputs ever inform an adverse action, the investigator's **documented reasoning** — not the AI's score — is the record of decision for regulatory and legal purposes.

---

## 12. Phased Rollout

### Phase 1 — Foundation (Complete)

**Goal:** Operational graph database with rule-based detection and investigator workflow live.

- [x] Neo4j Aura provisioned and populated with historical claims (14,292 nodes · 28,690 edges)
- [x] Identity resolution pipeline active (phone/address/VIN/IP hashing)
- [x] Rule-based signal layer (Layer 1) live on all new claim ingestion
- [x] Mandatory Override criteria enforced
- [x] Immutable audit log operational

---

### Phase 2 — GNN Layer (Complete)

**Goal:** ML-driven suspicion scoring integrated into the investigator workflow.

- [x] GraphSAGE model trained on historical data with SMOTE/ADASYN balancing
- [x] HINormer layer added for heterogeneous node types
- [x] Ensemble: XGBoost + Random Forest + LightGBM on GNN embeddings
- [x] GNN scores written to Neo4j (`gnn_suspicion_score`, `final_suspicion_score`, `adjuster_priority_tier`)
- [x] Gradient saliency explainer for reasoning traces
- [x] Score tiers: Critical (>= 0.90) / High Priority (>= 0.70) / Standard (< 0.70)

---

### Phase 3 — GraphRAG + Agentic Layer (Complete)

**Goal:** Natural language querying, analogous ring retrieval, and feedback loop live.

- [x] LangGraph StateGraph pipeline: retrieve_subgraph → find_analogous_rings → generate_reasoning
- [x] Pinecone vector knowledge base populated (384-dim, 20 FraudRing embeddings, fastembed-indexed)
- [x] Natural language querying live via NL → Cypher → Neo4j → NL summary
- [x] LLM reasoning via OpenRouter: Claude (paid) or Gemma (free with Google key)
- [x] Feedback loop: Approve / Dismiss / Escalate → HumanReview node → retraining trigger
- [x] Unified CLI: `python src/main.py rag index | explain | query | feedback | retrain | stats`

---

### Phase 3b — Production Deployment & Optimization (Complete)

**Goal:** Streamlined frontend, robust backend on constrained infrastructure, cost-optimized LLM routing.

- [x] Replaced PyTorch sentence-transformers with fastembed (ONNX): 100 MB RAM vs. 380 MB
- [x] FastAPI REST API deployed on Render.com (free plan, 512 MB RAM, fits with fastembed)
- [x] Streamlit investigator UI deployed on Streamlit Community Cloud (free)
- [x] 4-tab investigator workflow: Investigate Claim / Graph Query / Record Feedback / Stats & History
- [x] Support for both Claude (best quality, paid) and Gemma (free, with Google API key)
- [x] System prompts optimized for both LLM models (explicit structure, no ambiguity)
- [x] Increased API timeouts for slow endpoints (120s for GraphRAG + NL query)
- [x] All 12 architecture diagrams created in `diagrams/` (Mermaid markdown)

---

### Phase 4 — Network Effect (Not Started)

**Goal:** Real-time pre-payment interception and cross-carrier intelligence exchange.

- [ ] Real-time streaming ingest (Kinesis → Kafka → Neo4j) live for pre-payment interception
- [ ] Payment hold automation for Mandatory Override rings
- [ ] Cross-carrier data exchange integration (anonymized entity fingerprints)
- [ ] RL-GNN model upgrade for adaptive ring detection
- [ ] Full production validation and load testing
- [ ] Consortium ring alert participation active

**Exit criteria:** Human Agreement Rate >= 80%. FPR < 20%. Pre-payment interception rate tracked and improving. Cross-carrier alerts flowing bidirectionally.

---

## Appendix A — Fraud Signal Reference

### Node-Level Fraud Flags

```text
Customer:
  synthetic_identity_flag      — fabricated identity across multiple policies
  role_switching_flag          — same person appears as driver/witness across claims
  shared_phone_flag            — phone shared with unrelated policyholders
  shared_bank_flag             — payment account shared with unrelated policyholders
  ip_flagged                   — login IP matches known fraud cluster

Claim:
  staged_accident_flag         — pattern consistent with deliberate collision staging
  claim_padding_flag           — inflated damages beyond actual incident
  closed_loop_flag             — same entities recycled across claim chain
  ring_member_flag             — assigned to a fraud ring cluster
  manual_override_flag         — requires mandatory human review before any action
  cross_role_participation_flag — entity appears in multiple roles across ring claims

Witness:
  professional_witness_flag    — same witness in 3+ unrelated claims
  coached_statement_flag       — statement language matches ring pattern
  same_phone_claims_count      — phone reused across witness pool

MedicalReport:
  upcoding_flag                — billed at higher severity than documented
  unbundling_flag              — services split to maximize billing
  phantom_billing_flag         — services billed but not rendered

FinancialTransaction:
  velocity_flag                — payment issued <3 days from claim open
  round_amount_flag            — suspiciously round payment amount
  is_shared_account_flag       — payment account shared across ring members
```

### Edge-Level Risk Scores

```text
SHARES_ATTRIBUTE fraud_signal values:
  CRITICAL  — shared bank account (attribute_risk_score >= 0.95)
  HIGH      — shared IP address   (attribute_risk_score >= 0.80)
  MEDIUM    — shared phone        (attribute_risk_score >= 0.60)
  LOW       — shared zip code     (attribute_risk_score >= 0.30)
```

---

## Appendix B — Cypher Query Patterns

### Find all members of a fraud ring

```cypher
MATCH (r:FraudRing {ring_id: 'RING-007'})
MATCH (r)-[:RING_INVOLVES_CUSTOMER]->(c:Customer)
MATCH (r)-[:RING_CONTAINS_CLAIM]->(cl:Claim)
RETURN r, c, cl
```

### Detect closed-loop patterns (Customer → Lawyer → Claim → Shop → Customer)

```cypher
MATCH p = (c:Customer)-[:FILED_CLAIM]->(cl:Claim)
  -[:REPRESENTED_BY]->(l:Lawyer)
  -[:CO_LOCATED_WITH]->(s:RepairShop)
  <-[:REPAIRED_AT]-(cl2:Claim)
  <-[:FILED_CLAIM]-(c)
WHERE c.fraud_flag = 'Y'
RETURN p LIMIT 25
```

### Find witnesses appearing in 3+ claims

```cypher
MATCH (w:Witness)<-[:HAS_WITNESS]-(cl:Claim)
WITH w, count(cl) AS claim_count
WHERE claim_count >= 3
RETURN w.full_name, claim_count, w.professional_witness_flag
ORDER BY claim_count DESC
```

### Shared bank account ring detection

```cypher
MATCH (c1:Customer)-[e:SHARES_ATTRIBUTE]->(c2:Customer)
WHERE e.attribute_type = 'bank_account'
  AND e.fraud_signal = 'CRITICAL'
MATCH (c1)-[:FILED_CLAIM]->(cl1:Claim)
MATCH (c2)-[:FILED_CLAIM]->(cl2:Claim)
RETURN c1.full_name, c2.full_name, e.shared_value_hash,
       cl1.claim_id, cl2.claim_id
```

### Family fraud ring detection

```cypher
MATCH (c1:Customer)-[r:CUSTOMER_RELATIONSHIPS]->(c2:Customer)
WHERE r.relationship_type = 'FRAUD_RING_FAMILY_LINK'
  AND c1.fraud_flag = 'Y'
  AND c2.fraud_flag = 'Y'
RETURN c1.full_name, c2.full_name, r.shared_last_name,
       r.confidence_score
ORDER BY r.confidence_score DESC
```

---

*PRD v3.0 · Fraud Ring Detection: Graph-RAG System · Auto Insurance*
*Last updated: April 2026 · Confidential · Internal Use Only*
