# CLAUDE.md — Fraud Ring Detection: Graph-RAG System

## Project Overview

Production-grade **auto insurance fraud ring detection** system combining a Neo4j knowledge graph,
Graph Neural Networks (GraphSAGE + HINormer), GraphRAG (LangGraph + Claude via OpenRouter), a
Streamlit investigator UI, and a Human-in-the-Loop (HITL) feedback loop.

**Rollout status:**

| Phase | Scope | Status |
| ----- | ----- | ------ |
| Phase 1 — Graph Load | Neo4j graph + rule engine + HITL queue | Complete |
| Phase 2 — GNN Scoring | GraphSAGE + HINormer + XGB/RF/LGBM ensemble | Complete |
| Phase 3 — GraphRAG | LangGraph pipeline + NL query + feedback loop | Complete |
| Phase 3b — Frontend | Streamlit UI + FastAPI deployment on Render | Complete |
| Phase 4 — Streaming | Real-time Kinesis ingest + cross-carrier exchange | Not started |

Full product spec: [PRD.md](PRD.md)

---

## Repository Structure

```
FraudRingDetectionGraph-RAGSystem/
│
├── src/                                  — all backend application source code
│   ├── main.py                           — unified CLI entry point
│   ├── agent/
│   │   └── pipeline.py                   — LangGraph StateGraph: retrieve → embed → reason
│   ├── services/
│   │   ├── graph_retriever.py            — Neo4j subgraph retrieval + HumanReview write-back
│   │   ├── vector_store.py               — LocalVectorStore (numpy) + PineconeVectorStore
│   │   ├── feedback.py                   — investigator feedback collection + retraining trigger
│   │   └── gnn/
│   │       ├── config.py                 — node types, edge types, feature columns, hyperparams
│   │       ├── data_loader.py            — CSVs → PyG HeteroData + train/val/test splits
│   │       ├── model.py                  — FraudGNN: SAGEConv + HGTConv + classifier head
│   │       ├── train.py                  — training loop, SMOTE/ADASYN, ensemble training
│   │       ├── explainer.py              — gradient saliency → reasoning trace strings
│   │       └── scorer.py                 — batch inference + Neo4j score write-back
│   ├── tools/
│   │   ├── load_graph.py                 — Neo4j bulk loader (Phase 1)
│   │   └── nl_query.py                   — NL → Cypher → results → NL summary (Claude)
│   └── utils/
│       ├── config.py                     — API keys, model names, prompts, embedding config
│       ├── embedder.py                   — fastembed (ONNX) embeddings, no PyTorch
│       └── feature_utils.py              — numeric/binary/ordinal feature extraction for GNN
│
├── ui/                                   — Streamlit frontend (deploy to Streamlit Cloud)
│   ├── streamlit_app.py                  — main app: Investigate / Query / Feedback / Stats tabs
│   ├── requirements.txt                  — frontend deps: streamlit, requests, pandas
│   └── .streamlit/
│       ├── config.toml                   — dark theme configuration
│       └── secrets.toml.example          — api_url template (copy → secrets.toml, gitignored)
│
├── data/                                 — CSV node and edge files (14,292 nodes · 28,690 edges)
│   ├── nodes_*.csv                       — 24 node type files
│   ├── edges_*.csv                       — fixed-type relationship files
│   └── rel_*.csv                         — family relationship files
│
├── models/                               — saved artefacts (git-ignored)
│   ├── fraud_gnn.pt                      — trained GNN weights
│   ├── ensemble.pkl                      — XGB + RF + LGBM ensemble
│   ├── scaler.pkl                        — feature scaler
│   ├── feedback_labels.json              — investigator label overrides
│   └── feedback_labels_meta.json         — feedback loop metadata
│
├── api.py                                — FastAPI web service (deploy to Render.com)
├── build.sh                              — Render build script (CPU torch + requirements.txt)
├── render.yaml                           — Render service definition (free plan + Pinecone)
├── requirements.txt                      — API production deps (Phase 3 runtime, Pinecone)
├── requirements_phase2.txt               — local-only: PyTorch Geometric, XGBoost, LightGBM
├── README.md                             — setup, local dev, and deployment guide
├── PRD.md                                — full product requirements document
└── .env                                  — credentials (git-ignored, never commit)
```

---

## Environment Setup

### Prerequisites

- Python 3.11+
- Neo4j Aura instance
- OpenRouter API key
- Pinecone account (free tier)
- Virtual environment at `venv/`

### `.env` — all variables

```env
# ── Neo4j ─────────────────────────────────────────────────────
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_TRUST_ALL_CERTS=true        # required on Windows — Aura intermediate CA

# ── Graph loader ───────────────────────────────────────────────
DATA_DIR=./data
BATCH_SIZE=200

# ── OpenRouter (LLM) ───────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4-5   # or google/gemma-4-31b-it:free
GOOGLE_API_KEY=...                             # optional, needed for Gemma

# ── Embeddings ──────────────────────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5         # fastembed model (384-dim, ONNX)

# ── Vector store ───────────────────────────────────────────────
VECTOR_STORE_BACKEND=pinecone     # "local" or "pinecone"
PINECONE_API_KEY=...
PINECONE_INDEX=fraud-rings
```

> `.env` is gitignored. Never commit credentials.

### `ui/.streamlit/secrets.toml` — frontend config

```toml
api_url = "https://your-service.onrender.com"
```

> Copy from `ui/.streamlit/secrets.toml.example`. Also gitignored.

### Installation

```bash
# Phase 3 API runtime (FastAPI, LangGraph, fastembed, Neo4j, Pinecone)
# fastembed uses ONNX Runtime instead of PyTorch — ~100 MB RAM vs ~380 MB
pip install -r requirements.txt

# Streamlit frontend (lightweight — streamlit, requests, pandas only)
pip install -r ui/requirements.txt

# Phase 2 GNN training (local only — not needed on Render)
# PyTorch is NOT installed on Render; train locally then upload models
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements_phase2.txt
```

---

## Running the System

### Phase 1 — Load graph into Neo4j

```bash
python src/main.py load-graph --dry-run     # validate CSVs only, no DB writes
python src/main.py load-graph               # load using .env credentials
python src/main.py load-graph --batch 500   # override batch size
```

Verify in Neo4j Aura Query tab:

```cypher
MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC
MATCH ()-[r]->() RETURN type(r), count(r) ORDER BY count(r) DESC
```

### Phase 2 — GNN Scoring (local only — not on Render)

```bash
python src/main.py gnn train                              # train + save to models/
python src/main.py gnn score                              # score all claims + write to Neo4j
python src/main.py gnn score --dry-run                    # score without Neo4j writes
python src/main.py gnn explain --top-n 20 --output traces.json
python src/main.py gnn evaluate                           # re-evaluate saved model
```

### Phase 3 — GraphRAG Pipeline

```bash
# Run once after Phase 1/2 — embeds FraudRing nodes and pushes to Pinecone
python src/main.py rag index

# Full GraphRAG investigation brief for a claim
python src/main.py rag explain CLM-521585
python src/main.py rag explain CLM-521585 --verbose       # also prints raw subgraph

# Natural language queries against the fraud graph
python src/main.py rag query                              # interactive REPL
python src/main.py rag query --question "Which fraud rings have a lawyer in 3+ claims?"

# Record investigator decisions (writes HumanReview node to Neo4j)
python src/main.py rag feedback CLM-521585 --decision Approve   --investigator INV-001
python src/main.py rag feedback CLM-521585 --decision Dismiss   --investigator INV-001 --feedback FP
python src/main.py rag feedback CLM-521585 --decision Escalate  --investigator INV-001 --feedback FN

# Feedback-loop retraining
python src/main.py rag retrain                            # retrain if >= 20 new reviews
python src/main.py rag retrain --evaluate-only            # F1 metrics, no retraining
python src/main.py rag retrain --min-reviews 10           # lower threshold for testing

# Stats
python src/main.py rag stats
```

### Local development — API + UI together

```bash
# Terminal 1: start FastAPI backend
uvicorn api:app --reload --port 8000
# Docs at http://localhost:8000/docs

# Terminal 2: start Streamlit frontend
streamlit run ui/streamlit_app.py
# Opens at http://localhost:8501
# Set Backend URL in sidebar to http://localhost:8000
```

---

## Deployment

### API → Render.com

| File | Purpose |
| ---- | ------- |
| [api.py](api.py) | FastAPI web service wrapping the Phase 3 pipeline |
| [build.sh](build.sh) | Installs requirements.txt (no PyTorch — uses fastembed ONNX) |
| [render.yaml](render.yaml) | Service definition — free plan, fastembed + Pinecone |
| [requirements.txt](requirements.txt) | Production dependencies (fastembed>=0.3.0, no torch) |

**Steps:**

1. Push repo to GitHub
2. New **Web Service** on Render → connect repo → Render picks up `render.yaml`
3. Set secret env vars in Render dashboard:
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
   - `OPENROUTER_API_KEY`
   - `PINECONE_API_KEY`

### UI → Streamlit Community Cloud

| File | Purpose |
| ---- | ------- |
| [ui/streamlit_app.py](ui/streamlit_app.py) | Main Streamlit app (4 tabs) |
| [ui/requirements.txt](ui/requirements.txt) | streamlit + requests + pandas |
| [ui/.streamlit/config.toml](ui/.streamlit/config.toml) | Dark theme |
| [ui/.streamlit/secrets.toml.example](ui/.streamlit/secrets.toml.example) | api_url template |

**Steps:**

1. Go to share.streamlit.io → New app
2. Main file path: `ui/streamlit_app.py`
3. Requirements file: `ui/requirements.txt`
4. In Advanced settings → Secrets add: `api_url = "https://your-service.onrender.com"`

### Pinecone index (one-time, run locally before deploying)

1. Create index in Pinecone console: name=`fraud-rings`, dimensions=`384`, metric=`cosine`
2. Set `PINECONE_API_KEY` and `VECTOR_STORE_BACKEND=pinecone` in `.env`
3. Run: `python src/main.py rag index`

The index persists in Pinecone cloud — survives all Render deploys and cold starts.

### Free plan RAM budget (Render)

| Component | RAM |
| --------- | --- |
| onnxruntime (fastembed) | ~60 MB |
| `BAAI/bge-small-en-v1.5` weights | ~30 MB |
| FastAPI + LangGraph + Neo4j driver | ~80 MB |
| Pinecone client + other libs | ~30 MB |
| **Total** | **~200 MB — fits 512 MB free limit** |

> Previously used PyTorch + sentence-transformers (~380 MB, caused OOM). Switched to fastembed (ONNX)
> for 3x RAM savings. Both models output 384-dim — Pinecone index compatible.

### Phase 2 GNN on Render

`torch-geometric` is **not** installed on Render. `trigger_retrain()` lazy-imports it and will
fail clearly with `ImportError`. Run retraining locally and redeploy updated model weights.

---

## Graph Data Model

**Scale:** 14,292 nodes · 28,690 edges · 941 properties · 24 node types · 28 edge types

All CSVs use Neo4j bulk import format: `:ID`, `:LABEL`, `:START_ID`, `:END_ID`, `:TYPE` plus
property columns. All nodes use `_neo4j_id` (string) as the internal merge key.

### Key node types

| Label | Count | Primary ID | Key Fraud Signals |
| ----- | ----- | ---------- | ----------------- |
| `Customer` | 1,000 | `cust_id` | `synthetic_identity_flag`, `role_switching_flag`, `shared_bank_flag` |
| `Claim` | 1,000 | `claim_id` | `ring_member_flag`, `staged_accident_flag`, `manual_override_flag` |
| `FraudRing` | 20 | `ring_id` | `ring_score`, `status`, `closed_loop_detected` |
| `Witness` | 1,487 | `statement_id` | `professional_witness_flag`, `coached_statement_flag` |
| `Lawyer` | 7 | `lawyer_id` | `closed_loop_network_flag`, `known_to_siu`, `prior_fraud_involvement_flag` |
| `RepairShop` | 7 | `shop_id` | `fraud_flag`, `inflated_estimate_rate_pct`, `siu_referral_count` |
| `InvestigationCase` | 876 | `case_id` | `ai_fraud_score`, `manual_override_triggered` |
| `HumanReview` | 570 | `review_id` | `decision`, `override_ai_recommendation`, `feedback_to_model` |
| `NetworkFeature` | 2,000 | `feature_id` | `pagerank_score`, `ring_suspicion_score`, `hop_2_fraud_count` |

### Key relationship types

| Relationship | Pattern |
| ------------ | ------- |
| `SHARES_ATTRIBUTE` | Customer ↔ Customer — shared phone/bank/IP (CRITICAL/HIGH/MEDIUM/LOW) |
| `FILED_CLAIM` | Customer → Claim |
| `RING_CONTAINS_CLAIM` | FraudRing → Claim |
| `RING_INVOLVES_CUSTOMER` | FraudRing → Customer |
| `REPRESENTED_BY` | Claim → Lawyer |
| `REPAIRED_AT` | Claim → RepairShop |
| `HAS_WITNESS` | Claim → Witness |
| `HAS_MEDICAL_REPORT` | Claim → MedicalReport |
| `INVESTIGATES_CLAIM` | InvestigationCase → Claim |
| `REVIEWS_CASE` | HumanReview → InvestigationCase |

### Common Cypher patterns

```cypher
-- Fraud ring members
MATCH (r:FraudRing {ring_id: 'RING-007'})
MATCH (r)-[:RING_INVOLVES_CUSTOMER]->(c:Customer)
MATCH (r)-[:RING_CONTAINS_CLAIM]->(cl:Claim)
RETURN r, c, cl

-- Professional witnesses (3+ claims)
MATCH (w:Witness)<-[:HAS_WITNESS]-(cl:Claim)
WITH w, count(cl) AS claim_count WHERE claim_count >= 3
RETURN w.full_name, claim_count, w.professional_witness_flag ORDER BY claim_count DESC

-- Shared bank account detection
MATCH (c1:Customer)-[e:SHARES_ATTRIBUTE]->(c2:Customer)
WHERE e.attribute_type = 'bank_account' AND e.fraud_signal = 'CRITICAL'
MATCH (c1)-[:FILED_CLAIM]->(cl1:Claim), (c2)-[:FILED_CLAIM]->(cl2:Claim)
RETURN c1.full_name, c2.full_name, cl1.claim_id, cl2.claim_id

-- High-priority claims
MATCH (c:Claim) WHERE c.final_suspicion_score >= 0.70
RETURN c.claim_id, c.final_suspicion_score, c.adjuster_priority_tier
ORDER BY c.final_suspicion_score DESC
```

---

## Detection Architecture

| Layer | Speed | Technology | Trigger |
| ----- | ----- | ---------- | ------- |
| **Layer 1** Rule engine | <100 ms | Deterministic Cypher + heuristics | Every claim at ingestion |
| **Layer 2** GNN scoring | 42 ms/batch | PyTorch Geometric · GraphSAGE + HINormer | After Layer 1 signals |
| **Layer 3** GraphRAG reasoning | On-demand | LangGraph · Claude via OpenRouter · Pinecone | When HITL review triggered |

### Technology stack

| Component | Technology |
| --------- | ---------- |
| Graph DB | Neo4j Aura (GDB + GDS) |
| GNN | PyTorch Geometric · GraphSAGE (SAGEConv) · HINormer (HGTConv) |
| Explainability | Gradient saliency (d_fraud_score / d_features) |
| Embeddings | sentence-transformers · paraphrase-MiniLM-L3-v2 (384-dim, 17 MB) |
| Vector KB | Pinecone (production) / LocalVectorStore numpy (dev) |
| Agentic pipeline | LangGraph StateGraph |
| LLM reasoning | Claude via OpenRouter (openai-compatible API) |
| Web API | FastAPI + uvicorn |
| Frontend | Streamlit (4 tabs: Investigate / Query / Feedback / Stats) |
| Oversampling | SMOTE + ADASYN (training time only — never inference) |
| API hosting | Render.com (free plan) |
| UI hosting | Streamlit Community Cloud (free) |
| Streaming ingest | Amazon Kinesis + Neo4j Kafka Connector (Phase 4) |
| Batch ETL | AWS Glue + EMR (Phase 4) |
| ML serving | Amazon SageMaker (Phase 4) |

---

## GNN Scoring (Layer 2)

### Model architecture

```text
Input features (per node type, from CSV columns)
    |
    v  type-specific Linear → hidden_channels (128)
    |
    v  SAGEConv — inductive, handles new nodes at inference
       HeteroConv wrapper: one SAGEConv per edge type
    |
    v  BatchNorm + ReLU + Dropout(0.3)
    |
    v  HGTConv — heterogeneous transformer attention (heads=4)
       type-specific attention weights per (src_type, edge_type, dst_type)
    |
    v  BatchNorm + ReLU + Dropout(0.3)
    |
    v  Classifier head: Linear(128→64) → ReLU → Linear(64→1) → sigmoid
       Applied to Claim node embeddings only
```

### Class imbalance strategy

| Stage | Technique | When |
| ----- | --------- | ---- |
| GNN training | BCEWithLogitsLoss `pos_weight = #neg / #pos` | Training only |
| Embedding augmentation | SMOTE + ADASYN on Claim embeddings | Training only |
| Ensemble | XGBoost + RandomForest + LightGBM on augmented embeddings | Training + inference |

> **Critical rule:** `apply_smote_adasyn()` must never be called on val/test/production data.

### Neo4j properties written by scorer

| Property | Node | Description |
| -------- | ---- | ----------- |
| `gnn_suspicion_score` | Claim | Raw GNN sigmoid probability |
| `ensemble_suspicion_score` | Claim | Mean of XGB + RF + LGBM probabilities |
| `final_suspicion_score` | Claim | Blended score (0.5 x GNN + 0.5 x ensemble) |
| `adjuster_priority_tier` | Claim | Critical / High Priority / Standard |
| `ai_fraud_score` | InvestigationCase | `final_suspicion_score` of linked claim |

### Score tiers

| Score | Tier | Action |
| ----- | ---- | ------ |
| >= 0.90 | Critical | Mandatory Override (OVR-001) — payment HOLD |
| >= 0.70 | High Priority | Priority queue — 4 hr SLA |
| < 0.70 | Standard | Standard queue — 30 min SLA |

---

## GraphRAG Pipeline (Layer 3)

### Pipeline flow

```text
Claim ID
    |
    v  retrieve_subgraph  (src/services/graph_retriever.py)
       Neo4j Cypher: claim + customer + lawyer + shop + witnesses
                   + ring + investigation_case + network_feature
                   + shared_attributes
    |
    v  find_analogous_rings  (src/services/vector_store.py)
       Embed subgraph text (384-dim) → cosine search Pinecone
       top-K = 3 historical rings
    |
    v  generate_reasoning  (src/agent/pipeline.py)
       Claude via OpenRouter (anthropic/claude-sonnet-4-5)
       REASONING_SYSTEM_PROMPT + subgraph + analogous rings
       + OVR override trigger evaluation
       → Investigation Brief
```

### LangGraph state keys

| Key | Type | Description |
| --- | ---- | ----------- |
| `claim_id` | str | Input claim identifier |
| `fraud_score` | float | `final_suspicion_score` from Claim node |
| `subgraph` | dict | Raw Neo4j context (all neighbours) |
| `subgraph_text` | str | Human-readable subgraph for the LLM |
| `analogous_rings` | list | `[{id, score, metadata}, ...]` top-K similar rings |
| `override_triggers` | list[str] | OVR codes that fired |
| `reasoning_trace` | str | Claude investigation brief |
| `error` | str or None | Populated if any node fails |

### Mandatory Override triggers (payment HOLD)

| Code | Condition |
| ---- | --------- |
| OVR-001 | Ring suspicion score >= 0.90 |
| OVR-002 | Projected ring exposure > $75K |
| OVR-003 | Same attorney/witness/shop across jurisdictions |
| OVR-004 | Previously dismissed ring entity reappears |
| OVR-005 | Licensed attorney is primary connecting edge |
| OVR-006 | Vulnerable claimant (elderly >= 65, language barrier) |
| OVR-007 | Public figure / prior media coverage |
| OVR-008 | Any node has a prior SIU referral |

Claim-level auto-flag logic:

```python
manual_override_flag = any([
    total_claim_amount > 50_000,
    rag_confidence_score < 0.6 and fraud_reported == 'Y',
    hospitalization_required == 'Yes' and police_report_available in ('NO', '?'),
    claim_cluster_id != '',
])
```

---

## NL Query Engine

Investigators ask questions in plain English. Claude (via OpenRouter) converts the question to
Cypher, executes it read-only, then summarises the results.

```text
Question  →  NL_QUERY_SYSTEM_PROMPT + few-shot examples  →  Cypher
Cypher    →  FORBIDDEN_KEYWORDS validation  →  Neo4j execute  →  rows
rows      →  RESULT_FORMATTER_SYSTEM_PROMPT  →  NL answer
```

**Cypher safety** — any generated query containing write operations is rejected before execution:

```python
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(MERGE|CREATE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+apoc\.periodic)\b",
    re.IGNORECASE,
)
```

---

## HITL Feedback Loop

### Decision → label mapping

| Decision | Label | Meaning |
| -------- | ----- | ------- |
| Approve | 1 | Confirms fraud |
| Escalate | 1 | Confirmed serious fraud |
| Dismiss | 0 | False positive |

`feedback_to_model` signal: `Correct` / `FP` / `FN` / `Uncertain`
(`Uncertain` is excluded from retraining labels)

### Retraining flow

```text
Investigator decisions (Approve / Dismiss / Escalate)
    ↓  stored in Neo4j HumanReview nodes
FeedbackStore.collect()
    ↓  pulls rows since last training run, maps to binary labels
compute_f1_delta()
    ↓  compares model predictions vs investigator ground truth
trigger_retrain()  (requires >= 20 labelled reviews)
    ↓  saves feedback_labels.json
    ↓  calls Phase 2 run_training() with feedback-weighted labels
    ↓  computes F1 delta — promotes new model only if F1 improves
```

### LLM call sites

All LLM calls use the `openai` Python package pointed at OpenRouter:

| File | System prompt | Purpose |
| ---- | ------------- | ------- |
| `src/agent/pipeline.py` | `REASONING_SYSTEM_PROMPT` | Investigation brief generation |
| `src/tools/nl_query.py` | `NL_QUERY_SYSTEM_PROMPT` | NL → Cypher conversion |
| `src/tools/nl_query.py` | `RESULT_FORMATTER_SYSTEM_PROMPT` | Result summarisation |

Model: configurable via `OPENROUTER_MODEL` env var.
- Default: `anthropic/claude-sonnet-4-5` (paid, ~$3–15/M tokens, best quality)
- Alternative: `google/gemma-4-31b-it:free` (free, requires Google API key in OpenRouter integrations)

System prompts are optimized for both models (explicit structure, no ambiguity).

### Vector store backends

| Backend | Config | Use case |
| ------- | ------ | -------- |
| `pinecone` (default) | `PINECONE_API_KEY` + `PINECONE_INDEX` | Production — persists across deploys |
| `local` | No extra config | Development / offline / <= 100K rings |

---

## Streamlit Frontend

Four-tab investigator UI in `ui/streamlit_app.py`:

| Tab | Function |
| --- | -------- |
| Investigate Claim | Enter claim ID → fraud score banner + override triggers + analogous rings + Claude brief |
| Graph Query | Natural language question → Cypher → Neo4j → NL answer |
| Record Feedback | Form to submit Approve / Dismiss / Escalate with confidence and override reason |
| Stats & History | Total reviews, vector store size, F1 history table |

The app reads `api_url` from `ui/.streamlit/secrets.toml` (Streamlit Cloud) or the sidebar input (local dev). All heavy computation runs in the FastAPI backend — the UI is request-only.

---

## Success Targets

| Metric | Launch | Month 6 |
| ------ | ------ | ------- |
| Human Agreement Rate | > 65% | > 80% |
| False Positive Rate | < 35% | < 20% |
| Model AUC-ROC | >= 0.91 | >= 0.95 |
| Mean Time to Decision (Standard) | < 30 min | < 20 min |
| Mean Time to Decision (Override) | < 4 hours | < 2 hours |

---

## Compliance Rules

- **No automated adverse action** — system outputs are investigative leads only, never determinations
- **PII never stored raw** — only hashed representations (SSN, bank account, phone, VIN)
- **Immutable audit log** — every AI output, investigator decision, and feedback signal logged with timestamp + user ID
- **Quarterly bias audits** — model output distributions reviewed by compliance officer before each version promotion
- **SMOTE/ADASYN training only** — synthetic oversampling must never be applied at inference time

---

## Known Fixes Applied

| Issue | File | Fix |
| ----- | ---- | --- |
| `OPENROUTER_API_KEY` read as `None` despite being set in `.env` | `src/utils/config.py` | Added `load_dotenv()` before `os.getenv()` calls |
| `FutureWarning: get_sentence_embedding_dimension` renamed in sentence-transformers >= 3.x | `src/utils/embedder.py` | `getattr` probe for `get_embedding_dimension()` with fallback |
| `ReduceLROnPlateau` `verbose` kwarg removed in PyTorch >= 2.2 | `src/services/gnn/train.py` | Removed `verbose=False` from scheduler init |
| LightGBM `UserWarning: X does not have valid feature names` at inference | `src/services/gnn/scorer.py`, `train.py` | Cast to plain numpy + `warnings.catch_warnings()` filter |
| `argparse.ArgumentError: conflicting subparser: score` in main.py | `src/main.py` | Removed duplicate `score` subparser registration |
| `load_graph.py` `parser.parse_args()` ran at import time | `src/tools/load_graph.py` | Moved all arg parsing inside `main()` via `_build_parser()` |
| `pinecone-client` package renamed to `pinecone` | `requirements.txt` | Changed to `pinecone>=5.0.0`; uninstalled old package |
| Emoji in print statements caused `UnicodeEncodeError` on Windows cp1252 terminal | `src/main.py`, `src/tools/load_graph.py` | Replaced all emoji with ASCII equivalents |
