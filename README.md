# Fraud Ring Detection — Graph-RAG System

Production-grade auto insurance fraud ring detection combining a **Neo4j knowledge graph**, **Graph Neural Networks** (GraphSAGE + HINormer), **GraphRAG** (LangGraph + Claude via OpenRouter), and a **Human-in-the-Loop** investigator workflow.

---

## Architecture

```text
Claim Intake
     │
     ▼
Layer 1 — Rule Engine          < 100 ms   Deterministic Cypher + heuristics
     │
     ▼
Layer 2 — GNN Scoring           42 ms     GraphSAGE + HINormer + XGB/RF/LGBM ensemble
     │
     ▼
Layer 3 — GraphRAG Reasoning   On-demand  LangGraph + Claude + Pinecone vector search
     │
     ▼
Layer 4 — HITL Review                     Investigator decisions feed back into retraining
```

| Phase | Scope | Status |
| ------- | ------- | -------- |
| Phase 1 — Graph Load | Neo4j graph + rule engine + HITL queue | Complete |
| Phase 2 — GNN Scoring | GraphSAGE + HINormer + XGB/RF/LGBM ensemble | Complete |
| Phase 3 — GraphRAG | LangGraph pipeline + NL query + feedback loop | Complete |
| Phase 4 — Streaming | Real-time Kinesis ingest + cross-carrier exchange | Not started |

**Graph scale:** 14,292 nodes · 28,690 edges · 941 properties · 24 node types · 28 edge types

---

## Repository Structure

```
FraudRingDetectionGraph-RAGSystem/
├── src/
│   ├── main.py                        Unified CLI entry point
│   ├── agent/
│   │   └── pipeline.py                LangGraph StateGraph (retrieve -> embed -> reason)
│   ├── services/
│   │   ├── graph_retriever.py         Neo4j subgraph retrieval + HumanReview write-back
│   │   ├── vector_store.py            LocalVectorStore (numpy) + PineconeVectorStore
│   │   ├── feedback.py                Investigator feedback + retraining trigger
│   │   └── gnn/
│   │       ├── config.py              Node types, edge types, feature columns, hyperparams
│   │       ├── data_loader.py         CSVs -> PyG HeteroData + train/val/test splits
│   │       ├── model.py               FraudGNN: SAGEConv + HGTConv + classifier head
│   │       ├── train.py               Training loop, SMOTE/ADASYN, ensemble training
│   │       ├── explainer.py           Gradient saliency -> reasoning trace strings
│   │       └── scorer.py              Batch inference + Neo4j score write-back
│   ├── tools/
│   │   ├── load_graph.py              Neo4j bulk loader (Phase 1)
│   │   └── nl_query.py                NL -> Cypher -> results -> NL summary
│   └── utils/
│       ├── config.py                  API keys, model names, prompts, embedding config
│       ├── embedder.py                Sentence-transformer ring/claim embeddings
│       └── feature_utils.py           Numeric/binary/ordinal feature extraction for GNN
├── data/                              CSV node and edge files
│   ├── nodes_*.csv                    24 node type files
│   └── edges_*.csv / rel_*.csv        Edge and relationship files
├── models/                            Saved artefacts (git-ignored)
│   ├── fraud_gnn.pt                   Trained GNN weights
│   ├── ensemble.pkl                   XGB + RF + LGBM ensemble
│   └── scaler.pkl                     Feature scaler
├── api.py                             FastAPI web service (Render.com)
├── build.sh                           Render build script (CPU torch + requirements.txt)
├── render.yaml                        Render service definition
├── requirements.txt                   Production dependencies
└── requirements_phase2.txt            Local-only: PyTorch Geometric, XGBoost, LightGBM
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Neo4j Aura instance ([console.neo4j.io](https://console.neo4j.io))
- OpenRouter API key ([openrouter.ai](https://openrouter.ai))
- Pinecone account ([pinecone.io](https://pinecone.io)) — free tier works

### 1. Clone and install

```bash
git clone <repo-url>
cd FraudRingDetectionGraph-RAGSystem
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Phase 3 runtime (API, GraphRAG, embeddings)
pip install -r requirements.txt

# Phase 2 GNN training (local only — not needed on Render)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements_phase2.txt
```

### 2. Configure `.env`

Create a `.env` file in the project root:

```env
# Neo4j Aura
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>
NEO4J_TRUST_ALL_CERTS=true

# OpenRouter (LLM)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4-5

# Pinecone vector store
VECTOR_STORE_BACKEND=pinecone
PINECONE_API_KEY=<your-pinecone-key>
PINECONE_INDEX=fraud-rings

# Data
DATA_DIR=./data
BATCH_SIZE=200
```

### 3. Create Pinecone index

In the Pinecone console, create an index with:
- **Name:** `fraud-rings`
- **Dimensions:** `384`
- **Metric:** `Cosine`

### 4. Load the graph (Phase 1)

```bash
# Validate CSV files (no DB writes)
python src/main.py load-graph --dry-run

# Load all nodes and edges into Neo4j
python src/main.py load-graph
```

### 5. Train the GNN (Phase 2 — local only)

```bash
python src/main.py gnn train
python src/main.py gnn score          # write scores to Neo4j
python src/main.py gnn explain --top-n 20 --output traces.json
```

### 6. Build the vector knowledge base (Phase 3)

```bash
# Embed all FraudRing nodes and push to Pinecone (run once)
python src/main.py rag index
```

### 7. Run investigations

```bash
# Full GraphRAG investigation brief for a claim
python src/main.py rag explain CLM-521585
python src/main.py rag explain CLM-521585 --verbose

# Natural language queries
python src/main.py rag query --question "Which fraud rings have a lawyer in 3+ claims?"
python src/main.py rag query    # interactive REPL

# Record investigator decisions
python src/main.py rag feedback CLM-521585 --decision Approve --investigator INV-001
python src/main.py rag feedback CLM-521585 --decision Dismiss --investigator INV-001 --feedback FP

# Feedback-loop retraining
python src/main.py rag retrain --evaluate-only
python src/main.py rag retrain --min-reviews 20

# Stats
python src/main.py rag stats
```

---

## REST API

Start locally:

```bash
uvicorn api:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

| Method | Endpoint | Description |
| -------- | ---------- | ------------- |
| `GET` | `/health` | Liveness probe |
| `POST` | `/explain/{claim_id}` | Full GraphRAG investigation brief |
| `POST` | `/query` | Natural language graph query |
| `POST` | `/feedback/{claim_id}` | Record investigator decision |
| `GET` | `/stats` | Feedback history + vector store size |

---

## GNN Model

### Architecture

```
Input features (per node type)
    |
    v  type-specific Linear -> 128 hidden channels
    |
    v  SAGEConv (inductive, handles new nodes at inference)
       HeteroConv wrapper: one SAGEConv per edge type
    |
    v  BatchNorm + ReLU + Dropout(0.3)
    |
    v  HGTConv (heterogeneous transformer attention, heads=4)
    |
    v  BatchNorm + ReLU + Dropout(0.3)
    |
    v  Classifier: Linear(128->64) -> ReLU -> Linear(64->1) -> sigmoid
       Applied to Claim nodes only
```

### Score tiers written to Neo4j

| Score | Tier | Action |
| ------- | ------ | -------- |
| >= 0.90 | Critical | Mandatory Override — payment HOLD |
| >= 0.70 | High Priority | Priority queue — 4 hr SLA |
| < 0.70 | Standard | Standard queue — 30 min SLA |

### Properties written to Neo4j by the scorer

| Property | Node | Description |
| ---------- | ------ | ------------- |
| `gnn_suspicion_score` | Claim | Raw GNN sigmoid probability |
| `ensemble_suspicion_score` | Claim | Mean of XGB + RF + LGBM |
| `final_suspicion_score` | Claim | 0.5 x GNN + 0.5 x ensemble |
| `adjuster_priority_tier` | Claim | Critical / High Priority / Standard |
| `ai_fraud_score` | InvestigationCase | final_suspicion_score of linked claim |

---

## GraphRAG Pipeline

```
Claim ID
    |
    v  retrieve_subgraph
       Neo4j Cypher: claim + customer + lawyer + shop + witnesses + ring + investigation
    |
    v  find_analogous_rings
       Embed subgraph text (384-dim) -> cosine search Pinecone -> top-3 historical rings
    |
    v  generate_reasoning
       Claude (via OpenRouter) + subgraph + analogous rings -> Investigation Brief
```

### Mandatory Override triggers

| Code | Condition |
| ------ | ----------- |
| OVR-001 | Ring suspicion score >= 0.90 |
| OVR-002 | Projected ring exposure > $75K |
| OVR-003 | Same attorney/witness/shop across jurisdictions |
| OVR-004 | Previously dismissed ring entity reappears |
| OVR-005 | Licensed attorney is primary connecting edge |
| OVR-006 | Vulnerable claimant (elderly >= 65, language barrier) |
| OVR-007 | Public figure / prior media coverage |
| OVR-008 | Any node has a prior SIU referral |

---

## Render.com Deployment

### Deploy steps

1. Push repo to GitHub
2. Create a **Web Service** on Render and connect the repo
3. Render picks up `render.yaml` automatically (free plan)
4. Set secret env vars in the Render dashboard (`sync: false` fields):
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
   - `OPENROUTER_API_KEY`
   - `PINECONE_API_KEY`

### Free plan RAM budget

| Component | RAM |
| ----------- | ----- |
| PyTorch CPU runtime | ~200 MB |
| paraphrase-MiniLM-L3-v2 weights | ~17 MB |
| FastAPI + dependencies | ~80 MB |
| **Total** | **~300 MB** — fits 512 MB free limit |

> To use the higher-quality `all-MiniLM-L6-v2` model (90 MB weights), set
> `EMBEDDING_MODEL=all-MiniLM-L6-v2` and upgrade to the Standard plan.
> Both models output 384-dim vectors — the Pinecone index does not change.

### Why Pinecone?

Render's free plan has no persistent disk. Pinecone stores the vector index in the cloud so it
survives every deploy and cold start. Run `python src/main.py rag index` once locally to populate
it; the Render service only reads from it at query time.

---

## HITL Feedback Loop

```
Investigator decisions (Approve / Dismiss / Escalate)
    |  stored as HumanReview nodes in Neo4j
    v
FeedbackStore.collect()
    |  pulls rows since last training run, maps to binary labels
    v
compute_f1_delta()
    |  compares model predictions vs investigator ground truth
    v
trigger_retrain()  (requires >= 20 labelled reviews)
    |  saves feedback_labels.json
    |  calls Phase 2 run_training() with feedback-weighted labels
    v  promotes new model only if F1 improves
```

| Decision | Label |
| ---------- | ------- |
| Approve | 1 (fraud confirmed) |
| Escalate | 1 (serious fraud) |
| Dismiss | 0 (false positive) |

---

## Success Targets

| Metric | Launch | Month 6 |
| -------- | -------- | --------- |
| Human Agreement Rate | > 65% | > 80% |
| False Positive Rate | < 35% | < 20% |
| Model AUC-ROC | >= 0.91 | >= 0.95 |
| Mean Time to Decision (Standard) | < 30 min | < 20 min |
| Mean Time to Decision (Override) | < 4 hours | < 2 hours |

---

## Compliance

- **No automated adverse action** — outputs are investigative leads only, never determinations
- **PII hashed** — SSN, bank account, phone, VIN stored as hashed representations only
- **Immutable audit log** — every AI output, investigator decision, and feedback signal logged
- **Quarterly bias audits** — model output distributions reviewed before each version promotion
- **SMOTE/ADASYN training only** — synthetic oversampling never applied at inference time

---

## Technology Stack

| Component | Technology |
| ----------- | ----------- |
| Graph DB | Neo4j Aura |
| GNN | PyTorch Geometric — GraphSAGE (SAGEConv) + HINormer (HGTConv) |
| Ensemble | XGBoost + RandomForest + LightGBM |
| Embeddings | sentence-transformers — paraphrase-MiniLM-L3-v2 (384-dim) |
| Vector store | Pinecone (production) / LocalVectorStore numpy (dev) |
| Agentic pipeline | LangGraph StateGraph |
| LLM | Claude via OpenRouter (openai-compatible API) |
| Web API | FastAPI + uvicorn |
| Deployment | Render.com (free plan) |
