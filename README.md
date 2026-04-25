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
| Phase 1 — Graph Load | Neo4j graph + rule engine + HITL queue | ✓ Complete |
| Phase 2 — GNN Scoring | GraphSAGE + HINormer + XGB/RF/LGBM ensemble | ✓ Complete |
| Phase 3 — GraphRAG | LangGraph pipeline + NL query + feedback loop | ✓ Complete |
| Phase 3b — Frontend | Streamlit UI + FastAPI on Render | ✓ Complete |
| Phase 4 — Streaming | Real-time Kinesis ingest + cross-carrier exchange | Not started |

**Graph scale:** 14,292 nodes · 28,690 edges · 941 properties · 24 node types · 28 edge types

---

## Repository Structure

```text
FraudRingDetectionGraph-RAGSystem/
├── src/                               Backend application source
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
│       ├── embedder.py                fastembed (ONNX) ring/claim embeddings
│       └── feature_utils.py           Feature extraction for GNN
│
├── ui/                                Streamlit frontend (deploy to Streamlit Cloud)
│   ├── streamlit_app.py               Main Streamlit app (4 tabs)
│   ├── requirements.txt               Frontend deps: streamlit, requests, pandas
│   └── .streamlit/
│       ├── config.toml                Dark theme
│       └── secrets.toml.example       API URL template (copy -> secrets.toml)
│
├── data/                              CSV node and edge files
│   ├── nodes_*.csv                    24 node type files
│   └── edges_*.csv / rel_*.csv        Edge and relationship files
│
├── models/                            Saved artefacts (git-ignored)
│   ├── fraud_gnn.pt                   Trained GNN weights
│   ├── ensemble.pkl                   XGB + RF + LGBM ensemble
│   └── scaler.pkl                     Feature scaler
│
├── api.py                             FastAPI web service (deploy to Render.com)
├── build.sh                           Render build script
├── render.yaml                        Render service definition
├── requirements.txt                   API production dependencies
└── requirements_phase2.txt            Local-only: PyTorch Geometric, XGBoost, LightGBM
```

---

## External Services Required

| Service | Purpose | Free tier |
| ------- | ------- | --------- |
| Neo4j Aura | Graph database | Free (200K nodes) |
| OpenRouter | LLM API | Claude: pay-per-token · Gemma: free (with Google key) |
| Pinecone | Vector store | Free (1 index, 100K vectors) |
| Render.com | API hosting | Free (512 MB RAM, fastembed fits) |
| Streamlit Community Cloud | UI hosting | Free |
| Google AI Studio | Gemma API (optional) | Free tier available |
| LangSmith | LLM observability & tracing (optional) | Free (7-day retention, unlimited traces) |

---

## Setup

### System Requirements

- **Python:** 3.11 or newer
- **Virtual environment:** Python venv recommended
- **Git:** For cloning the repository
- **Memory:** 512 MB minimum (with fastembed ONNX embeddings, no PyTorch)
- **Network:** Internet access for Neo4j Aura, OpenRouter, Pinecone

### External Accounts (Free Tier Available)

1. **Neo4j Aura** — Graph database (Free: 200K nodes limit)
2. **OpenRouter** — LLM API (Free: Gemma, paid: Claude)
3. **Pinecone** — Vector search (Free: 1 index, 100K vectors)
4. **Render.com** — API hosting (Free: 512 MB RAM)
5. **Streamlit Cloud** — UI hosting (Free)
6. **LangSmith** — Observability (Free: 7-day traces, unlimited)

---

## Installation

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/saikiranaravala/FraudRingDetectionGraph-RAGSystem.git
cd FraudRingDetectionGraph-RAGSystem

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env  # (if available)
# Edit .env with your API keys and credentials

# 5. Start API (optional, for local testing)
uvicorn api:app --reload

# 6. Start Streamlit UI (optional, in another terminal)
streamlit run ui/streamlit_app.py
```

### Detailed Installation Steps

See **Local Development** section below for comprehensive step-by-step instructions including Pinecone index creation, graph data loading, and GNN model training.

---

## Local Development

### Prerequisites

- Python 3.11+
- Git

### Step 1 — Clone and create virtual environment

```bash
git clone <repo-url>
cd FraudRingDetectionGraph-RAGSystem
python -m venv venv

# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
# API backend (FastAPI, LangGraph, embeddings, Neo4j, Pinecone)
pip install -r requirements.txt

# Streamlit frontend (separate, lightweight)
pip install -r ui/requirements.txt

# Phase 2 GNN training (optional — only needed to retrain the model)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements_phase2.txt
```

### Step 3 — Configure `.env`

Create `.env` in the project root:

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

# LangSmith monitoring (optional — for debugging and observability)
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=FraudRingDetectionGraph-RAG
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Data
DATA_DIR=./data
BATCH_SIZE=200
```

> **LangSmith setup (optional):** Get a free API key at [smith.langchain.com](https://smith.langchain.com). When enabled, all LangGraph node executions and LLM calls are traced and visible in the LangSmith dashboard for debugging and performance monitoring.

### Step 4 — Create Pinecone index (one-time)

In the [Pinecone console](https://app.pinecone.io), create an index:

| Setting | Value |
| ------- | ----- |
| Name | `fraud-rings` |
| Dimensions | `384` |
| Metric | `Cosine` |
| Pod type | `Starter` (free) |

### Step 5 — Load graph data into Neo4j (one-time)

```bash
# Validate CSV files first (no DB writes)
python src/main.py load-graph --dry-run

# Load all 14,292 nodes and 28,690 edges
python src/main.py load-graph
```

### Step 6 — Train GNN and score claims (one-time, optional)

```bash
python src/main.py gnn train                                    # trains + saves to models/
python src/main.py gnn score                                    # writes scores to Neo4j
python src/main.py gnn explain --top-n 20 --output traces.json
```

### Step 7 — Index fraud rings into Pinecone (one-time)

```bash
# Embeds all FraudRing nodes and pushes 384-dim vectors to Pinecone
python src/main.py rag index
```

### Step 8 — Run the API backend

```bash
uvicorn api:app --reload --port 8000
# API docs at http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

### Step 9 — Run the Streamlit frontend

Open a second terminal (keep the API running):

```bash
# Activate venv if not already active
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

streamlit run ui/streamlit_app.py
# Opens at http://localhost:8501
```

In the Streamlit sidebar, set the **Backend URL** to `http://localhost:8000` and click **Check Connection**.

### CLI commands (without the UI)

```bash
# GraphRAG investigation brief
python src/main.py rag explain CLM-521585
python src/main.py rag explain CLM-521585 --verbose

# Natural language queries
python src/main.py rag query --question "Which fraud rings have a lawyer in 3+ claims?"
python src/main.py rag query                # interactive REPL

# Record investigator decisions
python src/main.py rag feedback CLM-521585 --decision Approve   --investigator INV-001
python src/main.py rag feedback CLM-521585 --decision Dismiss   --investigator INV-001 --feedback FP
python src/main.py rag feedback CLM-521585 --decision Escalate  --investigator INV-001 --feedback FN

# Feedback-loop retraining
python src/main.py rag retrain --evaluate-only
python src/main.py rag retrain --min-reviews 20

# Stats
python src/main.py rag stats
```

---

## Production Deployment

### Overview

```text
GitHub repo
    │
    ├──► Render.com          → hosts api.py        (FastAPI backend)
    │         URL: https://your-service.onrender.com
    │
    └──► Streamlit Cloud     → hosts ui/streamlit_app.py  (Streamlit frontend)
              URL: https://your-app.streamlit.app
```

The Streamlit frontend calls the Render API over HTTPS. No backend code runs in Streamlit Cloud.

---

### Part A — Deploy API to Render.com

#### A1 — Push repo to GitHub

```bash
git add .
git commit -m "ready for deployment"
git push origin main
```

#### A2 — Create Render Web Service

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Render detects `render.yaml` automatically and pre-fills:
   - **Build command:** `bash build.sh`
   - **Start command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free

#### A3 — Set secret environment variables

In the Render dashboard → **Environment**, add these (marked **Secret**):

| Key | Value |
| --- | ----- |
| `NEO4J_URI` | `neo4j+s://<instance-id>.databases.neo4j.io` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASSWORD` | your Neo4j password |
| `OPENROUTER_API_KEY` | `sk-or-...` |
| `PINECONE_API_KEY` | your Pinecone API key |

These are already declared in `render.yaml` with `sync: false` — Render will prompt for their values at deploy time.

#### A4 — Deploy

Click **Create Web Service**. Render runs `build.sh` (installs CPU torch + requirements.txt) then starts the API.

**Verify:** `https://your-service.onrender.com/health` should return `{"status": "ok"}`.

> **Free plan note:** The service spins down after 15 minutes of inactivity. The first request after a cold start takes ~30 seconds. Upgrade to the Starter plan ($7/mo) to eliminate cold starts.

---

### Part B — Deploy UI to Streamlit Community Cloud

#### B1 — Configure Streamlit secrets locally

```bash
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml
```

Edit `ui/.streamlit/secrets.toml`:

```toml
api_url = "https://your-service.onrender.com"
```

> `ui/.streamlit/secrets.toml` is gitignored — never commit it.

#### B2 — Create Streamlit Cloud app

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Fill in the fields:

   | Field | Value |
   | ----- | ----- |
   | Repository | your GitHub repo |
   | Branch | `main` |
   | Main file path | `ui/streamlit_app.py` |
   | App URL | choose a name (e.g. `fraud-ring-detection`) |

3. Click **Advanced settings** and set the **Python version** to `3.11`

#### B3 — Add secrets in Streamlit Cloud

In **Advanced settings → Secrets**, paste:

```toml
api_url = "https://your-service.onrender.com"
```

Replace the URL with your actual Render service URL from Part A.

#### B4 — Deploy

Click **Deploy**. Streamlit Cloud installs from `ui/requirements.txt` (only `streamlit`, `requests`, `pandas` — no heavy ML deps).

**Verify:** The app opens, enter the Render URL in the sidebar, click **Check Connection** — should show **API online**.

---

### Free Plan RAM Budget (Render)

| Component | RAM |
| --------- | --- |
| PyTorch CPU runtime | ~200 MB |
| `paraphrase-MiniLM-L3-v2` weights | ~17 MB |
| FastAPI + all other libs | ~80 MB |
| **Total** | **~300 MB — fits 512 MB free limit** |

> To use the higher-quality `all-MiniLM-L6-v2` model (90 MB, better embeddings), set `EMBEDDING_MODEL=all-MiniLM-L6-v2` in Render and upgrade to the Standard plan. Both models output 384-dim vectors — the Pinecone index does not change.

---

## REST API Reference

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `GET` | `/health` | Liveness probe |
| `POST` | `/explain/{claim_id}` | Full GraphRAG investigation brief |
| `POST` | `/query` | Natural language graph query |
| `POST` | `/feedback/{claim_id}` | Record investigator decision |
| `GET` | `/stats` | Feedback history + vector store size |

Interactive docs: `http://localhost:8000/docs` (local) or `https://your-service.onrender.com/docs` (production).

---

## GNN Model

### Model Architecture

```text
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
| ----- | ---- | ------ |
| >= 0.90 | Critical | Mandatory Override — payment HOLD |
| >= 0.70 | High Priority | Priority queue — 4 hr SLA |
| < 0.70 | Standard | Standard queue — 30 min SLA |

### Properties written to Neo4j by the scorer

| Property | Node | Description |
| -------- | ---- | ----------- |
| `gnn_suspicion_score` | Claim | Raw GNN sigmoid probability |
| `ensemble_suspicion_score` | Claim | Mean of XGB + RF + LGBM |
| `final_suspicion_score` | Claim | 0.5 x GNN + 0.5 x ensemble |
| `adjuster_priority_tier` | Claim | Critical / High Priority / Standard |
| `ai_fraud_score` | InvestigationCase | `final_suspicion_score` of linked claim |

---

## GraphRAG Pipeline

```text
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
| ---- | --------- |
| OVR-001 | Ring suspicion score >= 0.90 |
| OVR-002 | Projected ring exposure > $75K |
| OVR-003 | Same attorney/witness/shop across jurisdictions |
| OVR-004 | Previously dismissed ring entity reappears |
| OVR-005 | Licensed attorney is primary connecting edge |
| OVR-006 | Vulnerable claimant (elderly >= 65, language barrier) |
| OVR-007 | Public figure / prior media coverage |
| OVR-008 | Any node has a prior SIU referral |

---

## HITL Feedback Loop

```text
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
| -------- | ----- |
| Approve | 1 (fraud confirmed) |
| Escalate | 1 (serious fraud) |
| Dismiss | 0 (false positive) |

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

## Compliance

- **No automated adverse action** — outputs are investigative leads only, never determinations
- **PII hashed** — SSN, bank account, phone, VIN stored as hashed representations only
- **Immutable audit log** — every AI output, investigator decision, and feedback signal logged
- **Quarterly bias audits** — model output distributions reviewed before each version promotion
- **SMOTE/ADASYN training only** — synthetic oversampling never applied at inference time

---

## Technology Stack

| Component | Technology |
| --------- | ---------- |
| Graph DB | Neo4j Aura |
| GNN | PyTorch Geometric — GraphSAGE (SAGEConv) + HINormer (HGTConv) |
| Ensemble | XGBoost + RandomForest + LightGBM |
| Embeddings | sentence-transformers — paraphrase-MiniLM-L3-v2 (384-dim) |
| Vector store | Pinecone (production) / LocalVectorStore numpy (dev) |
| Agentic pipeline | LangGraph StateGraph |
| LLM | Claude via OpenRouter (openai-compatible API) |
| Web API | FastAPI + uvicorn |
| Frontend | Streamlit |
| API hosting | Render.com (free plan) |
| UI hosting | Streamlit Community Cloud (free) |
