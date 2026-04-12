# CLAUDE.md — Fraud Ring Detection: Graph-RAG System

## Project Overview

Production-grade **auto insurance fraud ring detection** system combining a Neo4j knowledge graph, Graph Neural Networks (GraphSAGE + HINormer), GraphRAG (LangGraph + Claude), and a Human-in-the-Loop (HITL) investigator workflow.

**Current phase:** Phase 3 complete — GraphRAG + LangGraph agentic pipeline with Claude-powered reasoning, NL query engine, and investigator feedback loop.

Full product spec: [PRD.md](PRD.md)

---

## Environment Setup

### Prerequisites

- Python 3.11+
- Neo4j Aura instance (credentials in `.env`)
- Virtual environment at `venv/`

### Install dependencies

```bash
pip install neo4j pandas tqdm python-dotenv
```

### Environment variables (`.env`)

```
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USER=<username>
NEO4J_PASSWORD=<password>
DATA_DIR=./data
BATCH_SIZE=200
NEO4J_TRUST_ALL_CERTS=true   # required on Windows — Aura intermediate CA
```

> `.env` is gitignored. Never commit credentials.

---

## Key Files

| File | Purpose |
|------|---------|
| [load_graph.py](load_graph.py) | Loads all 24 node types + 28 edge types into Neo4j Aura |
| [data/](data/) | All CSV node and edge files (14,292 nodes · 28,690 edges) |
| [PRD.md](PRD.md) | Full product requirements document |

---

## Running the Graph Loader

```bash
# Validate files only (no DB writes)
python load_graph.py --dry-run

# Load using .env credentials
python load_graph.py

# Override specific settings
python load_graph.py --uri "neo4j+s://xxxx.databases.neo4j.io" --password "pw"
python load_graph.py --batch 500
```

After loading, verify in Neo4j Aura console (Query tab):

```cypher
MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC
MATCH ()-[r]->() RETURN type(r), count(r) ORDER BY count(r) DESC
```

---

## Graph Data Model

**Scale:** 14,292 nodes · 28,690 edges · 941 properties · 24 node types · 28 edge types

### Node ID convention

All nodes use `_neo4j_id` (string) as the internal merge key, sourced from the `:ID` column in CSV files. Uniqueness constraints are set on the domain ID property per label (e.g., `claim_id`, `cust_id`).

### Key node types

| Label | Count | Primary ID | Fraud Signals |
|-------|-------|------------|---------------|
| `Customer` | 1,000 | `cust_id` | `synthetic_identity_flag`, `role_switching_flag`, `shared_bank_flag` |
| `Claim` | 1,000 | `claim_id` | `ring_member_flag`, `staged_accident_flag`, `manual_override_flag` |
| `FraudRing` | 20 | `ring_id` | `ring_score`, `status`, `closed_loop_detected` |
| `Witness` | 1,487 | `statement_id` | `professional_witness_flag`, `coached_statement_flag` |
| `InvestigationCase` | 876 | `case_id` | `ai_fraud_score`, `manual_override_triggered` |
| `HumanReview` | 570 | `review_id` | `decision`, `override_ai_recommendation`, `feedback_to_model` |
| `NetworkFeature` | 2,000 | `feature_id` | `pagerank_score`, `ring_suspicion_score`, `hop_2_fraud_count` |

### Key edge types

| Relationship | Pattern |
|---|---|
| `SHARES_ATTRIBUTE` | Customer ↔ Customer — shared phone/bank/IP (CRITICAL/HIGH/MEDIUM/LOW risk) |
| `RING_CONTAINS_CLAIM` | FraudRing → Claim |
| `RING_INVOLVES_CUSTOMER` | FraudRing → Customer |
| `FILED_CLAIM` | Customer → Claim |
| `REPRESENTED_BY` | Claim → Lawyer |
| `REPAIRED_AT` | Claim → RepairShop |
| `HAS_WITNESS` | Claim → Witness |
| `REVIEWS_CASE` | HumanReview → InvestigationCase |

---

## Cypher Query Patterns

### Fraud ring members
```cypher
MATCH (r:FraudRing {ring_id: 'RING-007'})
MATCH (r)-[:RING_INVOLVES_CUSTOMER]->(c:Customer)
MATCH (r)-[:RING_CONTAINS_CLAIM]->(cl:Claim)
RETURN r, c, cl
```

### Professional witnesses (3+ claims)
```cypher
MATCH (w:Witness)<-[:HAS_WITNESS]-(cl:Claim)
WITH w, count(cl) AS claim_count
WHERE claim_count >= 3
RETURN w.full_name, claim_count, w.professional_witness_flag
ORDER BY claim_count DESC
```

### Shared bank account detection
```cypher
MATCH (c1:Customer)-[e:SHARES_ATTRIBUTE]->(c2:Customer)
WHERE e.attribute_type = 'bank_account' AND e.fraud_signal = 'CRITICAL'
MATCH (c1)-[:FILED_CLAIM]->(cl1:Claim)
MATCH (c2)-[:FILED_CLAIM]->(cl2:Claim)
RETURN c1.full_name, c2.full_name, cl1.claim_id, cl2.claim_id
```

### Closed-loop ring detection (Customer → Lawyer → Shop → Customer)
```cypher
MATCH p = (c:Customer)-[:FILED_CLAIM]->(cl:Claim)
  -[:REPRESENTED_BY]->(l:Lawyer)
  -[:CO_LOCATED_WITH]->(s:RepairShop)
  <-[:REPAIRED_AT]-(cl2:Claim)
  <-[:FILED_CLAIM]-(c)
WHERE c.fraud_flag = 'Y'
RETURN p LIMIT 25
```

---

## Architecture — Three Detection Layers

| Layer | Speed | Tech | Trigger |
|-------|-------|------|---------|
| **Layer 1** Rule engine | <100ms | Deterministic Cypher + heuristics | Every claim at ingestion |
| **Layer 2** GNN scoring | 42ms/batch | PyTorch Geometric · GraphSAGE + HINormer | After Layer 1 signals |
| **Layer 3** GraphRAG reasoning | On-demand | LangGraph · Claude · Pinecone/Weaviate | When HITL review triggered |

### Mandatory Override (payment HOLD) triggers

Claims are auto-held and require **senior investigator sign-off** when:
- Ring suspicion score ≥ 90 (`OVR-001`)
- Projected ring exposure > $75K–$100K (`OVR-002`)
- Same attorney/witness/shop across jurisdictions (`OVR-003`)
- Previously dismissed ring entity reappears (`OVR-004`)
- Licensed attorney is the primary connecting edge (`OVR-005`)
- Vulnerable claimant involved (`OVR-006`)
- Public figure / prior media coverage (`OVR-007`)
- Any node has a prior SIU referral (`OVR-008`)

Claim-level auto-flag logic:
```python
manual_override_flag = True if any([
    total_claim_amount > 50000,
    rag_confidence_score < 0.6 and fraud_reported == 'Y',
    hospitalization_required == 'Yes' and police_report_available in ('NO', '?'),
    claim_cluster_id != '',
]) else False
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Graph DB | Neo4j Aura (GDB + GDS) |
| GNN | PyTorch Geometric · GraphSAGE · HINormer |
| Explainability | GNNExplainer / PGExplainer |
| Streaming ingest | Amazon Kinesis + Neo4j Kafka Connector |
| Batch ETL | AWS Glue + EMR |
| Agentic pipeline | LangGraph |
| Vector KB | Pinecone or Weaviate |
| LLM reasoning | Claude via OpenRouter (openai-compatible API) |
| Oversampling | SMOTE + ADASYN (training time only) |
| Visualization | Neo4j Bloom + React dashboard |
| ML serving | Amazon SageMaker |

> **Critical rule:** SMOTE/ADASYN synthetic oversampling must **never** be applied at inference time — training only.

---

## Phased Rollout Status

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** (Months 1–3) | Neo4j graph + rule engine + HITL queue | Complete ✅ |
| **Phase 2** (Months 4–6) | GNN scoring (GraphSAGE + HINormer) | Complete ✅ |
| **Phase 3** (Months 7–9) | GraphRAG + LangGraph agentic pipeline | Complete ✅ |
| **Phase 4** (Months 10–12) | Real-time streaming + cross-carrier exchange | Not started |

## Phase 2 — GNN Scoring

### Files

| File | Purpose |
| ------ | --------- |
| [requirements_phase2.txt](requirements_phase2.txt) | PyTorch, PyG, imbalanced-learn, XGBoost, LightGBM |
| [phase2_gnn/config.py](phase2_gnn/config.py) | Node types, feature column defs, edge types, hyperparameters |
| [phase2_gnn/feature_utils.py](phase2_gnn/feature_utils.py) | Numeric/binary/ordinal feature extraction from CSVs |
| [phase2_gnn/data_loader.py](phase2_gnn/data_loader.py) | CSVs → PyG HeteroData + stratified train/val/test splits |
| [phase2_gnn/model.py](phase2_gnn/model.py) | FraudGNN: input projections + SAGEConv + HGTConv + classifier |
| [phase2_gnn/train.py](phase2_gnn/train.py) | Training loop, SMOTE/ADASYN, XGBoost+RF+LightGBM ensemble |
| [phase2_gnn/explainer.py](phase2_gnn/explainer.py) | Gradient saliency → reasoning trace strings per claim |
| [phase2_gnn/scorer.py](phase2_gnn/scorer.py) | Batch inference + write `gnn_suspicion_score` back to Neo4j |
| [run_phase2.py](run_phase2.py) | CLI: `train` \| `score` \| `explain` \| `evaluate` |

### Install

```bash
# 1. PyTorch (CPU build — replace with GPU variant if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

# 3. Everything else
pip install -r requirements_phase2.txt
```

### Run

```bash
# Train (saves to models/)
python run_phase2.py train

# Score all claims + write to Neo4j
python run_phase2.py score

# Dry-run (no DB writes)
python run_phase2.py score --dry-run

# Explain top 20 flagged claims
python run_phase2.py explain --top-n 20 --output traces.json

# Re-evaluate saved model
python run_phase2.py evaluate
```

### Model architecture

```
Input features (per node type, exact CSV columns)
    │
    ▼  type-specific Linear → hidden_channels (128)
    │
    ▼  SAGEConv (inductive — handles new nodes at inference)
       HeteroConv wrapper: one SAGEConv per edge type
    │
    ▼  BatchNorm + ReLU + Dropout
    │
    ▼  HGTConv (HINormer-equivalent: heterogeneous transformer attention)
       Type-specific attention weights per (src_type, edge_type, dst_type)
    │
    ▼  BatchNorm + ReLU + Dropout
    │
    ▼  Classification head: Linear(128→64) → ReLU → Linear(64→1) → sigmoid
       Applied to Claim node embeddings only
```

### Class imbalance strategy

| Stage | Technique | When |
| ------- | ----------- | ------ |
| GNN training | BCEWithLogitsLoss pos_weight = #neg/#pos | Training only |
| Embedding augmentation | SMOTE + ADASYN on Claim embeddings | Training only ⚠️ |
| Ensemble | XGBoost + RandomForest + LightGBM on augmented embeddings | Training + inference |

**Critical rule:** `apply_smote_adasyn()` is called on training embeddings ONLY. Never on val/test/production data.

### Neo4j properties written by scorer

| Property | Node | Description |
| ---------- | ------ | ------------- |
| `gnn_suspicion_score` | Claim | Raw GNN sigmoid probability |
| `ensemble_suspicion_score` | Claim | Mean of XGB+RF+LGBM probabilities |
| `final_suspicion_score` | Claim | Blended score (0.5×GNN + 0.5×ensemble) |
| `adjuster_priority_tier` | Claim | Critical / High Priority / Standard |
| `ai_fraud_score` | InvestigationCase | final_suspicion_score of linked claim |

### Score tiers

| Score | Tier | Action |
| ------- | ------ | -------- |
| ≥ 0.90 | Critical | Mandatory Override (OVR-001) — payment HOLD |
| ≥ 0.70 | High Priority | Priority queue — 4hr SLA |
| < 0.70 | Standard | Standard queue — 30min SLA |

---

## Data Directory Structure

```
data/
  nodes_*.csv          — 24 node type files
  edges_*.csv          — edge files (fixed-type relationships)
  rel_*.csv            — family relationship files
  edges_SHARED_ATTRIBUTES_master.csv
```

All CSVs use Neo4j bulk import format: `:ID`, `:LABEL`, `:START_ID`, `:END_ID`, `:TYPE` columns plus property columns.

---

## Success Targets

| Metric | Launch | Month 6 |
|--------|--------|---------|
| Human Agreement Rate | > 65% | > 80% |
| False Positive Rate | < 35% | < 20% |
| Model AUC-ROC | ≥ 0.91 | ≥ 0.95 |
| Mean Time to Decision (Standard) | < 30 min | < 20 min |
| Mean Time to Decision (Override) | < 4 hours | < 2 hours |

---

## Compliance Rules

- **No automated adverse action** — system outputs are investigative leads only, never determinations
- **PII never stored raw** — only hashed representations (SSN, bank account, phone, VIN)
- **Immutable audit log** — every AI output, investigator decision, and feedback signal logged with timestamp + user ID
- **Quarterly bias audits** — model output distributions reviewed by compliance officer before each version promotion

---

## Phase 3 — GraphRAG + LangGraph Pipeline

**Status:** ✅ Complete

### Phase 3 Files

| File | Purpose |
| ---- | ------- |
| [requirements_phase3.txt](requirements_phase3.txt) | Phase 3 Python dependencies |
| [run_phase3.py](run_phase3.py) | CLI entry point (index / explain / query / feedback / retrain / stats) |
| [phase3_rag/config.py](phase3_rag/config.py) | API keys, prompts, embedding config |
| [phase3_rag/embedder.py](phase3_rag/embedder.py) | Sentence-transformer embeddings for rings/claims/subgraphs |
| [phase3_rag/vector_store.py](phase3_rag/vector_store.py) | LocalVectorStore (numpy .npz) + PineconeVectorStore |
| [phase3_rag/graph_retriever.py](phase3_rag/graph_retriever.py) | Neo4j subgraph retrieval + HumanReview write-back |
| [phase3_rag/pipeline.py](phase3_rag/pipeline.py) | LangGraph StateGraph: retrieve → embed → reason |
| [phase3_rag/nl_query.py](phase3_rag/nl_query.py) | NL → Cypher → results → NL summary (via Claude) |
| [phase3_rag/feedback.py](phase3_rag/feedback.py) | Investigator feedback collection + retraining trigger |

### Install Phase 3

```bash
pip install -r requirements_phase3.txt
```

Additional environment variables (`.env`):

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-sonnet-4-5   # optional, default shown
VECTOR_STORE_BACKEND=local                      # or "pinecone"
PINECONE_API_KEY=...                            # only if using pinecone backend
PINECONE_INDEX=fraud-rings                      # only if using pinecone backend
```

### Phase 3 Run Commands

```bash
# 1. Index all FraudRing nodes into the vector knowledge base (run once after Phase 1/2)
python run_phase3.py index

# 2. Run full GraphRAG pipeline for a claim
python run_phase3.py explain CLM-521585
python run_phase3.py explain CLM-521585 --verbose   # also prints raw subgraph

# 3. Natural language graph queries
python run_phase3.py query                                          # interactive REPL
python run_phase3.py query --question "Which fraud rings have a lawyer in 3+ claims?"

# 4. Record investigator decision (writes HumanReview node to Neo4j)
python run_phase3.py feedback CLM-521585 --decision Approve   --investigator INV-001
python run_phase3.py feedback CLM-521585 --decision Dismiss   --investigator INV-001 --feedback FP
python run_phase3.py feedback CLM-521585 --decision Escalate  --investigator INV-001 --feedback FN

# 5. Trigger feedback-loop retraining
python run_phase3.py retrain
python run_phase3.py retrain --evaluate-only   # compute F1 metrics without retraining
python run_phase3.py retrain --min-reviews 10  # lower threshold for testing

# 6. Print stats
python run_phase3.py stats
```

### Phase 3 Pipeline Architecture

```
Investigator question
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph (pipeline.py)                             │
│                                                                 │
│  retrieve_subgraph          find_analogous_rings                │
│  ─────────────────          ────────────────────                │
│  Neo4j Cypher query    →    Embed subgraph text           →     │
│  claim neighbours           cosine search vector KB             │
│  (customer, lawyer,         (LocalVectorStore / Pinecone)       │
│   shop, witnesses,          top-K=3 historical rings            │
│   medical, ring,                                                │
│   investigation_case)            generate_reasoning             │
│                                  ─────────────────             │
│                                  Claude via OpenRouter           │
│                                  (anthropic/claude-sonnet-4-5)  │
│                                  REASONING_SYSTEM_PROMPT        │
│                                  + subgraph + analogous rings   │
│                                  + OVR override triggers        │
│                                  → Investigation Brief          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐   ┌──────────────────────────────────┐
│  NL Query Engine        │   │  Feedback Loop                   │
│  (nl_query.py)          │   │  (feedback.py)                   │
│                         │   │                                  │
│  NL → Cypher (Claude)   │   │  Investigator decisions          │
│  safe execute (no write)│   │  → HumanReview nodes (Neo4j)     │
│  results → NL (Claude)  │   │  → F1 delta measurement          │
│  via OpenRouter         │   │  → Phase 2 run_training()        │
└─────────────────────────┘   └──────────────────────────────────┘
```

### Vector Store Backends

| Backend           | Config                                    | Use Case                     |
| ----------------- | ----------------------------------------- | ---------------------------- |
| `local` (default) | No extra config                           | Development / ≤ 100K rings   |
| `pinecone`        | `PINECONE_API_KEY` + `PINECONE_INDEX`     | Production ANN search        |

### Investigator Feedback → Label Mapping

| Decision | Label | Meaning                 |
| -------- | ----- | ----------------------- |
| Approve  | 1     | Confirms fraud          |
| Escalate | 1     | Confirmed serious fraud |
| Dismiss  | 0     | False positive          |

`feedback_to_model` signal: `Correct` / `FP` / `FN` / `Uncertain` (Uncertain excluded from retraining)

### LLM Routing via OpenRouter

All Claude calls go through [OpenRouter](https://openrouter.ai) using the OpenAI-compatible API (`openai` Python package, `base_url=https://openrouter.ai/api/v1`). Three call sites:

- `pipeline.py` — REASONING_SYSTEM_PROMPT → investigation brief generation
- `nl_query.py` — NL_QUERY_SYSTEM_PROMPT → Cypher generation
- `nl_query.py` — RESULT_FORMATTER_SYSTEM_PROMPT → result summarisation

Model is configurable via `OPENROUTER_MODEL` env var. Default: `anthropic/claude-sonnet-4-5`.

### Cypher Safety

The NL query engine enforces read-only access via regex block:

```python
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(MERGE|CREATE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+apoc\.periodic)\b",
    re.IGNORECASE,
)
```

Any generated Cypher containing write operations raises `ValueError` before execution.

### Known Fixes Applied

| Issue | File | Fix |
| ----- | ---- | --- |
| `OPENROUTER_API_KEY` read as `None` despite being set in `.env` | [phase3_rag/config.py](phase3_rag/config.py) | Added `load_dotenv()` call at top of config before `os.getenv()` |
| `FutureWarning: get_sentence_embedding_dimension` renamed in sentence-transformers ≥ 3.x | [phase3_rag/embedder.py](phase3_rag/embedder.py) | `getattr` probe for new `get_embedding_dimension()` name with fallback to old name |
| `ReduceLROnPlateau` `verbose` kwarg removed in PyTorch ≥ 2.2 | [phase2_gnn/train.py](phase2_gnn/train.py) | Removed `verbose=False` from scheduler init |
| LightGBM `UserWarning: X does not have valid feature names` at inference | [phase2_gnn/scorer.py](phase2_gnn/scorer.py), [train.py](phase2_gnn/train.py) | Cast to plain numpy + `warnings.catch_warnings()` filter |
