# CLAUDE.md — Fraud Ring Detection: Graph-RAG System

## Project Overview

Production-grade **auto insurance fraud ring detection** system combining a Neo4j knowledge graph, Graph Neural Networks (GraphSAGE + HINormer), GraphRAG (LangGraph + Claude), and a Human-in-the-Loop (HITL) investigator workflow.

**Current phase:** Phase 1 complete — full graph loaded into Neo4j Aura.

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
| LLM reasoning | Claude (Anthropic API) |
| Oversampling | SMOTE + ADASYN (training time only) |
| Visualization | Neo4j Bloom + React dashboard |
| ML serving | Amazon SageMaker |

> **Critical rule:** SMOTE/ADASYN synthetic oversampling must **never** be applied at inference time — training only.

---

## Phased Rollout Status

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** (Months 1–3) | Neo4j graph + rule engine + HITL queue | Graph loaded ✅ |
| **Phase 2** (Months 4–6) | GNN scoring (GraphSAGE + HINormer) | Not started |
| **Phase 3** (Months 7–9) | GraphRAG + LangGraph agentic pipeline | Not started |
| **Phase 4** (Months 10–12) | Real-time streaming + cross-carrier exchange | Not started |

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
