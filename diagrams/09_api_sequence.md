# API Sequence Diagrams

Interaction sequences between Streamlit UI, FastAPI, and backend services.

## Investigate Claim — Full Sequence

```mermaid
sequenceDiagram
    actor INV as Investigator
    participant STR as Streamlit UI
    participant API as FastAPI (Render)
    participant NEO as Neo4j Aura
    participant EMB as MiniLM Embedder
    participant PCN as Pinecone
    participant LLM as OpenRouter / Claude

    INV->>STR: Enter claim ID "CLM-521585"\nClick Investigate

    STR->>API: GET /investigate/CLM-521585
    Note over STR,API: HTTPS · timeout 60s

    API->>NEO: MATCH (c:Claim {claim_id: 'CLM-521585'})\nRETURN c.final_suspicion_score, c.adjuster_priority_tier
    NEO-->>API: {score: 0.94, tier: "Critical"}

    API->>NEO: MATCH Customer → FILED_CLAIM → Claim
    NEO-->>API: customer properties

    API->>NEO: MATCH Claim → REPRESENTED_BY → Lawyer
    NEO-->>API: lawyer properties

    API->>NEO: MATCH Claim → REPAIRED_AT → RepairShop
    NEO-->>API: shop properties

    API->>NEO: MATCH Claim → HAS_WITNESS → Witness (LIMIT 5)
    NEO-->>API: witnesses list

    API->>NEO: MATCH FraudRing → RING_CONTAINS_CLAIM
    NEO-->>API: ring properties

    API->>NEO: MATCH InvestigationCase → INVESTIGATES_CLAIM
    NEO-->>API: case properties

    API->>NEO: MATCH NetworkFeature → DESCRIBES_ENTITY (Customer)
    NEO-->>API: network features

    API->>NEO: MATCH Customer -[SHARES_ATTRIBUTE]-> Customer
    NEO-->>API: shared attribute edges

    Note over API: Build subgraph_text string\n~2,000 chars

    API->>EMB: encode(subgraph_text)
    EMB-->>API: 384-dim vector

    API->>PCN: query(vector, top_k=3)
    Note over PCN: cosine similarity search
    PCN-->>API: [{ring_id, score, metadata}] × 3

    Note over API: Evaluate OVR triggers\nOVR-001 to OVR-008

    API->>LLM: chat.completions.create\nmodel: anthropic/claude-sonnet-4-5\nsystem: REASONING_SYSTEM_PROMPT\nuser: subgraph_text + analogous_rings + score
    LLM-->>API: Investigation Brief\n~800 tokens

    API-->>STR: JSON {\n  fraud_score: 0.94,\n  override_triggers: ["OVR-001","OVR-005"],\n  analogous_rings: [...],\n  reasoning_trace: "1. FRAUD SIGNALS..."\n}

    STR-->>INV: Render UI\nRed banner: CRITICAL (0.94)\nOverride triggers list\nAnalogous rings table\nClaude Investigation Brief
```

## Record Feedback — Sequence

```mermaid
sequenceDiagram
    actor INV as Investigator
    participant STR as Streamlit UI
    participant API as FastAPI (Render)
    participant NEO as Neo4j Aura
    participant FS as FeedbackStore

    INV->>STR: Fill feedback form\nclaim_id: CLM-521585\ndecision: Approve\nconfidence: 90\nfeedback: Correct

    STR->>API: POST /feedback\n{\n  claim_id: "CLM-521585",\n  decision: "Approve",\n  investigator_id: "INV-001",\n  confidence: 90,\n  feedback_to_model: "Correct"\n}

    API->>NEO: CREATE (hr:HumanReview {...})\nMATCH (ic:InvestigationCase)\nCREATE (hr)-[:REVIEWS_CASE]->(ic)
    NEO-->>API: review_id written

    API->>FS: collect_feedback()\npull new HumanReview nodes\nmap to binary labels
    FS-->>API: {labelled: 47, pending_retrain: 7}

    API-->>STR: {"success": true, "review_id": "REV-1234", "pending_retrain": 7}
    STR-->>INV: "Feedback recorded. 7 reviews pending retraining."
```

## NL Graph Query — Sequence

```mermaid
sequenceDiagram
    actor INV as Investigator
    participant STR as Streamlit UI
    participant API as FastAPI (Render)
    participant LLM as OpenRouter / Claude
    participant NEO as Neo4j Aura

    INV->>STR: Type question:\n"Which lawyers appear in\nmore than 3 fraud ring claims?"

    STR->>API: POST /query\n{"question": "Which lawyers..."}

    API->>LLM: NL_QUERY_SYSTEM_PROMPT\n+ few-shot Cypher examples\n+ user question
    LLM-->>API: MATCH (l:Lawyer)<-[:REPRESENTED_BY]-(c:Claim)\nWHERE c.ring_member_flag = true\nWITH l, count(c) AS cnt\nWHERE cnt > 3\nRETURN l.lawyer_id, l.bar_number, cnt

    Note over API: FORBIDDEN_KEYWORDS check\nno MERGE/CREATE/DELETE/SET

    API->>NEO: Execute Cypher (read-only)
    NEO-->>API: [{lawyer_id: "L-001", cnt: 6}, ...]

    API->>LLM: RESULT_FORMATTER_SYSTEM_PROMPT\n+ raw rows JSON\n+ original question
    LLM-->>API: "Two lawyers appear in more than\n3 fraud ring claims: L-001 (6 claims)\nand L-003 (4 claims)..."

    API-->>STR: {"answer": "Two lawyers...", "cypher": "MATCH..."}
    STR-->>INV: NL answer + expandable Cypher panel
```

## REST API Endpoints

```mermaid
flowchart LR
    subgraph ENDPOINTS["FastAPI — api.py"]
        direction TB
        E1["GET /health\n→ {status: ok, version: ...}"]
        E2["GET /investigate/{claim_id}\n→ PipelineResult\n  fraud_score\n  override_triggers\n  analogous_rings\n  reasoning_trace"]
        E3["POST /feedback\nbody: FeedbackRequest\n  claim_id · decision\n  investigator_id\n  confidence\n  feedback_to_model\n→ {success, review_id}"]
        E4["POST /query\nbody: {question: str}\n→ {answer, cypher, rows}"]
        E5["GET /stats\n→ {total_reviews\n   vector_store_size\n   f1_history}"]
    end
```
