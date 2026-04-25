---
name: Graph Data Patterns & NL Query Guidelines
description: Actual data distributions in fraud graph; which queries work; witness query fix
type: reference
originSessionId: 61524b88-846a-4d26-9715-c826e91fc72a
---
## Critical Finding (2026-04-24)

**Original NL query example "List witnesses who appear in more than 2 claims" returns 0 results because NO witnesses appear in multiple claims.**

All 10 witnesses in the graph have exactly 1 claim each (claim_count = 1). This is a **data characteristic**, not a bug.

## Actual Data Distributions

### Lawyers (7 total)
- **Top lawyer:** Susan M. Olson — 149 claims
- **Range:** 100-150 claims per lawyer
- **Key property:** Some appear across fraud rings with shared clients

**Works:** "Which lawyers appear in the most claims?" ✓

### Witnesses (10 total)
- **Distribution:** All have exactly 1 claim
- **Professional witnesses:** No professional_witness_flag set to true
- **Criminal records:** No criminal_record_flag set to true

**Doesn't work:** Queries about multi-claim witnesses, professional witnesses, criminal records ✗

### Fraud Rings (11+ total)
- **High-risk rings:** 11+ rings with ring_score >= 0.9 (scores 0.9-1.0)
- **Claim counts:** 8-12 claims per ring
- **Example:** RING-002 (score=1.0, 12 claims), RING-016 (score=1.0, 11 claims)

**Works:** "Show fraud rings with the highest scores" ✓

### Repair Shops (7 total)
- **Distribution:** Some appear in 12-16 fraud rings
- **Top shop:** FastFix Collision — 16 fraud rings, 26+ claims
- **Flagged shops:** Some have fraud_flag='Yes'

**Works:** "Which repair shops appear in the most fraud ring claims?" ✓

### Shared Attributes (CRITICAL)
- **Pattern:** Customers sharing bank accounts, phones, addresses
- **Fraud signal:** All marked CRITICAL in SHARES_ATTRIBUTE edges
- **Examples:** Gregory R. Robinson ↔ Nicole L. Phillips (shared bank), Kimberly K. Hall ↔ Nicholas D. Phillips

**Works:** "Find customers who share bank accounts" ✓

### High-Risk Claims (25+ total)
- **Distribution:** 25+ claims with final_suspicion_score >= 0.70
- **Range:** 0.7611 - 0.7634 (high priority tier)
- **Incident states:** Distributed across WV, SC, OH, etc.

**Works:** "Show me high-risk claims that need investigation" ✓

## Updated Few-Shot Examples

These are the new examples that replaced the unrealistic ones:

```
Q: Show me fraud rings with the highest scores
A: MATCH (r:FraudRing)-[:RING_CONTAINS_CLAIM]->(c:Claim) 
   RETURN r.ring_id, r.ring_name, r.ring_score, count(c) AS claim_count 
   ORDER BY r.ring_score DESC LIMIT 25

Q: Which lawyers appear in the most claims?
A: MATCH (l:Lawyer)<-[:REPRESENTED_BY]-(c:Claim) 
   RETURN l.full_name, l.known_to_siu, count(c) AS claim_count 
   ORDER BY claim_count DESC LIMIT 25

Q: Find customers who share bank accounts
A: MATCH (c1:Customer)-[s:SHARES_ATTRIBUTE]->(c2:Customer) 
   WHERE s.attribute_type = 'bank_account' AND s.fraud_signal = 'CRITICAL' 
   RETURN c1.full_name, c2.full_name, s.fraud_signal LIMIT 25

Q: Which repair shops appear in the most fraud ring claims?
A: MATCH (s:RepairShop)<-[:REPAIRED_AT]-(c:Claim)<-[:RING_CONTAINS_CLAIM]-(r:FraudRing) 
   WITH s, count(DISTINCT r) AS ring_count, count(c) AS claim_count 
   RETURN s.name, s.fraud_flag, ring_count, claim_count 
   ORDER BY ring_count DESC LIMIT 25

Q: Show me high-risk claims that need investigation
A: MATCH (c:Claim) WHERE c.final_suspicion_score >= 0.70 
   RETURN c.claim_id, c.final_suspicion_score, c.incident_state, c.total_claim_amount 
   ORDER BY c.final_suspicion_score DESC LIMIT 25
```

## How to Apply

When updating NL query few-shot examples in future sessions:
1. Always validate examples against actual data with Cypher queries
2. Run: `python -c "from src.services.graph_retriever import GraphRetriever; r = GraphRetriever(); print(r.run_cypher('your_cypher'))"`
3. Check that examples return >= 1 result (not empty)
4. Update `src/tools/nl_query.py` FEW_SHOT_EXAMPLES with working patterns
5. Update `ui/streamlit_app.py` example_questions to match

**Why:** Unrealistic examples confuse Claude into generating Cypher that returns 0 results. Few-shot examples should mirror actual graph patterns.
