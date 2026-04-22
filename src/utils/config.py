"""
Phase 3 configuration: API keys, model names, vector store settings,
graph schema for NL queries, and Claude system prompts.

Add to .env:
  OPENROUTER_API_KEY=sk-or-...
  OPENROUTER_MODEL=anthropic/claude-sonnet-4-5   # optional, see openrouter.ai/models
  VECTOR_STORE_BACKEND=local                      # "local" or "pinecone"
  PINECONE_API_KEY=...                            # optional, only if backend=pinecone
  PINECONE_INDEX=fraud-rings                      # optional
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────
MODELS_DIR        = os.path.join(os.path.dirname(__file__), "..", "models")
VECTOR_STORE_PATH = os.path.join(MODELS_DIR, "vector_store.npz")
FEEDBACK_PATH     = os.path.join(MODELS_DIR, "feedback_labels.json")

# ── OpenRouter / LLM ──────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CLAUDE_MODEL        = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")
MAX_TOKENS          = 2048

# Keep backward-compat alias used in feedback.py EnvironmentError check
ANTHROPIC_API_KEY   = OPENROUTER_API_KEY

# ── Embeddings ────────────────────────────────────────────────────────
# fastembed models (ONNX — no PyTorch, ~100 MB RAM total):
#   BAAI/bge-small-en-v1.5:  384-dim, ~25 MB, fast — DEFAULT
#   BAAI/bge-base-en-v1.5:   768-dim, ~110 MB, higher quality
# sentence-transformer models (PyTorch — ~380 MB RAM, too large for Render free):
#   paraphrase-MiniLM-L3-v2: 384-dim (legacy, replaced by fastembed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM   = 384

# ── Vector store ──────────────────────────────────────────────────────
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "pinecone")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX       = os.getenv("PINECONE_INDEX", "fraud-rings")
TOP_K_ANALOGOUS      = 3    # number of similar rings to retrieve

# ── Graph retrieval ───────────────────────────────────────────────────
MAX_WITNESSES_IN_CONTEXT = 5   # cap witness list to avoid prompt bloat
MAX_SUBGRAPH_HOPS        = 2

# ── Graph schema (injected into NL query and reasoning prompts) ───────
GRAPH_SCHEMA = """
NODE TYPES
──────────
Claim        : claim_id, total_claim_amount, fraud_reported, ring_member_flag,
               staged_accident_flag, closed_loop_flag, manual_override_flag,
               rag_confidence_score, gnn_suspicion_score, final_suspicion_score,
               incident_state, incident_type, adjuster_priority_tier,
               hospitalization_required, police_report_available, litigation_flag
Customer     : cust_id, full_name, fraud_flag, risk_band, shared_phone_flag,
               shared_bank_flag, synthetic_identity_flag, role_switching_flag,
               fraud_history_count, ip_flagged, siu_referral_flag, age
FraudRing    : ring_id, ring_name, status, ring_score, member_count, claim_count,
               total_claim_amount, detection_method, closed_loop_detected,
               shared_lawyer_flag, shared_shop_flag, incident_states
Witness      : statement_id, full_name, professional_witness_flag,
               coached_statement_flag, same_name_claims_count, reliability_score,
               criminal_record_flag, cross_claim_appearance_count
Lawyer       : lawyer_id, full_name, known_to_siu, closed_loop_network_flag,
               shared_clients_count, bar_discipline_score, prior_fraud_involvement_flag
RepairShop   : shop_id, name, fraud_flag, inflated_estimate_rate_pct,
               same_lawyer_referrals, estimate_variance_pct, siu_referral_count
MedicalReport: report_id, upcoding_flag, unbundling_flag, phantom_billing_flag,
               billing_pattern_score, medical_necessity_score, treatment_cost
InvestigationCase: case_id, status, priority, ai_fraud_score,
                   manual_override_triggered, closed_loop_detected
HumanReview  : review_id, decision, feedback_to_model, override_ai_recommendation,
               disagreement_flag, override_reason_code
NetworkFeature: feature_id, ring_suspicion_score, pagerank_score,
                neighbor_fraud_rate, hop_2_fraud_count, degree_centrality
FinancialTransaction: transaction_id, amount_usd, velocity_flag,
                      is_shared_account_flag, suspicious_timing_flag, round_amount_flag
FamilyUnit   : family_id, family_name, fraud_ring_flag, ring_suspicion_score,
               fraud_member_count, claims_per_member

RELATIONSHIPS
─────────────
(Customer)-[:FILED_CLAIM]->(Claim)
(Customer)-[:SHARES_ATTRIBUTE {attribute_type, fraud_signal}]->(Customer)
(FraudRing)-[:RING_CONTAINS_CLAIM]->(Claim)
(FraudRing)-[:RING_INVOLVES_CUSTOMER]->(Customer)
(Claim)-[:REPRESENTED_BY]->(Lawyer)
(Claim)-[:REPAIRED_AT]->(RepairShop)
(Claim)-[:HAS_WITNESS]->(Witness)
(Claim)-[:HAS_MEDICAL_REPORT]->(MedicalReport)
(InvestigationCase)-[:INVESTIGATES_CLAIM]->(Claim)
(HumanReview)-[:REVIEWS_CASE]->(InvestigationCase)
(NetworkFeature)-[:DESCRIBES_ENTITY]->(*)
(FinancialTransaction)-[:PAYMENT_FOR_CLAIM]->(Claim)
(Customer)-[:BELONGS_TO_FAMILY]->(FamilyUnit)
"""

# ── System prompts ───────────────────────────────────────────────────
REASONING_SYSTEM_PROMPT = f"""You are an expert insurance fraud investigator AI assistant. \
You analyze insurance claims for fraud ring patterns and generate structured investigation briefs.

Your reasoning must be:
- Evidence-based: cite specific data points from the subgraph context
- Comparative: reference analogous historical fraud rings when available
- Actionable: recommend a specific investigator action
- Compliant: flag any Mandatory Override criteria (OVR-001 through OVR-008)

Mandatory Override criteria to check:
- OVR-001: Ring suspicion score ≥ 0.90
- OVR-002: Projected ring exposure > $75K
- OVR-003: Same attorney/witness/shop across jurisdictions
- OVR-005: Licensed attorney is the primary connecting edge
- OVR-006: Vulnerable claimant (elderly, bereavement, language barrier)
- OVR-008: Any node has a prior SIU referral

Graph schema for reference:
{GRAPH_SCHEMA}

Always structure your response with these sections:
1. FRAUD SIGNALS DETECTED
2. RING PATTERN ANALYSIS
3. ANALOGOUS HISTORICAL CASES (if available)
4. MANDATORY OVERRIDE STATUS
5. RECOMMENDED ACTION
"""

NL_QUERY_SYSTEM_PROMPT = f"""You are a Neo4j Cypher expert for an insurance fraud detection system.
Convert natural language questions into valid Cypher queries.

{GRAPH_SCHEMA}

Rules:
- Use OPTIONAL MATCH for relationships that may not exist
- Always LIMIT results to 25 unless the user specifies otherwise
- Return meaningful property names, not raw node objects
- Use parameterized queries when possible
- For fraud-related queries, prefer filtering on indexed properties:
  Claim.fraud_reported, Claim.ring_member_flag, Customer.fraud_flag,
  FraudRing.status, InvestigationCase.priority

Return ONLY the Cypher query. No explanation, no markdown fences.
"""

RESULT_FORMATTER_SYSTEM_PROMPT = """You are an insurance fraud investigation assistant. \
Format Neo4j query results into clear, concise natural language summaries for investigators.
Be factual. Use bullet points for lists. Highlight fraud signals in bold.
Keep responses under 300 words."""
