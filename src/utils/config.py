"""
Phase 3 configuration: API keys, model names, vector store settings,
graph schema for NL queries, and LLM system prompts.

Add to .env:
  OPENROUTER_API_KEY=sk-or-...
  OPENROUTER_MODEL=google/gemma-4-31b-it:free   # free model, or anthropic/claude-sonnet-4-5
  VECTOR_STORE_BACKEND=pinecone                  # "local" or "pinecone"
  PINECONE_API_KEY=...                           # required if backend=pinecone
  PINECONE_INDEX=fraud-rings                     # optional
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
# Default to google/gemma-4-31b-it:free (free on OpenRouter)
# For better quality, use anthropic/claude-sonnet-4-5 (paid, ~$3–15/M tokens)
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL           = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
MAX_TOKENS          = 2048

# Backward-compat alias
CLAUDE_MODEL        = LLM_MODEL

# Keep backward-compat alias used in feedback.py EnvironmentError check
ANTHROPIC_API_KEY   = OPENROUTER_API_KEY

# ── LangSmith Configuration (monitoring & debugging) ────────────────────
# LangSmith integrates with LangChain/LangGraph via environment variables.
# Set LANGCHAIN_TRACING_V2=true to enable tracing to LangSmith dashboard.
LANGSMITH_API_KEY        = os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_TRACING_V2     = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT        = os.getenv("LANGCHAIN_PROJECT", "FraudRingDetectionGraph-RAG")
LANGSMITH_ENDPOINT       = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Validate LangSmith configuration if tracing is enabled
if LANGCHAIN_TRACING_V2 and not LANGSMITH_API_KEY:
    import logging as _logging
    _log = _logging.getLogger(__name__)
    _log.warning(
        "LANGCHAIN_TRACING_V2=true but LANGSMITH_API_KEY is not set. "
        "LangSmith tracing will not work. Set LANGSMITH_API_KEY in .env to enable."
    )

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
REASONING_SYSTEM_PROMPT = f"""You are an insurance fraud investigator. Analyze the claim and provide a brief investigation summary.

TASK: Analyze the subgraph data below. Look for fraud signals. Compare to historical rings if provided.

MANDATORY CHECKS (always evaluate these):
1. Is ring_suspicion_score >= 0.90? → OVR-001
2. Is total_claim_amount > $75,000? → OVR-002
3. Do lawyer/witness/shop appear across multiple states? → OVR-003
4. Is a licensed attorney the main connection? → OVR-005
5. Is claimant elderly (>65) or vulnerable? → OVR-006
6. Do any entities have prior SIU referrals? → OVR-008

REQUIRED OUTPUT FORMAT:
1. FRAUD SIGNALS DETECTED: List specific red flags with values (e.g., "staged_accident_flag=true", "shared_bank_flag=true")
2. RING PATTERN ANALYSIS: Describe connections between entities (customer, lawyer, shop, witnesses)
3. ANALOGOUS HISTORICAL CASES: If historical rings provided, compare patterns
4. MANDATORY OVERRIDE STATUS: List triggered OVR codes (e.g., "OVR-001, OVR-003")
5. RECOMMENDED ACTION: One sentence on next investigator step

Graph schema:
{GRAPH_SCHEMA}

Be specific. Use exact values from the data. Keep response under 400 words.
"""

NL_QUERY_SYSTEM_PROMPT = f"""You are a Cypher query writer for Neo4j fraud graph.
Write ONLY a valid Cypher query that answers the question. No explanation.

IMPORTANT:
1. Return only the query text, no markdown or explanation
2. LIMIT 25 unless user says otherwise
3. Use OPTIONAL MATCH for relationships that may not exist
4. Include property names in RETURN clause
5. For fraud queries, filter on: fraud_reported, ring_member_flag, fraud_flag, ring_score, priority

Graph nodes and relationships:
{GRAPH_SCHEMA}

Example formats:
- "MATCH (c:Claim) WHERE c.fraud_reported = true RETURN c.claim_id, c.final_suspicion_score LIMIT 10"
- "MATCH (r:FraudRing)-[:RING_CONTAINS_CLAIM]->(c:Claim) WHERE r.ring_score >= 0.9 RETURN r.ring_id, count(c) LIMIT 5"

Write the Cypher query now.
"""

RESULT_FORMATTER_SYSTEM_PROMPT = """Summarize Neo4j query results for a fraud investigator.

Rules:
1. Be factual — use exact values from results
2. Use bullet points for lists
3. Bold fraud-related fields (e.g., **fraud_flag**, **ring_score**)
4. Keep under 300 words
5. Highlight any high-risk patterns (ring_score >= 0.9, multiple claims, shared attributes)

Summarize the results now."""
