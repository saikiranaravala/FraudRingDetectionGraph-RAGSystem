"""
Neo4j subgraph retrieval for the GraphRAG pipeline.

Extracts structured context around a Claim or FraudRing node:
  - The claim's direct neighbours (customer, lawyer, shop, witnesses, medical)
  - Ring membership and ring-level properties
  - NetworkFeature scores
  - InvestigationCase status
  - Shared attribute edges (CRITICAL/HIGH fraud signals)

Also provides:
  - get_all_fraud_rings()  — for initial vector store indexing
  - get_all_claims()       — for batch scoring / NL query seeding
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_driver():
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError("neo4j Python driver required: pip install neo4j") from exc

    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw   = os.getenv("NEO4J_PASSWORD")

    if not uri or not pw:
        raise EnvironmentError(
            "NEO4J_URI and NEO4J_PASSWORD must be set in .env"
        )

    trust_all = os.getenv("NEO4J_TRUST_ALL_CERTS", "true").lower() == "true"
    _UP = {"neo4j+s": "neo4j+ssc", "bolt+s": "bolt+ssc"}
    if trust_all:
        for s, r in _UP.items():
            if uri.startswith(s + "://"):
                uri = r + uri[len(s):]
                break

    driver = GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    return driver


def _node_to_dict(node) -> Dict[str, Any]:
    """Convert a Neo4j Node object to a plain dict."""
    if node is None:
        return {}
    return dict(node.items())


class GraphRetriever:
    """
    Retrieves subgraphs from Neo4j for GraphRAG context building.

    Create once and reuse — the driver connection is kept open.
    Call close() when done (or use as a context manager).
    """

    def __init__(self):
        self._driver = _get_driver()

    def close(self):
        if self._driver:
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Claim subgraph ────────────────────────────────────────────────
    def get_claim_subgraph(self, claim_id: str) -> Dict[str, Any]:
        """
        Return a structured dict with all entities connected to a Claim.

        Keys:
          claim, customer, lawyer, repair_shop, witnesses[],
          medical_report, fraud_ring, investigation_case,
          network_feature, shared_attributes[]
        """
        query = """
        MATCH (c:Claim {claim_id: $claim_id})
        OPTIONAL MATCH (cust:Customer)-[:FILED_CLAIM]->(c)
        OPTIONAL MATCH (c)-[:REPRESENTED_BY]->(l:Lawyer)
        OPTIONAL MATCH (c)-[:REPAIRED_AT]->(s:RepairShop)
        OPTIONAL MATCH (c)-[:HAS_MEDICAL_REPORT]->(m:MedicalReport)
        OPTIONAL MATCH (r:FraudRing)-[:RING_CONTAINS_CLAIM]->(c)
        OPTIONAL MATCH (ic:InvestigationCase)-[:INVESTIGATES_CLAIM]->(c)
        OPTIONAL MATCH (nf:NetworkFeature)-[:DESCRIBES_ENTITY]->(c)
        RETURN c, cust, l, s, m, r, ic, nf
        """
        witness_query = """
        MATCH (c:Claim {claim_id: $claim_id})-[:HAS_WITNESS]->(w:Witness)
        RETURN w ORDER BY w.reliability_score ASC LIMIT 10
        """
        shared_query = """
        MATCH (c:Claim {claim_id: $claim_id})
        MATCH (cust:Customer)-[:FILED_CLAIM]->(c)
        MATCH (cust)-[sa:SHARES_ATTRIBUTE]->(other:Customer)
        WHERE sa.fraud_signal IN ['CRITICAL', 'HIGH']
        RETURN sa.attribute_type AS attr_type,
               sa.fraud_signal   AS signal,
               other.full_name   AS other_name
        LIMIT 10
        """

        with self._driver.session() as session:
            row = session.run(query, claim_id=claim_id).single()
            if row is None:
                log.warning("Claim not found in Neo4j: %s", claim_id)
                return {}

            subgraph: Dict[str, Any] = {
                "claim":            _node_to_dict(row["c"]),
                "customer":         _node_to_dict(row["cust"]),
                "lawyer":           _node_to_dict(row["l"]),
                "repair_shop":      _node_to_dict(row["s"]),
                "medical_report":   _node_to_dict(row["m"]),
                "fraud_ring":       _node_to_dict(row["r"]),
                "investigation_case": _node_to_dict(row["ic"]),
                "network_feature":  _node_to_dict(row["nf"]),
            }

            # Witnesses
            witnesses = session.run(witness_query, claim_id=claim_id)
            subgraph["witnesses"] = [_node_to_dict(r["w"]) for r in witnesses]

            # Shared attributes (CRITICAL/HIGH only)
            sa_rows = session.run(shared_query, claim_id=claim_id)
            subgraph["shared_attributes"] = [
                {
                    "attr_type":  r["attr_type"],
                    "fraud_signal": r["signal"],
                    "other_name": r["other_name"],
                }
                for r in sa_rows
            ]

        return subgraph

    # ── Ring subgraph ─────────────────────────────────────────────────
    def get_ring_subgraph(self, ring_id: str) -> Dict[str, Any]:
        """Return ring properties plus all member customers and claims."""
        query = """
        MATCH (r:FraudRing {ring_id: $ring_id})
        OPTIONAL MATCH (r)-[:RING_INVOLVES_CUSTOMER]->(c:Customer)
        OPTIONAL MATCH (r)-[:RING_CONTAINS_CLAIM]->(cl:Claim)
        RETURN r,
               collect(DISTINCT {
                   cust_id: c.cust_id,
                   full_name: c.full_name,
                   fraud_flag: c.fraud_flag,
                   risk_band: c.risk_band
               }) AS members,
               collect(DISTINCT {
                   claim_id: cl.claim_id,
                   amount: cl.total_claim_amount,
                   state: cl.incident_state,
                   staged: cl.staged_accident_flag
               }) AS claims
        """
        with self._driver.session() as session:
            row = session.run(query, ring_id=ring_id).single()
            if row is None:
                return {}
            return {
                "ring":    _node_to_dict(row["r"]),
                "members": list(row["members"]),
                "claims":  list(row["claims"]),
            }

    # ── Bulk fetchers ─────────────────────────────────────────────────
    def get_all_fraud_rings(self) -> List[Dict[str, Any]]:
        """Return all FraudRing property dicts for vector store indexing."""
        with self._driver.session() as session:
            result = session.run("MATCH (r:FraudRing) RETURN r")
            return [_node_to_dict(row["r"]) for row in result]

    def get_all_claims(self, limit: int = 0) -> List[Dict[str, Any]]:
        """Return Claim property dicts (all or limited)."""
        q = "MATCH (c:Claim) RETURN c" + (f" LIMIT {limit}" if limit else "")
        with self._driver.session() as session:
            return [_node_to_dict(r["c"]) for r in session.run(q)]

    def get_claim_properties(self, claim_id: str) -> Dict[str, Any]:
        """Return a single Claim's raw property dict."""
        with self._driver.session() as session:
            row = session.run(
                "MATCH (c:Claim {claim_id: $cid}) RETURN c", cid=claim_id
            ).single()
            return _node_to_dict(row["c"]) if row else {}

    def get_high_priority_claims(self, threshold: float = 0.70) -> List[Dict[str, Any]]:
        """Return claims with final_suspicion_score >= threshold, sorted descending."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Claim)
                WHERE c.final_suspicion_score >= $threshold
                RETURN c ORDER BY c.final_suspicion_score DESC
                """,
                threshold=threshold,
            )
            return [_node_to_dict(r["c"]) for r in result]

    def run_cypher(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute arbitrary Cypher and return results as list of dicts.
        Used by the NL query engine.
        """
        with self._driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

    # ── HumanReview write-back ─────────────────────────────────────────
    def record_human_review(
        self,
        claim_id: str,
        decision: str,
        investigator_id: str,
        feedback_to_model: str,
        override_reason: str = "",
        confidence: float = 1.0,
    ) -> bool:
        """
        Write an investigator decision to the HumanReview node linked
        to the InvestigationCase for this claim.

        decision          : "Approve" | "Dismiss" | "Escalate"
        feedback_to_model : "Correct" | "FP" | "FN" | "Uncertain"
        """
        query = """
        MATCH (ic:InvestigationCase {linked_claim_id: $claim_id})
        MERGE (hr:HumanReview {review_id: 'REV-' + $claim_id + '-' + $investigator_id})
        SET hr.decision                 = $decision,
            hr.investigator_id          = $investigator_id,
            hr.feedback_to_model        = $feedback_to_model,
            hr.override_reason_code     = $override_reason,
            hr.decision_confidence      = $confidence,
            hr.reviewed_at              = datetime(),
            hr.override_ai_recommendation = ($decision = 'Dismiss' AND
                                              ic.ai_fraud_score > 0.5),
            hr.disagreement_flag        = ($decision = 'Dismiss' AND
                                           ic.ai_fraud_score > 0.7)
        MERGE (hr)-[:REVIEWS_CASE]->(ic)
        RETURN hr.review_id AS review_id
        """
        try:
            with self._driver.session() as session:
                row = session.run(
                    query,
                    claim_id=claim_id,
                    decision=decision,
                    investigator_id=investigator_id,
                    feedback_to_model=feedback_to_model,
                    override_reason=override_reason,
                    confidence=confidence,
                ).single()
                log.info("Review recorded: %s", row["review_id"] if row else "?")
                return True
        except Exception as e:
            log.error("Failed to record review for %s: %s", claim_id, e)
            return False

    def get_feedback_since(self, since_iso: str = "1970-01-01") -> List[Dict]:
        """Retrieve HumanReview decisions since a given ISO date for retraining."""
        query = """
        MATCH (hr:HumanReview)-[:REVIEWS_CASE]->(ic:InvestigationCase)
        WHERE hr.reviewed_at >= datetime($since)
        RETURN ic.linked_claim_id   AS claim_id,
               hr.decision          AS decision,
               hr.feedback_to_model AS feedback,
               ic.ai_fraud_score    AS ai_score,
               hr.reviewed_at       AS reviewed_at
        ORDER BY hr.reviewed_at DESC
        """
        with self._driver.session() as session:
            return [dict(r) for r in session.run(query, since=since_iso)]
