"""
Text embedding for fraud ring and claim context.

Converts structured Neo4j property dicts into descriptive text strings
then embeds them using fastembed (ONNX-based, no PyTorch required).

fastembed uses ONNX Runtime instead of PyTorch — ~100 MB RAM vs ~380 MB
for sentence-transformers+torch. Required for Render free plan (512 MB).

Default model: BAAI/bge-small-en-v1.5 (384-dim, same as paraphrase-MiniLM-L3-v2)

Used to populate the vector knowledge base (FraudRing nodes) and to
embed a new ring/claim at query time for analogous-ring retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from fastembed import TextEmbedding
    HAS_FE = True
except ImportError:
    HAS_FE = False
    log.warning(
        "fastembed not installed. "
        "Install with: pip install fastembed"
    )

from src.utils import config as C


# ── Text serialisers ──────────────────────────────────────────────────
def ring_to_text(props: Dict[str, Any]) -> str:
    """
    Convert a FraudRing property dict to a descriptive text string
    suitable for embedding.
    """
    def yn(v) -> str:
        if v is None:
            return "unknown"
        if isinstance(v, bool):
            return "yes" if v else "no"
        s = str(v).strip().lower()
        return "yes" if s in ("true", "1", "y", "yes") else "no"

    amount = props.get("total_claim_amount", 0)
    try:
        amount = float(amount)
        amount_str = f"${amount:,.0f}"
    except (TypeError, ValueError):
        amount_str = str(amount)

    return (
        f"Fraud ring {props.get('ring_id', 'unknown')} — "
        f"status: {props.get('status', 'unknown')}, "
        f"ring score: {props.get('ring_score', 0)}, "
        f"members: {props.get('member_count', 0)}, "
        f"claims: {props.get('claim_count', 0)}, "
        f"total exposure: {amount_str}, "
        f"detection method: {props.get('detection_method', 'unknown')}, "
        f"states: {props.get('incident_states', 'unknown')}, "
        f"incident types: {props.get('incident_types', 'unknown')}, "
        f"closed loop: {yn(props.get('closed_loop_detected'))}, "
        f"shared lawyer: {yn(props.get('shared_lawyer_flag'))}, "
        f"shared shop: {yn(props.get('shared_shop_flag'))}, "
        f"shared witness: {yn(props.get('shared_witness_flag'))}, "
        f"law enforcement notified: {yn(props.get('law_enforcement_notified'))}, "
        f"nicb case filed: {yn(props.get('nicb_case_filed'))}"
    )


def claim_to_text(props: Dict[str, Any]) -> str:
    """
    Convert a Claim property dict to text for embedding / context.
    """
    def yn(v) -> str:
        if v is None:
            return "unknown"
        s = str(v).strip().lower()
        return "yes" if s in ("true", "1", "y", "yes") else "no"

    amount = props.get("total_claim_amount", 0)
    try:
        amount_str = f"${float(amount):,.0f}"
    except (TypeError, ValueError):
        amount_str = str(amount)

    return (
        f"Insurance claim {props.get('claim_id', 'unknown')}: "
        f"amount {amount_str}, "
        f"type: {props.get('incident_type', 'unknown')}, "
        f"state: {props.get('incident_state', 'unknown')}, "
        f"fraud reported: {yn(props.get('fraud_reported'))}, "
        f"ring member: {yn(props.get('ring_member_flag'))}, "
        f"staged accident: {yn(props.get('staged_accident_flag'))}, "
        f"closed loop: {yn(props.get('closed_loop_flag'))}, "
        f"manual override: {yn(props.get('manual_override_flag'))}, "
        f"gnn score: {props.get('gnn_suspicion_score', props.get('rag_confidence_score', 0)):.3f}"
    )


def subgraph_to_text(subgraph: Dict[str, Any]) -> str:
    """
    Convert a claim subgraph dict (from graph_retriever) to text for
    inclusion in the Claude reasoning prompt.
    """
    parts: list[str] = []

    claim = subgraph.get("claim", {})
    if claim:
        parts.append(f"CLAIM: {claim_to_text(claim)}")

    customer = subgraph.get("customer", {})
    if customer:
        parts.append(
            f"CUSTOMER: {customer.get('full_name', 'unknown')}, "
            f"risk={customer.get('risk_band', '?')}, "
            f"fraud_history={customer.get('fraud_history_count', 0)}, "
            f"shared_bank={customer.get('shared_bank_flag', 'N')}, "
            f"synthetic_identity={customer.get('synthetic_identity_flag', 'N')}, "
            f"role_switching={customer.get('role_switching_flag', 'N')}, "
            f"siu_referral={customer.get('siu_referral_flag', 'N')}"
        )

    lawyer = subgraph.get("lawyer", {})
    if lawyer:
        parts.append(
            f"LAWYER: {lawyer.get('full_name', 'unknown')} "
            f"({lawyer.get('firm_name', '?')}), "
            f"known_to_siu={lawyer.get('known_to_siu', 'N')}, "
            f"shared_clients={lawyer.get('shared_clients_count', 0)}, "
            f"closed_loop_network={lawyer.get('closed_loop_network_flag', 'N')}, "
            f"bar_discipline_score={lawyer.get('bar_discipline_score', 0)}"
        )

    shop = subgraph.get("repair_shop", {})
    if shop:
        parts.append(
            f"REPAIR SHOP: {shop.get('name', 'unknown')}, "
            f"fraud_flag={shop.get('fraud_flag', 'N')}, "
            f"estimate_variance={shop.get('estimate_variance_pct', 0):.1f}%, "
            f"same_lawyer_referrals={shop.get('same_lawyer_referrals', 0)}, "
            f"siu_referrals={shop.get('siu_referral_count', 0)}"
        )

    witnesses = subgraph.get("witnesses", [])
    pro_witnesses = [w for w in witnesses if str(w.get("professional_witness_flag", "")).lower() in ("true", "1", "y", "yes")]
    if witnesses:
        parts.append(
            f"WITNESSES: {len(witnesses)} total, "
            f"{len(pro_witnesses)} professional witnesses, "
            f"coached statements: {sum(1 for w in witnesses if str(w.get('coached_statement_flag','')).lower() in ('true','1','y','yes'))}"
        )
        for w in witnesses[:C.MAX_WITNESSES_IN_CONTEXT]:
            parts.append(
                f"  - {w.get('full_name', 'unknown')}: "
                f"reliability={w.get('reliability_score', 0):.2f}, "
                f"appears_in={w.get('cross_claim_appearance_count', w.get('same_name_claims_count', 0))} claims"
            )

    medical = subgraph.get("medical_report", {})
    if medical:
        flags = []
        for f in ("upcoding_flag", "unbundling_flag", "phantom_billing_flag"):
            if str(medical.get(f, "")).lower() in ("true", "1", "y", "yes"):
                flags.append(f.replace("_flag", ""))
        parts.append(
            f"MEDICAL REPORT: billed={medical.get('billed_amount', 0):.0f}, "
            f"paid={medical.get('insurance_paid_amount', 0):.0f}, "
            f"billing_score={medical.get('billing_pattern_score', 0):.2f}, "
            f"fraud flags: {', '.join(flags) if flags else 'none'}"
        )

    ring = subgraph.get("fraud_ring", {})
    if ring:
        parts.append(
            f"RING MEMBERSHIP: {ring.get('ring_id', '?')} "
            f"({ring.get('ring_name', '?')}), "
            f"status={ring.get('status', '?')}, "
            f"score={ring.get('ring_score', 0)}, "
            f"members={ring.get('member_count', 0)}"
        )

    network = subgraph.get("network_feature", {})
    if network:
        parts.append(
            f"NETWORK: pagerank={network.get('pagerank_score', 0):.3f}, "
            f"ring_suspicion={network.get('ring_suspicion_score', 0):.3f}, "
            f"neighbor_fraud_rate={network.get('neighbor_fraud_rate', 0):.3f}, "
            f"hop2_fraud_count={network.get('hop_2_fraud_count', 0)}"
        )

    inv_case = subgraph.get("investigation_case", {})
    if inv_case:
        parts.append(
            f"INVESTIGATION: case={inv_case.get('case_id', '?')}, "
            f"status={inv_case.get('status', '?')}, "
            f"ai_score={inv_case.get('ai_fraud_score', 0):.3f}, "
            f"override_triggered={inv_case.get('manual_override_triggered', 'N')}"
        )

    shared_attrs = subgraph.get("shared_attributes", [])
    if shared_attrs:
        crit = [a for a in shared_attrs if a.get("fraud_signal") == "CRITICAL"]
        high = [a for a in shared_attrs if a.get("fraud_signal") == "HIGH"]
        parts.append(
            f"SHARED ATTRIBUTES: {len(shared_attrs)} total — "
            f"CRITICAL: {len(crit)} (shared bank/account), "
            f"HIGH: {len(high)} (shared IP/phone)"
        )

    return "\n".join(parts) if parts else "No subgraph context available."


# ── Embedder class ────────────────────────────────────────────────────
class FraudEmbedder:
    """
    Embeds fraud ring and claim text using fastembed (ONNX Runtime).

    fastembed avoids PyTorch entirely — uses onnxruntime instead,
    cutting RAM from ~380 MB to ~100 MB. Compatible with Render free plan.

    Default model: BAAI/bge-small-en-v1.5 (384-dim)
    Falls back to a deterministic hash-based pseudo-embedding if fastembed
    is not installed (useful for dry-run / testing).
    """

    def __init__(self, model_name: str = C.EMBEDDING_MODEL):
        self._model: Optional[Any] = None
        self._model_name = model_name
        self._dim = C.EMBEDDING_DIM

        if HAS_FE:
            try:
                log.info("Loading fastembed model: %s", model_name)
                self._model = TextEmbedding(model_name=model_name)
                log.info("fastembed model loaded (ONNX, no PyTorch required)")
            except Exception as e:
                log.warning("Could not load fastembed model: %s. Using fallback.", e)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string → float32 array of shape (dim,)."""
        if self._model is not None:
            vecs = list(self._model.embed([text]))
            return np.array(vecs[0], dtype=np.float32)
        return self._fallback_embed(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts → float32 array of shape (N, dim)."""
        if self._model is not None:
            vecs = list(self._model.embed(texts))
            return np.array(vecs, dtype=np.float32)
        return np.vstack([self._fallback_embed(t) for t in texts])

    def embed_ring(self, ring_props: Dict[str, Any]) -> np.ndarray:
        return self.embed(ring_to_text(ring_props))

    def embed_claim(self, claim_props: Dict[str, Any]) -> np.ndarray:
        return self.embed(claim_to_text(claim_props))

    def embed_subgraph(self, subgraph: Dict[str, Any]) -> np.ndarray:
        return self.embed(subgraph_to_text(subgraph))

    def _fallback_embed(self, text: str) -> np.ndarray:
        """
        Deterministic pseudo-embedding when sentence-transformers is absent.
        Uses a character n-gram hash projection — consistent but not semantic.
        Only for development / CI testing.
        """
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
