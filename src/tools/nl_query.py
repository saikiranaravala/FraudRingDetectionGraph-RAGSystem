"""
Natural Language Query Engine

Investigators ask questions in plain English about the fraud graph.
Claude converts the question to Cypher, executes it against Neo4j,
then Claude formats the raw results into a human-readable answer.

Two-step pipeline (each step is a separate Claude call):
  1. NL → Cypher  (system prompt cached)
  2. results → NL summary  (system prompt cached)

Safety
──────
  - Cypher is executed read-only (no MERGE/CREATE/DELETE allowed)
  - Claude-generated Cypher is validated before execution
  - Results are truncated at MAX_RESULT_ROWS to prevent prompt bloat

Usage
─────
    engine = NLQueryEngine()
    answer = engine.query("Which fraud rings have a lawyer appearing in 3+ claims?")
    print(answer)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.utils import config as C
from src.services.graph_retriever import GraphRetriever

log = logging.getLogger(__name__)

MAX_RESULT_ROWS = 25
MAX_RESULT_CHARS = 4000   # cap raw results before sending to Claude

# Write operations that must never appear in generated Cypher
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(MERGE|CREATE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+apoc\.periodic)\b",
    re.IGNORECASE,
)

# Few-shot examples injected into the Cypher-generation prompt
# Updated to match actual data patterns in the fraud graph
FEW_SHOT_EXAMPLES = """
Examples:
Q: Show me fraud rings with the highest scores
A: MATCH (r:FraudRing)-[:RING_CONTAINS_CLAIM]->(c:Claim) RETURN r.ring_id, r.ring_name, r.ring_score, count(c) AS claim_count ORDER BY r.ring_score DESC LIMIT 25

Q: Which lawyers appear in the most claims?
A: MATCH (l:Lawyer)<-[:REPRESENTED_BY]-(c:Claim) RETURN l.full_name, l.known_to_siu, count(c) AS claim_count ORDER BY claim_count DESC LIMIT 25

Q: Find customers who share bank accounts
A: MATCH (c1:Customer)-[s:SHARES_ATTRIBUTE]->(c2:Customer) WHERE s.attribute_type = 'bank_account' AND s.fraud_signal = 'CRITICAL' RETURN c1.full_name, c2.full_name, s.fraud_signal LIMIT 25

Q: Which repair shops appear in the most fraud ring claims?
A: MATCH (s:RepairShop)<-[:REPAIRED_AT]-(c:Claim)<-[:RING_CONTAINS_CLAIM]-(r:FraudRing) WITH s, count(DISTINCT r) AS ring_count, count(c) AS claim_count RETURN s.name, s.fraud_flag, ring_count, claim_count ORDER BY ring_count DESC LIMIT 25

Q: Show me high-risk claims that need investigation
A: MATCH (c:Claim) WHERE c.final_suspicion_score >= 0.70 RETURN c.claim_id, c.final_suspicion_score, c.incident_state, c.total_claim_amount ORDER BY c.final_suspicion_score DESC LIMIT 25

Q: List all claims in fraud ring RING-002
A: MATCH (r:FraudRing {ring_id: 'RING-002'})-[:RING_CONTAINS_CLAIM]->(c:Claim) RETURN c.claim_id, c.total_claim_amount, c.incident_state, c.final_suspicion_score LIMIT 25
"""


class NLQueryEngine:
    """
    Natural language query engine for the fraud graph.

    Requires OPENROUTER_API_KEY in .env.
    Routes requests through OpenRouter's OpenAI-compatible API.
    """

    def __init__(
        self,
        retriever:  Optional[GraphRetriever] = None,
        llm_client: Optional[OpenAI]         = None,
    ):
        if not C.OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY must be set in .env")

        self._retriever = retriever or GraphRetriever()
        self._llm       = llm_client or OpenAI(
            api_key=C.OPENROUTER_API_KEY,
            base_url=C.OPENROUTER_BASE_URL,
        )

    # ── Main entry point ──────────────────────────────────────────────
    def query(self, question: str, verbose: bool = False) -> str:
        """
        Convert a natural language question to Cypher, execute, and summarise.

        Returns a formatted natural language answer.
        """
        # Step 1: NL → Cypher
        cypher = self._generate_cypher(question)
        if verbose:
            print(f"\n  Generated Cypher:\n  {cypher}\n")

        # Step 2: Execute
        try:
            rows = self._execute_safe(cypher)
        except Exception as e:
            log.warning("Cypher execution failed: %s", e)
            return (
                f"I generated this Cypher query but it failed to execute:\n"
                f"```\n{cypher}\n```\n"
                f"Error: {e}\n\n"
                "Please rephrase your question or check the graph schema."
            )

        if not rows:
            return f"No results found for: '{question}'"

        # Step 3: Format results
        return self._format_results(question, cypher, rows)

    # ── Interactive REPL ──────────────────────────────────────────────
    def interactive(self):
        """Start an interactive query REPL. Type 'exit' to quit."""
        print("\nFraud Graph Query Engine")
        print("Type your question in plain English. Type 'exit' to quit.\n")
        while True:
            try:
                question = input("investigator> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                break
            answer = self.query(question, verbose=True)
            print(f"\n{answer}\n")

    # ── Step 1: Cypher generation ─────────────────────────────────────
    def _generate_cypher(self, question: str) -> str:
        user_msg = f"{FEW_SHOT_EXAMPLES}\n\nQ: {question}\nA:"

        response = self._llm.chat.completions.create(
            model=C.CLAUDE_MODEL,
            max_tokens=512,
            messages=[
                {"role": "system", "content": C.NL_QUERY_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        cypher = response.choices[0].message.content.strip()

        # Strip markdown fences if Claude included them
        cypher = re.sub(r"^```(?:cypher)?\s*", "", cypher, flags=re.MULTILINE)
        cypher = re.sub(r"```\s*$", "", cypher, flags=re.MULTILINE).strip()

        return cypher

    # ── Step 2: Safe execution ────────────────────────────────────────
    def _execute_safe(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute Cypher after validating it contains no write operations."""
        if FORBIDDEN_KEYWORDS.search(cypher):
            raise ValueError(
                f"Generated Cypher contains forbidden write operation. "
                f"Query: {cypher[:200]}"
            )
        return self._retriever.run_cypher(cypher)[:MAX_RESULT_ROWS]

    # ── Step 3: Result formatting ─────────────────────────────────────
    def _format_results(
        self, question: str, cypher: str, rows: List[Dict]
    ) -> str:
        # Serialise rows to text, capped at MAX_RESULT_CHARS
        rows_text = self._rows_to_text(rows)

        user_msg = (
            f"Question: {question}\n\n"
            f"Cypher executed:\n{cypher}\n\n"
            f"Results ({len(rows)} rows):\n{rows_text}\n\n"
            "Summarise these results for the fraud investigator."
        )

        response = self._llm.chat.completions.create(
            model=C.CLAUDE_MODEL,
            max_tokens=C.MAX_TOKENS,
            messages=[
                {"role": "system", "content": C.RESULT_FORMATTER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _rows_to_text(rows: List[Dict]) -> str:
        if not rows:
            return "(empty)"
        lines = []
        for i, row in enumerate(rows, 1):
            parts = []
            for k, v in row.items():
                if v is not None:
                    parts.append(f"{k}={v}")
            lines.append(f"  {i}. {' | '.join(parts)}")
            if sum(len(l) for l in lines) > MAX_RESULT_CHARS:
                lines.append(f"  … (truncated, {len(rows)} total rows)")
                break
        return "\n".join(lines)
