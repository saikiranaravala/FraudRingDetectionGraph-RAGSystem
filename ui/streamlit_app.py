"""
Fraud Ring Detection — Streamlit Frontend

Connects to the FastAPI backend (api.py) deployed on Render.com.

Configure the API URL via:
  - Streamlit Cloud: .streamlit/secrets.toml  →  api_url = "https://..."
  - Local dev: sidebar input field
"""

from __future__ import annotations

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Ring Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _api(method: str, path: str, timeout: int = 60, **kwargs):
    """Call the backend API, return (data_dict | None, error_str | None)."""
    url = st.session_state.api_url.rstrip("/") + path
    try:
        resp = getattr(requests, method)(url, timeout=timeout, **kwargs)
        if resp.status_code == 200:
            return resp.json(), None
        detail = resp.json().get("detail", resp.text) if resp.content else resp.text
        return None, f"HTTP {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API. Check the URL and ensure the service is running."
    except requests.exceptions.Timeout:
        return None, (
            f"Request timed out ({timeout} s). "
            "The Render free plan may be cold-starting — try again. "
            "NL queries make two LLM calls and may need up to 2 minutes."
        )
    except Exception as exc:
        return None, str(exc)


def _score_color(score: float) -> str:
    if score >= 0.90:
        return "#d62728"   # red
    if score >= 0.70:
        return "#ff7f0e"   # orange
    return "#2ca02c"       # green


def _score_tier(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    if score >= 0.70:
        return "HIGH PRIORITY"
    return "STANDARD"


# ── Session state defaults ────────────────────────────────────────────────────
if "api_url" not in st.session_state:
    try:
        st.session_state.api_url = st.secrets["api_url"]
    except Exception:
        st.session_state.api_url = "http://localhost:8000"

if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Fraud Ring Detection")
    st.caption("GraphRAG Investigation System")
    st.divider()

    st.subheader("API Connection")
    new_url = st.text_input(
        "Backend URL",
        value=st.session_state.api_url,
        placeholder="https://your-service.onrender.com",
        help="FastAPI backend deployed on Render.com",
    )
    if new_url != st.session_state.api_url:
        st.session_state.api_url = new_url
        st.session_state.api_healthy = None

    if st.button("Check Connection", use_container_width=True):
        data, err = _api("get", "/health")
        st.session_state.api_healthy = err is None
        if err:
            st.error(err)

    if st.session_state.api_healthy is True:
        st.success("API online")
    elif st.session_state.api_healthy is False:
        st.error("API unreachable")
    else:
        st.info("Not checked yet")

    st.divider()
    st.caption(
        "**Render free plan:** services spin down after 15 min of inactivity. "
        "The first request may take ~30 s to cold-start."
    )
    st.divider()
    st.markdown(
        "**Stack:** Neo4j · GraphSAGE · LangGraph · Claude · Pinecone\n\n"
        "**Phases:** Graph Load · GNN Scoring · GraphRAG · HITL Feedback"
    )


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_invest, tab_query, tab_feedback, tab_stats = st.tabs([
    "Investigate Claim",
    "Graph Query",
    "Record Feedback",
    "Stats & History",
])


# ── Tab 1: Investigate ────────────────────────────────────────────────────────
with tab_invest:
    st.header("Claim Investigation")
    st.markdown(
        "Run the full GraphRAG pipeline for a claim ID. "
        "Retrieves the Neo4j subgraph, finds analogous fraud rings in Pinecone, "
        "and generates a Claude investigation brief."
    )

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        claim_id = st.text_input(
            "Claim ID",
            placeholder="CLM-521585",
            label_visibility="collapsed",
        )
    with col_btn:
        run_btn = st.button("Investigate", type="primary", use_container_width=True)

    if run_btn:
        if not claim_id.strip():
            st.warning("Enter a claim ID.")
        else:
            with st.spinner(f"Running GraphRAG pipeline for {claim_id} — this may take 60–90 s (Neo4j + Pinecone + Claude) ..."):
                data, err = _api("post", f"/explain/{claim_id.strip()}", timeout=120)

            if err:
                st.error(err)
            else:
                # ── Score banner ──────────────────────────────────────────
                score = data.get("fraud_score", 0.0)
                tier  = _score_tier(score)
                color = _score_color(score)

                st.markdown(
                    f"""
                    <div style="
                        background:{color}22;
                        border-left: 6px solid {color};
                        border-radius: 6px;
                        padding: 12px 20px;
                        margin-bottom: 16px;
                    ">
                        <span style="font-size:1.6rem; font-weight:700; color:{color};">
                            {score:.4f}
                        </span>
                        &nbsp;&nbsp;
                        <span style="font-size:1rem; color:{color}; font-weight:600;">
                            {tier}
                        </span>
                        <span style="float:right; font-size:0.9rem; color:#888;">
                            Claim: {data['claim_id']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Override triggers ─────────────────────────────────────
                overrides = data.get("override_triggers", [])
                if overrides:
                    with st.expander(f"Override Triggers ({len(overrides)}) — PAYMENT HOLD", expanded=True):
                        for t in overrides:
                            st.markdown(f"- **{t}**")

                # ── Analogous rings ───────────────────────────────────────
                analogous = data.get("analogous_rings", [])
                if analogous:
                    with st.expander(f"Analogous Fraud Rings (top {len(analogous)})", expanded=True):
                        cols = st.columns(len(analogous))
                        for i, match in enumerate(analogous):
                            meta = match.get("metadata", {})
                            sim  = match.get("score", 0)
                            with cols[i]:
                                st.metric(
                                    label=match.get("id", "—"),
                                    value=f"{sim:.3f}",
                                    help="Cosine similarity",
                                )
                                st.caption(
                                    f"Status: **{meta.get('status', '?')}**  \n"
                                    f"Ring score: `{meta.get('ring_score', '?')}`  \n"
                                    f"Members: `{meta.get('member_count', '?')}`  \n"
                                    f"Claims: `{meta.get('total_claim_amount', '?')}`"
                                )

                # ── Investigation brief ───────────────────────────────────
                st.subheader("Investigation Brief")
                st.markdown(data.get("reasoning_trace", "*(no trace generated)*"))


# ── Tab 2: Graph Query ────────────────────────────────────────────────────────
with tab_query:
    st.header("Natural Language Graph Query")
    st.markdown(
        "Ask questions in plain English. Claude converts them to Cypher, "
        "executes them read-only against Neo4j, and summarises the results."
    )

    example_questions = [
        "Show fraud rings with the highest scores",
        "Which lawyers appear in the most claims?",
        "Find customers who share bank accounts",
        "Which repair shops appear in the most fraud ring claims?",
        "Show me high-risk claims that need investigation",
    ]

    with st.expander("Example questions"):
        for q in example_questions:
            if st.button(q, key=f"eg_{q[:20]}"):
                st.session_state.nl_question = q

    question = st.text_area(
        "Your question",
        value=st.session_state.get("nl_question", ""),
        height=80,
        placeholder="Which fraud rings have a lawyer in 3 or more claims?",
    )

    if st.button("Submit Query", type="primary"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Generating Cypher and querying Neo4j — two LLM calls, may take up to 2 min ..."):
                data, err = _api("post", "/query", json={"question": question.strip()}, timeout=120)

            if err:
                st.error(err)
            else:
                st.markdown("---")
                st.markdown(data.get("answer", "*(no answer)*"))


# ── Tab 3: Record Feedback ────────────────────────────────────────────────────
with tab_feedback:
    st.header("Record Investigator Decision")
    st.markdown(
        "Record a decision for a claim. This writes a `HumanReview` node to Neo4j "
        "and accumulates labels for feedback-loop retraining."
    )

    with st.form("feedback_form"):
        col1, col2 = st.columns(2)
        with col1:
            fb_claim_id = st.text_input("Claim ID", placeholder="CLM-521585")
            fb_investigator = st.text_input("Investigator ID", placeholder="INV-001")
            fb_decision = st.selectbox(
                "Decision",
                ["Approve", "Dismiss", "Escalate"],
                help="Approve / Escalate = fraud confirmed (label 1). Dismiss = false positive (label 0).",
            )
        with col2:
            fb_feedback = st.selectbox(
                "Feedback to model",
                ["Correct", "FP", "FN", "Uncertain"],
                help="Correct = model was right. FP = false positive. FN = false negative.",
            )
            fb_confidence = st.slider(
                "Confidence", min_value=0.0, max_value=1.0, value=1.0, step=0.05
            )
            fb_override_reason = st.text_input(
                "Override reason (optional)",
                placeholder="e.g. Additional SIU evidence",
            )

        submitted = st.form_submit_button("Submit Feedback", type="primary", use_container_width=True)

    if submitted:
        if not fb_claim_id.strip() or not fb_investigator.strip():
            st.warning("Claim ID and Investigator ID are required.")
        else:
            payload = {
                "decision": fb_decision,
                "investigator_id": fb_investigator.strip(),
                "feedback_to_model": fb_feedback,
                "override_reason": fb_override_reason.strip(),
                "confidence": fb_confidence,
            }
            with st.spinner("Recording decision ..."):
                data, err = _api("post", f"/feedback/{fb_claim_id.strip()}", json=payload)

            if err:
                st.error(err)
            else:
                st.success(
                    f"Decision recorded for **{data['claim_id']}**: "
                    f"**{data['decision']}**"
                )


# ── Tab 4: Stats ──────────────────────────────────────────────────────────────
with tab_stats:
    st.header("Stats & Feedback History")

    if st.button("Refresh", use_container_width=False):
        with st.spinner("Loading stats ..."):
            data, err = _api("get", "/stats")

        if err:
            st.error(err)
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews", data.get("total_reviews", 0))
            col2.metric("Vector Store Entries", data.get("vector_store_size", 0))
            col3.metric(
                "Last Review",
                data.get("last_review") or "never",
            )

            st.caption(f"Last retrain: {data.get('last_retrain') or 'never'}")

            f1_history = data.get("f1_history", [])
            if f1_history:
                st.subheader("F1 History")
                import pandas as pd
                df = pd.DataFrame(f1_history)
                st.dataframe(df, use_container_width=True)
            else:
                st.info(
                    "No retraining history yet. "
                    "Run `python src/main.py rag retrain` after accumulating >= 20 reviews."
                )
    else:
        st.info("Click **Refresh** to load current stats from the API.")
