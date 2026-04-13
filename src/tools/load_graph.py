#!/usr/bin/env python3
"""
Neo4j Aura — Fraud Ring Detection Graph Loader
================================================
Loads all 24 node types and 23 edge types in the correct order.

Configuration (in priority order):
  1. CLI flags  — highest priority, override everything
  2. .env file  — set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_DIR, BATCH_SIZE
  3. Defaults   — user=neo4j, data-dir=./data, batch=200

Requirements:
    pip install neo4j pandas tqdm python-dotenv

Usage (zero-arg after filling .env):
    python load_graph.py

  Or override individual settings:
    python load_graph.py --uri "neo4j+s://xxxx.databases.neo4j.io" --password "pw"
    python load_graph.py --dry-run
"""

import os
import sys
import time
import argparse
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# ─── Load .env (silently skip if python-dotenv is not installed) ──
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env values must come from the shell environment instead

# ─── CLI args  (env vars are used as defaults so CLI always wins) ─
parser = argparse.ArgumentParser(
    description="Load Fraud Ring Detection graph into Neo4j Aura",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--uri",      default=os.getenv("NEO4J_URI"),      help="neo4j+s://xxxx.databases.neo4j.io  [env: NEO4J_URI]")
parser.add_argument("--user",     default=os.getenv("NEO4J_USER", "neo4j"), help="Database username  [env: NEO4J_USER]")
parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), help="Database password  [env: NEO4J_PASSWORD]")
parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data"), help="Directory containing CSV files  [env: DATA_DIR]")
parser.add_argument("--batch",    type=int, default=int(os.getenv("BATCH_SIZE", 200)), help="Rows per batch  [env: BATCH_SIZE]")
parser.add_argument("--trust-all-certs", action="store_true",
                    default=os.getenv("NEO4J_TRUST_ALL_CERTS", "true").lower() == "true",
                    help="Skip TLS cert verification (needed for Aura on Windows)  [env: NEO4J_TRUST_ALL_CERTS]")
parser.add_argument("--dry-run",  action="store_true", help="Validate files only, no DB writes")
args = parser.parse_args()

# ─── Validate required connection settings ────────────────────────
if not args.dry_run:
    missing = [name for name, val in [("--uri / NEO4J_URI", args.uri), ("--password / NEO4J_PASSWORD", args.password)] if not val]
    if missing:
        parser.error(f"Missing required config: {', '.join(missing)}\n"
                     "  Set them in .env or pass as CLI flags.")

DATA_DIR   = args.data_dir
BATCH_SIZE = args.batch

# ─── Connection ─────────────────────────────────────────────────
# URI scheme controls SSL mode — trusted_certificates param is NOT allowed
# alongside neo4j+s / bolt+s schemes (driver raises ConfigurationError).
#
#   neo4j+s://   → encrypted + verify cert   (default Aura)
#   neo4j+ssc:// → encrypted + skip verify   (needed when cert chain fails)
_SCHEME_UPGRADE = {"neo4j+s": "neo4j+ssc", "bolt+s": "bolt+ssc"}

def _resolve_uri(uri: str, trust_all: bool) -> str:
    if not trust_all:
        return uri
    for strict, relaxed in _SCHEME_UPGRADE.items():
        if uri.startswith(strict + "://"):
            return relaxed + uri[len(strict):]
    return uri  # already ssc or plain bolt/neo4j — leave as-is

def get_driver():
    uri = _resolve_uri(args.uri, args.trust_all_certs)
    driver = GraphDatabase.driver(uri, auth=(args.user, args.password))
    driver.verify_connectivity()
    print(f"\n✅ Connected to Neo4j Aura: {uri}\n")
    return driver

# ─── Helpers ─────────────────────────────────────────────────────
def safe_val(v):
    """Convert pandas NA/NaN to None so Neo4j receives null."""
    if pd.isna(v):
        return None
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v

def row_to_props(row, exclude=None):
    """Turn a DataFrame row into a clean property dict."""
    exclude = exclude or []
    return {k: safe_val(v) for k, v in row.items() if k not in exclude and not k.startswith(":")}

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def run_batches(session, query, rows, desc):
    total = 0
    for batch in tqdm(list(chunked(rows, BATCH_SIZE)), desc=desc, unit="batch"):
        session.run(query, rows=batch)
        total += len(batch)
    return total

# ─── CONSTRAINTS & INDEXES ───────────────────────────────────────
CONSTRAINTS = [
    # Uniqueness constraints (one per node type)
    ("Customer",            "cust_id"),
    ("Policy",              "policy_id"),
    ("Claim",               "claim_id"),
    ("Vehicle",             "vin_number"),
    ("Agent",               "agent_code"),
    ("Agency",              "agency_id"),
    ("Witness",             "statement_id"),
    ("PoliceOfficer",       "officer_id"),
    ("MedicalReport",       "report_id"),
    ("Doctor",              "doctor_id"),
    ("Hospital",            "hospital_id"),
    ("Lawyer",              "lawyer_id"),
    ("RepairShop",          "shop_id"),
    ("Contractor",          "contractor_id"),
    ("DrinkingVenue",       "venue_id"),
    ("FamilyUnit",          "family_id"),
    ("InvestigationCase",   "case_id"),
    ("HumanReview",         "review_id"),
    ("NetworkFeature",      "feature_id"),
    ("FinancialTransaction","transaction_id"),
    ("FraudRing",           "ring_id"),
    ("Event",               "event_id"),
    ("LiabilityPattern",    "pattern_id"),
    ("VehicleMake",         "make_id"),
]

INDEXES = [
    ("Claim",    "fraud_reported"),
    ("Claim",    "incident_state"),
    ("Claim",    "llm_judge_verdict"),
    ("Claim",    "adjuster_priority_tier"),
    ("Claim",    "ring_member_flag"),
    ("Claim",    "manual_override_flag"),
    ("Customer", "fraud_flag"),
    ("Customer", "risk_band"),
    ("Customer", "customer_tier"),
    ("Customer", "last_name"),
    ("FraudRing","status"),
    ("FraudRing","ring_score"),
    ("Witness",  "professional_witness_flag"),
    ("InvestigationCase", "status"),
    ("InvestigationCase", "priority"),
]

def create_schema(session):
    print("Creating constraints and indexes...")
    for label, prop in CONSTRAINTS:
        try:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
        except Exception as e:
            print(f"  ⚠️  Constraint {label}.{prop}: {e}")
    for label, prop in INDEXES:
        try:
            session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})")
        except Exception as e:
            print(f"  ⚠️  Index {label}.{prop}: {e}")
    print("  Schema ready.\n")

# ─── NODE LOADERS ────────────────────────────────────────────────
# Each entry: (filename, Neo4j_label, id_property_in_csv)
NODE_FILES = [
    ("nodes_Agency.csv",             "Agency",             ":ID"),
    ("nodes_Agent.csv",              "Agent",              ":ID"),
    ("nodes_LiabilityPattern.csv",   "LiabilityPattern",   ":ID"),
    ("nodes_VehicleMake.csv",        "VehicleMake",        ":ID"),
    ("nodes_FraudRing.csv",          "FraudRing",          ":ID"),
    ("nodes_FamilyUnit.csv",         "FamilyUnit",         ":ID"),
    ("nodes_Hospital.csv",           "Hospital",           ":ID"),
    ("nodes_Doctor.csv",             "Doctor",             ":ID"),
    ("nodes_Lawyer.csv",             "Lawyer",             ":ID"),
    ("nodes_RepairShop.csv",         "RepairShop",         ":ID"),
    ("nodes_Contractor.csv",         "Contractor",         ":ID"),
    ("nodes_DrinkingVenue.csv",      "DrinkingVenue",      ":ID"),
    ("nodes_Customer.csv",           "Customer",           ":ID"),
    ("nodes_Policy.csv",             "Policy",             ":ID"),
    ("nodes_Vehicle.csv",            "Vehicle",            ":ID"),
    ("nodes_Claim.csv",              "Claim",              ":ID"),
    ("nodes_PoliceOfficer.csv",      "PoliceOfficer",      ":ID"),
    ("nodes_Witness.csv",            "Witness",            ":ID"),
    ("nodes_MedicalReport.csv",      "MedicalReport",      ":ID"),
    ("nodes_InvestigationCase.csv",  "InvestigationCase",  ":ID"),
    ("nodes_HumanReview.csv",        "HumanReview",        ":ID"),
    ("nodes_NetworkFeature.csv",     "NetworkFeature",     ":ID"),
    ("nodes_FinancialTransaction.csv","FinancialTransaction",":ID"),
    ("nodes_Event.csv",              "Event",              ":ID"),
]

def load_nodes(session, label, filepath, id_col):
    df = pd.read_csv(filepath, low_memory=False)
    exclude = [":ID", ":LABEL"]
    rows = []
    for _, row in df.iterrows():
        props = row_to_props(row, exclude=exclude)
        props["_neo4j_id"] = str(row[id_col])   # always string
        rows.append(props)

    query = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{_neo4j_id: row._neo4j_id}})
    SET n += row
    """
    return run_batches(session, query, rows, f"  {label}")

# ─── EDGE LOADERS ────────────────────────────────────────────────
# Each entry: (filename, start_label, end_label, rel_type, start_id_prop, end_id_prop)
# start/end_id_prop = the property on the node that the CSV :START_ID/:END_ID maps to
EDGE_FILES = [
    # file                                   start_label         end_label           rel_type
    ("edges_CUSTOMER_TO_POLICY.csv",          "Customer",         "Policy",           "HAS_POLICY"),
    ("edges_POLICY_TO_CLAIM.csv",             "Policy",           "Claim",            "HAS_CLAIM"),
    ("edges_CUSTOMER_TO_CLAIM.csv",           "Customer",         "Claim",            "FILED_CLAIM"),
    ("edges_CUSTOMER_TO_AGENT.csv",           "Customer",         "Agent",            "MANAGED_BY"),
    ("edges_CLAIM_TO_VEHICLE.csv",            "Claim",            "Vehicle",          "INVOLVES_VEHICLE"),
    ("edges_CLAIM_TO_LAWYER.csv",             "Claim",            "Lawyer",           "REPRESENTED_BY"),
    ("edges_CLAIM_TO_REPAIRSHOP.csv",         "Claim",            "RepairShop",       "REPAIRED_AT"),
    ("edges_CLAIM_TO_WITNESS.csv",            "Claim",            "Witness",          "HAS_WITNESS"),
    ("edges_CLAIM_TO_MEDICALREPORT.csv",      "Claim",            "MedicalReport",    "HAS_MEDICAL_REPORT"),
    ("edges_CLAIM_TO_POLICEOFFICER.csv",      "Claim",            "PoliceOfficer",    "INVESTIGATED_BY"),
    ("edges_MEDICALREPORT_TO_DOCTOR.csv",     "MedicalReport",    "Doctor",           "TREATED_BY"),
    ("edges_MEDICALREPORT_TO_HOSPITAL.csv",   "MedicalReport",    "Hospital",         "ADMITTED_TO"),
    ("edges_CUSTOMER_RELATIONSHIPS.csv",      "Customer",         "Customer",         None),   # multi-type
    ("edges_CUSTOMER_TO_CUSTOMER_shared.csv", "Customer",         "Customer",         None),   # multi-type
    ("edges_SHARES_ATTRIBUTE.csv",            "Customer",         "Customer",         "SHARES_ATTRIBUTE"),
    ("edges_FRAUDRING_TO_CLAIM.csv",          "FraudRing",        "Claim",            "RING_CONTAINS_CLAIM"),
    ("edges_FRAUDRING_TO_CUSTOMER.csv",       "FraudRing",        "Customer",         "RING_INVOLVES_CUSTOMER"),
    ("edges_INVESTIGATIONCASE_TO_CLAIM.csv",  "InvestigationCase","Claim",            "INVESTIGATES_CLAIM"),
    ("edges_HUMANREVIEW_TO_CASE.csv",         "HumanReview",      "InvestigationCase","REVIEWS_CASE"),
    ("edges_NETWORKFEATURE_TO_ENTITY.csv",    "NetworkFeature",   None,               "DESCRIBES_ENTITY"),  # polymorphic
    ("edges_TRANSACTION_TO_CLAIM.csv",        "FinancialTransaction","Claim",          "PAYMENT_FOR_CLAIM"),
    ("edges_EVENT_TO_ENTITIES.csv",           "Event",            None,               None),   # polymorphic
]

def load_edges_fixed_type(session, filepath, start_label, end_label, rel_type):
    """Load edges where rel_type is fixed and both sides are known labels."""
    df = pd.read_csv(filepath, low_memory=False)
    rows = []
    for _, row in df.iterrows():
        props = row_to_props(row, exclude=[":START_ID",":END_ID",":TYPE"])
        rows.append({
            "start_id": str(row[":START_ID"]),
            "end_id":   str(row[":END_ID"]),
            "props":    props,
        })
    query = f"""
    UNWIND $rows AS row
    MATCH (a:{start_label} {{_neo4j_id: row.start_id}})
    MATCH (b:{end_label}   {{_neo4j_id: row.end_id}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r += row.props
    """
    return run_batches(session, query, rows, f"  {rel_type}")

def load_edges_typed_column(session, filepath, start_label, end_label=None):
    """Load edges where :TYPE is in a column (CUSTOMER_RELATIONSHIPS, shared attrs)."""
    df = pd.read_csv(filepath, low_memory=False)
    # Group by relationship type
    for rel_type, grp in df.groupby(":TYPE"):
        rows = []
        for _, row in grp.iterrows():
            props = row_to_props(row, exclude=[":START_ID",":END_ID",":TYPE"])
            rows.append({
                "start_id": str(row[":START_ID"]),
                "end_id":   str(row[":END_ID"]),
                "props":    props,
            })
        if end_label:
            query = f"""
            UNWIND $rows AS row
            MATCH (a:{start_label} {{_neo4j_id: row.start_id}})
            MATCH (b:{end_label}   {{_neo4j_id: row.end_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += row.props
            """
        else:
            query = f"""
            UNWIND $rows AS row
            MATCH (a:{start_label} {{_neo4j_id: row.start_id}})
            MATCH (b {{_neo4j_id: row.end_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += row.props
            """
        run_batches(session, query, rows, f"  {rel_type}")

def load_edges_polymorphic(session, filepath, start_label):
    """Load edges where end node can be any label (NetworkFeature, Event)."""
    df = pd.read_csv(filepath, low_memory=False)
    # Group by rel type
    type_col = ":TYPE" if ":TYPE" in df.columns else None
    rows = []
    for _, row in df.iterrows():
        props = row_to_props(row, exclude=[":START_ID",":END_ID",":TYPE"])
        rel_type = str(row[":TYPE"]) if type_col else "RELATED_TO"
        rows.append({
            "start_id": str(row[":START_ID"]),
            "end_id":   str(row[":END_ID"]),
            "rel_type": rel_type,
            "props":    props,
        })
    # Must use APOC or dynamic Cypher — fallback: group by type
    for rel_type, grp_rows in pd.DataFrame(rows).groupby("rel_type"):
        batch_rows = grp_rows.to_dict("records")
        query = f"""
        UNWIND $rows AS row
        MATCH (a:{start_label} {{_neo4j_id: row.start_id}})
        MATCH (b {{_neo4j_id: row.end_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += row.props
        """
        run_batches(session, query, batch_rows, f"  {rel_type}")

# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    if args.dry_run:
        print("\n🔍 DRY RUN — validating files only (no DB writes)\n")
        total_n = total_e = 0
        for fname, label, _ in NODE_FILES:
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                df = pd.read_csv(path, low_memory=False)
                total_n += len(df)
                print(f"  ✅ {fname}: {len(df)} rows")
            else:
                print(f"  ❌ MISSING: {fname}")
        for fname, *_ in EDGE_FILES:
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                df = pd.read_csv(path, low_memory=False)
                total_e += len(df)
                print(f"  ✅ {fname}: {len(df)} rows")
            else:
                print(f"  ❌ MISSING: {fname}")
        print(f"\nTotal nodes: {total_n:,}  (limit: 50,000)")
        print(f"Total edges:  {total_e:,}  (limit: 175,000)")
        return

    driver = get_driver()

    with driver.session() as session:
        # ── SCHEMA ──────────────────────────────────────────────
        create_schema(session)

        # ── NODES ───────────────────────────────────────────────
        print("=" * 55)
        print("LOADING NODES")
        print("=" * 55)
        total_nodes = 0
        for fname, label, id_col in NODE_FILES:
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                print(f"  ⚠️  Skipping missing file: {fname}")
                continue
            n = load_nodes(session, label, path, id_col)
            total_nodes += n
        print(f"\n  ✅ Nodes loaded: {total_nodes:,}\n")

        # ── EDGES ───────────────────────────────────────────────
        print("=" * 55)
        print("LOADING RELATIONSHIPS")
        print("=" * 55)

        for entry in EDGE_FILES:
            fname, start_label, end_label, rel_type = entry
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                print(f"  ⚠️  Skipping missing: {fname}")
                continue

            print(f"\n  → {fname}")

            # Multi-type column edges (CUSTOMER_RELATIONSHIPS, shared)
            if rel_type is None and end_label == "Customer":
                load_edges_typed_column(session, path, start_label, "Customer")
            # Polymorphic end node (NetworkFeature→anything, Event→anything)
            elif end_label is None:
                load_edges_polymorphic(session, path, start_label)
            # Standard fixed-type edges
            else:
                load_edges_fixed_type(session, path, start_label, end_label, rel_type)

        print("\n  ✅ Relationships loaded\n")

    driver.close()
    print("=" * 55)
    print("🎉 GRAPH LOAD COMPLETE")
    print("=" * 55)
    print("\nNext: Open Neo4j Aura console → Query tab and run:")
    print("  MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC")
    print("  MATCH ()-[r]->() RETURN type(r), count(r) ORDER BY count(r) DESC\n")

if __name__ == "__main__":
    main()
