# Fraud Ring Patterns

Graph patterns that indicate organized insurance fraud.

## Pattern 1 — Closed-Loop Ring (Highest Severity)

All fraud actors are interconnected: shared lawyer, shared repair shop, shared witnesses, shared bank accounts.

```mermaid
flowchart LR
    FR["FraudRing\nRING-007\nring_score: 0.97\nstatus: Active\nclosed_loop_detected: true"]

    subgraph CUSTOMERS["Recruited Customers"]
        C1["Customer A\nsynthetic_identity_flag: true"]
        C2["Customer B\nrole_switching_flag: true"]
        C3["Customer C\nshared_bank_flag: true"]
    end

    subgraph CLAIMS["Staged Claims"]
        CL1["Claim 1\nstaged_accident_flag: true\namount: $28,400"]
        CL2["Claim 2\nstaged_accident_flag: true\namount: $31,200"]
        CL3["Claim 3\nmanual_override_flag: true\namount: $24,800"]
    end

    LW["Lawyer\nknown_to_siu: true\nprior_fraud_involvement: true\nclosed_loop_network_flag: true"]

    RS["RepairShop\nfraud_flag: true\ninflated_estimate_rate: 42%\nsiu_referral_count: 11"]

    W1["Witness\nprofessional_witness_flag: true\nclaim_count: 9"]
    W2["Witness\ncoached_statement_flag: true"]

    FR -->|"RING_INVOLVES_CUSTOMER"| C1 & C2 & C3
    FR -->|"RING_CONTAINS_CLAIM"| CL1 & CL2 & CL3

    C1 -->|"FILED_CLAIM"| CL1
    C2 -->|"FILED_CLAIM"| CL2
    C3 -->|"FILED_CLAIM"| CL3

    CL1 & CL2 & CL3 -->|"REPRESENTED_BY"| LW
    CL1 & CL2 & CL3 -->|"REPAIRED_AT"| RS
    CL1 & CL2 & CL3 -->|"HAS_WITNESS"| W1
    CL2 & CL3 -->|"HAS_WITNESS"| W2

    C1 -->|"SHARES_ATTRIBUTE\nbank_account\nCRITICAL"| C2
    C2 -->|"SHARES_ATTRIBUTE\nphone\nHIGH"| C3
    C1 -->|"SHARES_ATTRIBUTE\nip_address\nMEDIUM"| C3
```

## Pattern 2 — Role Switching (Identity Fraud)

Same individual appears as Customer, Witness, and claimant across multiple rings.

```mermaid
flowchart TD
    P["Person (Real Entity)"]

    P -->|"identity A"| C1["Customer\ncust_id: CUST-101\nsynthetic_identity_flag: true"]
    P -->|"identity B"| C2["Customer\ncust_id: CUST-202\nrole_switching_flag: true"]
    P -->|"witness role"| W1["Witness\nstatement_id: WIT-055\nprofessional_witness_flag: true\nclaim_count: 12"]

    C1 -->|"FILED_CLAIM"| CL1["Claim A\nRING-003"]
    C2 -->|"FILED_CLAIM"| CL2["Claim B\nRING-007"]
    W1 -->|"testified in"| CL3["Claim C\nRING-003"]
    W1 -->|"testified in"| CL4["Claim D\nRING-007"]

    C1 -->|"SHARES_ATTRIBUTE\nSSN hash match\nCRITICAL"| C2
```

## Pattern 3 — Professional Witness Network

Single witness appearing across 5+ claims is a high-confidence fraud signal.

```mermaid
flowchart LR
    W["Witness\nJohn Doe\nprofessional_witness_flag: true\nclaim_count: 9\ncoached_statement_flag: true"]

    W -->|"HAS_WITNESS ← Claim"| CL1["Claim 1\nRING-002"]
    W -->|"HAS_WITNESS ← Claim"| CL2["Claim 2\nRING-002"]
    W -->|"HAS_WITNESS ← Claim"| CL3["Claim 3\nRING-007"]
    W -->|"HAS_WITNESS ← Claim"| CL4["Claim 4\nRING-007"]
    W -->|"HAS_WITNESS ← Claim"| CL5["Claim 5\nRING-011"]
    W -->|"HAS_WITNESS ← Claim"| CL6["Claim 6\nRING-011"]
    W -->|"... 3 more"| MORE["..."]

    CL1 & CL2 & CL3 & CL4 & CL5 & CL6 -->|"REPRESENTED_BY"| LW["Same Lawyer\nOVR-003 triggered:\nSame witness\nacross jurisdictions"]
```

## Pattern 4 — Shared Bank Account (Financial Nexus)

Multiple customers share a single bank account — strong indicator of synthetic identities coordinated by a ring organizer.

```mermaid
flowchart TD
    BANK["Shared Bank Account\nbank_hash: abc123\nfraud_signal: CRITICAL"]

    C1["Customer A"] -->|"SHARES_ATTRIBUTE\nbank_account"| BANK
    C2["Customer B"] -->|"SHARES_ATTRIBUTE\nbank_account"| BANK
    C3["Customer C"] -->|"SHARES_ATTRIBUTE\nbank_account"| BANK
    C4["Customer D"] -->|"SHARES_ATTRIBUTE\nbank_account"| BANK

    C1 --> CL1["Claim 1\n$22,000"]
    C2 --> CL2["Claim 2\n$31,000"]
    C3 --> CL3["Claim 3\n$19,500"]
    C4 --> CL4["Claim 4\n$28,000"]

    CL1 & CL2 & CL3 & CL4 --> TOTAL["Total Exposure\n$100,500\nOVR-002 triggered:\n> $75K"]
```

## Pattern 5 — Cross-Jurisdiction Attorney

Licensed attorney filing claims across state lines triggers OVR-003.

```mermaid
flowchart LR
    LW["Lawyer\nBar: TX-2891\nclosed_loop_network_flag: true\nknown_to_siu: true"]

    LW -->|"REPRESENTED_BY ← Claim"| CL1["Claim - Texas\naccident_location: TX"]
    LW -->|"REPRESENTED_BY ← Claim"| CL2["Claim - Florida\naccident_location: FL"]
    LW -->|"REPRESENTED_BY ← Claim"| CL3["Claim - Nevada\naccident_location: NV"]

    CL1 --> RS1["RepairShop TX\nfraud_flag: true"]
    CL2 --> RS2["RepairShop FL\nfraud_flag: true"]
    CL3 --> RS3["RepairShop NV\nfraud_flag: true"]

    CL1 & CL2 & CL3 --> OVR["OVR-003\nSame attorney + shop\nacross jurisdictions\nOVR-005\nLicensed attorney\nprimary connecting edge"]
```

## OVR Trigger Cypher — Common Detection Queries

```mermaid
flowchart TD
    subgraph QUERIES["Cypher Patterns for OVR Evaluation"]
        Q1["OVR-001: ring_suspicion_score >= 0.90\nMATCH (nf:NetworkFeature)-[:DESCRIBES_ENTITY]->(c:Customer)\nWHERE nf.ring_suspicion_score >= 0.90"]

        Q2["OVR-002: exposure > $75K\nMATCH (r:FraudRing)-[:RING_CONTAINS_CLAIM]->(cl:Claim)\nWITH r, sum(cl.total_claim_amount) AS total\nWHERE total > 75000"]

        Q3["OVR-003: same attorney/shop across jurisdictions\nMATCH (cl:Claim)-[:REPRESENTED_BY]->(l:Lawyer)\nMATCH (cl)-[:REPAIRED_AT]->(rs:RepairShop)\nWITH l, collect(DISTINCT cl.accident_state) AS states\nWHERE size(states) > 1"]

        Q4["OVR-008: prior SIU referral\nMATCH (rs:RepairShop)\nWHERE rs.siu_referral_count > 0\nOR (l:Lawyer).known_to_siu = true"]
    end
```
