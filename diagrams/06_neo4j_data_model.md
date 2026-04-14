# Neo4j Graph Data Model

Knowledge graph schema: 24 node types · 28 edge types · 14,292 nodes · 28,690 edges · 941 properties.

## Core Entity Relationships

```mermaid
erDiagram
    Customer {
        string cust_id PK
        string full_name
        string state
        bool synthetic_identity_flag
        bool role_switching_flag
        bool shared_bank_flag
        float pagerank_score
    }
    Claim {
        string claim_id PK
        float final_suspicion_score
        float gnn_suspicion_score
        float ensemble_suspicion_score
        string adjuster_priority_tier
        bool ring_member_flag
        bool staged_accident_flag
        bool manual_override_flag
        float total_claim_amount
    }
    FraudRing {
        string ring_id PK
        float ring_score
        string status
        bool closed_loop_detected
        int member_count
        float total_amount
    }
    Lawyer {
        string lawyer_id PK
        string bar_number
        bool closed_loop_network_flag
        bool known_to_siu
        bool prior_fraud_involvement_flag
    }
    Witness {
        string statement_id PK
        string full_name
        bool professional_witness_flag
        bool coached_statement_flag
        int claim_count
    }
    RepairShop {
        string shop_id PK
        string shop_name
        bool fraud_flag
        float inflated_estimate_rate_pct
        int siu_referral_count
    }
    MedicalReport {
        string report_id PK
        string diagnosis_code
        bool inconsistency_flag
        bool duplicate_billing_flag
    }
    InvestigationCase {
        string case_id PK
        float ai_fraud_score
        bool manual_override_triggered
        string status
    }
    HumanReview {
        string review_id PK
        string decision
        string investigator_id
        int confidence
        string feedback_to_model
        bool override_ai_recommendation
    }
    NetworkFeature {
        string feature_id PK
        float pagerank_score
        float ring_suspicion_score
        int hop_2_fraud_count
        float betweenness_centrality
    }

    Customer ||--o{ Claim : "FILED_CLAIM"
    Customer }o--o{ Customer : "SHARES_ATTRIBUTE"
    FraudRing ||--o{ Claim : "RING_CONTAINS_CLAIM"
    FraudRing ||--o{ Customer : "RING_INVOLVES_CUSTOMER"
    Claim }o--o| Lawyer : "REPRESENTED_BY"
    Claim }o--o| RepairShop : "REPAIRED_AT"
    Claim ||--o{ Witness : "HAS_WITNESS"
    Claim ||--o{ MedicalReport : "HAS_MEDICAL_REPORT"
    InvestigationCase }o--o| Claim : "INVESTIGATES_CLAIM"
    HumanReview }o--o| InvestigationCase : "REVIEWS_CASE"
    NetworkFeature }o--o| Customer : "DESCRIBES_ENTITY"
```

## Node Type Inventory

```mermaid
flowchart LR
    subgraph PRIMARY["Primary Fraud Entities"]
        direction TB
        C1["Customer\n1,000 nodes\ncust_id"]
        C2["Claim\n1,000 nodes\nclaim_id"]
        C3["FraudRing\n20 nodes\nring_id"]
    end

    subgraph NETWORK["Network Participants"]
        direction TB
        N1["Lawyer\n7 nodes\nlawyer_id"]
        N2["Witness\n1,487 nodes\nstatement_id"]
        N3["RepairShop\n7 nodes\nshop_id"]
        N4["MedicalProvider\nmedical_provider_id"]
        N5["BodyShop\nbody_shop_id"]
    end

    subgraph CASE["Case Management"]
        direction TB
        M1["InvestigationCase\n876 nodes\ncase_id"]
        M2["HumanReview\n570 nodes\nreview_id"]
        M3["SIUReferral\nreferral_id"]
        M4["ClaimAuditLog\nlog_id"]
    end

    subgraph INTELLIGENCE["Graph Intelligence"]
        direction TB
        I1["NetworkFeature\n2,000 nodes\nfeature_id\npagerank · betweenness\nring_suspicion_score"]
        I2["FraudPattern\npattern_id"]
        I3["RiskProfile\nprofile_id"]
    end

    subgraph CONTEXT["Context / Reference"]
        direction TB
        X1["Vehicle\nvin"]
        X2["Policy\npolicy_id"]
        X3["Jurisdiction\njurisdiction_id"]
        X4["MedicalReport\nreport_id"]
        X5["AccidentReport\nreport_id"]
    end
```

## Critical Edge Types

```mermaid
flowchart TD
    subgraph FRAUD_SIGNAL["Fraud Signal Edges (HIGH WEIGHT in GNN)"]
        direction LR
        E1["SHARES_ATTRIBUTE\nCustomer ↔ Customer\nattribute_type: phone/bank/IP/address\nfraud_signal: CRITICAL/HIGH/MEDIUM/LOW"]
        E2["RING_CONTAINS_CLAIM\nFraudRing → Claim\nring_score · inclusion_reason"]
        E3["RING_INVOLVES_CUSTOMER\nFraudRing → Customer\nrole: organizer/recruit/victim"]
    end

    subgraph CLAIM_NETWORK["Claim Network Edges"]
        direction LR
        E4["FILED_CLAIM\nCustomer → Claim\nfiling_date · channel"]
        E5["REPRESENTED_BY\nClaim → Lawyer\nretainer_date · fee_pct"]
        E6["REPAIRED_AT\nClaim → RepairShop\nestimate_amount · inflated_flag"]
        E7["HAS_WITNESS\nClaim → Witness\nstatement_date · coached_flag"]
        E8["HAS_MEDICAL_REPORT\nClaim → MedicalReport\nbilling_amount · duplicate_flag"]
    end

    subgraph REVIEW_CHAIN["Review Chain Edges"]
        direction LR
        E9["INVESTIGATES_CLAIM\nInvestigationCase → Claim\nopened_date · priority"]
        E10["REVIEWS_CASE\nHumanReview → InvestigationCase\ndecision · confidence\nfeedback_to_model"]
        E11["DESCRIBES_ENTITY\nNetworkFeature → Customer\npagerank · hop_2_fraud"]
    end
```

## SHARES_ATTRIBUTE Fraud Signal Detail

```mermaid
flowchart LR
    C1["Customer A"] -->|"SHARES_ATTRIBUTE\nattribute_type: bank_account\nfraud_signal: CRITICAL\nshared_value_hash: abc123"| C2["Customer B"]
    C2 -->|"SHARES_ATTRIBUTE\nattribute_type: phone\nfraud_signal: HIGH"| C3["Customer C"]
    C1 -->|"SHARES_ATTRIBUTE\nattribute_type: ip_address\nfraud_signal: MEDIUM"| C4["Customer D"]

    C1 -->|"FILED_CLAIM"| CL1["Claim 1\nfinal_suspicion_score: 0.94"]
    C2 -->|"FILED_CLAIM"| CL2["Claim 2\nfinal_suspicion_score: 0.91"]
    C3 -->|"FILED_CLAIM"| CL3["Claim 3\nfinal_suspicion_score: 0.88"]

    FR["FraudRing\nring_score: 0.95\nstatus: Active"] -->|"RING_CONTAINS_CLAIM"| CL1
    FR -->|"RING_CONTAINS_CLAIM"| CL2
    FR -->|"RING_CONTAINS_CLAIM"| CL3
    FR -->|"RING_INVOLVES_CUSTOMER"| C1
    FR -->|"RING_INVOLVES_CUSTOMER"| C2
    FR -->|"RING_INVOLVES_CUSTOMER"| C3

    CL1 & CL2 & CL3 -->|"REPRESENTED_BY"| LW["Lawyer\nknown_to_siu: true\nprior_fraud_involvement: true"]
    CL1 & CL2 & CL3 -->|"REPAIRED_AT"| RS["RepairShop\nfraud_flag: true\nsiu_referral_count: 8"]
