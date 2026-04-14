# Deployment Architecture

Production topology across cloud services.

```mermaid
flowchart TB
    subgraph DEV["Developer Machine (Local)"]
        direction TB
        CODE["Source Code\nGitHub Repo"]
        LOCAL_ENV[".env\nNEO4J_URI\nOPENROUTER_API_KEY\nPINECONE_API_KEY"]
        GNN_TRAIN["GNN Training\npython src/main.py gnn train\nrequires: PyTorch Geometric\nNOT on Render"]
        RAG_INDEX["Pinecone Indexing\npython src/main.py rag index\nrun once before deploy"]
    end

    subgraph GITHUB["GitHub"]
        REPO["Repository\nmain branch\napi.py · src/ · ui/\nrequirements.txt\nbuild.sh · render.yaml"]
    end

    subgraph RENDER["Render.com — Free Plan"]
        direction TB
        BUILD["build.sh\npip install torch CPU\npip install -r requirements.txt"]
        API_SVC["FastAPI Web Service\nuvicorn api:app\nPort 10000\n512 MB RAM"]
        ENV_RENDER["Secret Env Vars\nNEO4J_URI\nNEO4J_USER\nNEO4J_PASSWORD\nOPENROUTER_API_KEY\nPINECONE_API_KEY"]
        COLD["Cold start after\n15 min inactivity\n~30 s wake-up"]
    end

    subgraph STREAMLIT_CLOUD["Streamlit Community Cloud"]
        direction TB
        UI_SVC["Streamlit App\nui/streamlit_app.py\nui/requirements.txt"]
        SECRET_ST["Secrets\napi_url = https://render-url"]
        ALWAYS_ON["Always-on\nno cold starts\nfree tier"]
    end

    subgraph NEO4J_AURA["Neo4j Aura (Cloud)"]
        direction TB
        GDB["Graph Database\n14,292 nodes\n28,690 edges\nGDS plugin"]
        BOLT["neo4j+s://\nBolt protocol\nTLS encrypted"]
    end

    subgraph PINECONE_CLOUD["Pinecone (Cloud)"]
        direction TB
        VIDX["Vector Index\nfraud-rings\n384-dim cosine\n20 FraudRing embeddings\nfree starter tier"]
        PERSIST["Persists across\nRender deploys\nand cold starts"]
    end

    subgraph OPENROUTER["OpenRouter"]
        direction TB
        LLM["anthropic/claude-sonnet-4-5\nOpenAI-compatible API\nhttps://openrouter.ai/api/v1\nPay-per-token"]
    end

    CODE -->|"git push"| REPO
    LOCAL_ENV -.->|"credentials\nnever committed"| CODE
    GNN_TRAIN -->|"models/fraud_gnn.pt\nmodels/ensemble.pkl\ngit commit + push"| REPO
    RAG_INDEX -->|"push embeddings once"| PINECONE_CLOUD

    REPO -->|"auto-deploy\nrender.yaml"| RENDER
    REPO -->|"connect repo\nmain file: ui/streamlit_app.py"| STREAMLIT_CLOUD

    BUILD --> API_SVC
    ENV_RENDER --> API_SVC

    API_SVC <-->|"Bolt/TLS\nCypher queries\nwrite HumanReview"| NEO4J_AURA
    API_SVC <-->|"HTTPS\ncosine search\ntop-K rings"| PINECONE_CLOUD
    API_SVC <-->|"HTTPS\nchat.completions"| OPENROUTER

    UI_SVC <-->|"HTTPS REST\nGET /investigate\nPOST /feedback\nGET /stats"| API_SVC
    SECRET_ST --> UI_SVC
```

## RAM Budget — Render Free Plan (512 MB)

```mermaid
flowchart LR
    subgraph RAM["512 MB Render Free Plan"]
        direction TB
        R1["PyTorch CPU runtime\n~200 MB"]
        R2["paraphrase-MiniLM-L3-v2\nembedding weights\n~17 MB"]
        R3["FastAPI + uvicorn\n~20 MB"]
        R4["LangGraph + langchain\n~30 MB"]
        R5["Neo4j driver + pinecone\n~15 MB"]
        R6["Other libs\n~18 MB"]
        TOTAL["Total: ~300 MB\nFits free plan\n212 MB headroom"]
    end
```

## Request Flow — Investigate Claim

```mermaid
flowchart LR
    INV["Investigator\nbrowser"] -->|"Enter CLM-521585\nclick Investigate"| STR["Streamlit\nCommunity Cloud"]
    STR -->|"GET /investigate/CLM-521585\nHTTPS"| RND["Render.com\nFastAPI"]
    RND -->|"9 Cypher queries\nBolt TLS"| NEO["Neo4j Aura"]
    NEO -->|"subgraph dict"| RND
    RND -->|"embed 384-dim\nlocal sentence-transformer"| EMB["MiniLM weights\nin memory"]
    EMB -->|"vector"| RND
    RND -->|"cosine search\nHTTPS"| PCN["Pinecone"]
    PCN -->|"top-3 ring IDs\n+ metadata"| RND
    RND -->|"chat.completions\nHTTPS"| OPR["OpenRouter\nClaude Sonnet"]
    OPR -->|"Investigation Brief\n~800 tokens"| RND
    RND -->|"JSON response\n~2 KB"| STR
    STR -->|"Render UI\nfraud score banner\noverride triggers\nanalogous rings\nClaude brief"| INV
```

## Deployment Checklist

```mermaid
flowchart TD
    START(["Deploy"]) --> S1["1. Train GNN locally\npython src/main.py gnn train"]
    S1 --> S2["2. Score claims\npython src/main.py gnn score"]
    S2 --> S3["3. Create Pinecone index\nfraud-rings · 384-dim · cosine"]
    S3 --> S4["4. Index FraudRings\npython src/main.py rag index"]
    S4 --> S5["5. Commit model weights\nmodels/fraud_gnn.pt\nmodels/ensemble.pkl"]
    S5 --> S6["6. Push to GitHub\ngit push origin main"]
    S6 --> S7["7. Render auto-deploys\nrender.yaml picks up changes"]
    S7 --> S8["8. Set Render secrets\nNEO4J_URI · NEO4J_USER\nNEO4J_PASSWORD\nOPENROUTER_API_KEY\nPINECONE_API_KEY"]
    S8 --> S9["9. Verify API health\nGET /health → 200 OK"]
    S9 --> S10["10. Deploy Streamlit UI\nshare.streamlit.io\nMain file: ui/streamlit_app.py\nRequirements: ui/requirements.txt"]
    S10 --> S11["11. Set Streamlit secret\napi_url = https://your.onrender.com"]
    S11 --> DONE(["System Live"])
