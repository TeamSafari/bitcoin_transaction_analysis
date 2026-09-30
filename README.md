# Bitcoin Transaction Forensics - SIH PSID - 26146

> **An explainable, multi-layered forensic intelligence platform for detecting suspicious Bitcoin wallets, tracing fund-flow patterns, and turning complex transaction/network behavior into investigator-ready risk narratives.**

---

## 1. Problem Statement

Bitcoin transactions are transparent, but **transparency does not automatically mean traceability**.

Investigators and crypto-compliance teams often have to work across large transaction graphs, pseudonymous wallet addresses, rapidly changing counterparties, and noisy network activity. Traditional block explorers and rule-based monitoring can surface suspicious transactions, but they often leave the investigator with three difficult questions:

1. **Who is actually behind the observed wallet activity?**
2. **How does value move through multiple hops and interacting wallets?**
3. **Why was a wallet flagged, and what evidence supports the alert?**

The result is a workflow that can become highly manual: analysts inspect individual transactions, traverse graphs, correlate network observations, compare behavioral statistics, and assemble evidence into a coherent case.

### The core challenge

We need a system that can combine:

**Transaction behavior + graph topology + temporal behavior + network fingerprints + anomaly detection + explainability**

into a single wallet-level forensic view.

---

# 2. Our Solution

## Bitcoin Transaction Forensics

Our solution is an **end-to-end forensic analytics pipeline** that transforms raw Bitcoin transaction and network metadata into:

- wallet-level risk scores
- graph-based behavioral intelligence
- anomaly signals from multiple independent detectors
- interpretable feature-level evidence
- multi-hop fund-flow traces
- human-readable forensic summaries
- interactive investigator visualizations
- API-ready alerts and evidence

Instead of relying on a single detection rule or black-box model, the platform combines **graph analytics, supervised ML, unsupervised anomaly detection, deterministic statistics, and local LLM summarisation**.

### High-level flow

```text
Raw Transaction + Network Data
              │
              ▼
     Ingestion & Validation
              │
              ▼
      Normalisation + Identity
          Resolution
              │
              ▼
     Graph + Behavioral Features
              │
        ┌─────┴─────┐
        ▼           ▼
   GraphSAGE     NetworkX
   Embeddings    Analytics
        └─────┬─────┘
              ▼
        Feature Fusion
              │
              ▼
       Hybrid Detection
    ┌─────────┼──────────┐
    ▼         ▼          ▼
 XGBoost   Isolation   Autoencoder
           Forest
              │
              ▼
     Deterministic Statistics
              │
              ▼
        Risk Score Fusion
              │
        ┌─────┼────────────┐
        ▼     ▼            ▼
      SHAP  Pattern      Local LLM
            Tracking     Explanation
        └─────┼────────────┘
              ▼
       FastAPI + React UI
```

---

# 3. Technical Architecture

The platform is organised as a continuous five-phase pipeline, preceded by raw data ingestion.

## Phase 0 — Raw Ingestion Sources

The pipeline currently operates on structured Bitcoin and network metadata.

### Bitcoin metadata

| File | Purpose |
|---|---|
| `wallets.csv` | Wallet identifiers, script types, and entity associations |
| `transactions.csv` | Transaction-level metadata, input/output references, and fee accounting |
| `transaction_inputs.csv` | Wallet → transaction relationships and input amounts |
| `transaction_outputs.csv` | Transaction → wallet relationships and output amounts |
| `entities.csv` | Synthetic entity metadata |
| `wallet_entity_links.csv` | Confidence-scored wallet/entity relationships |

### Network metadata

| File | Purpose |
|---|---|
| `network_observations.csv` | Source/destination IPs, ports, protocol, latency, bytes, message type |
| `ip_metadata.csv` | GeoIP information including country, city, ASN, coordinates, and network type |

The eight canonical tables form the raw analytical foundation of the system.

---

# 4. Phase 1 — Ingestion, Validation & Normalisation

The first layer converts heterogeneous CSV inputs into reliable analytical tables.

### Load & validate

The pipeline performs:

- multi-file ingestion with Pandas
- column-level schema validation
- referential integrity checks
- duplicate ID detection
- transaction accounting validation

For example:

```text
Σ(transaction inputs) - Σ(transaction outputs) = transaction fee
```

This prevents downstream models from learning from structurally invalid records.

### Type normalisation

Raw values are standardised before feature generation:

- timestamps → consistent datetime representation
- satoshi values → numeric fields
- list-like columns → parsed arrays
- latitude/longitude/ASN → numeric types
- network fields → numeric/canonical representations

### Identity resolution

Wallet activity is enriched using:

- wallet ↔ entity confidence links
- wallet IP fingerprints
- unique IP/ASN profiles
- shared infrastructure detection

The result is an enriched wallet-level representation containing both transactional identity signals and network context.

---

# 5. Phase 2 — Graph Modelling & Feature Engineering

The central analytical abstraction is a **directed wallet transaction graph**.

```text
Wallet A ─────► Wallet B
   │                │
   ├────► Wallet C  │
   │                └────► Wallet D
   └──────────────────────► Wallet E
```

## Transaction Graph

Using NetworkX, transactions are converted into wallet-to-wallet relationships.

Edges can capture:

- transaction count
- total transferred amount
- first observed time
- last observed time

### Graph intelligence

For every wallet, the platform derives structural signals such as:

- degree
- in-degree / out-degree
- weighted transaction volume
- PageRank
- betweenness centrality
- clustering coefficient
- community membership
- community size
- two-hop neighbourhood reach

These features allow the system to identify wallets whose behaviour is unusual not only individually, but also **relative to their position in the broader transaction network**.

## GraphSAGE Embeddings

A GraphSAGE-based graph neural network generates compact node embeddings from local neighbourhood structure.

The embeddings provide a learned representation of wallet context and complement explicit graph statistics.

The architecture deliberately combines:

```text
Explicit graph statistics
          +
Learned graph embeddings
          +
Transaction behaviour
          +
Temporal behaviour
          +
Network behaviour
```

rather than relying on any single representation.

## Multi-domain features

### Transaction behaviour

Examples include:

- transaction frequency
- sent/received volume
- average / median / standard deviation of amounts
- fan-in / fan-out
- input/output relationships
- unique counterparties

### Temporal behaviour

Examples include:

- active duration
- inter-transaction timing
- burstiness
- off-hour activity

### Network behaviour

Examples include:

- unique source/destination IPs
- protocol diversity
- ASN/country diversity
- country and ASN entropy
- latency
- bytes transferred

### Cross-domain correlations

The system additionally captures relationships between network observations and transaction activity, including:

- network ↔ transaction timing differences
- observations per transaction
- rapid-hop activity

---

# 6. Feature Fusion

All feature domains are consolidated into a single wallet-level feature matrix.

```text
Transaction Features ─┐
Temporal Features ────┤
Network Features ─────┤
Graph Features ───────┤──► Feature Fusion ──► Wallet Feature Matrix
GraphSAGE Embeddings ─┘
```

Fusion is performed around `wallet_id`, followed by data-quality processing such as:

- median imputation
- constant-feature removal
- consistent feature alignment

This gives every downstream detector a common view of wallet behaviour.

---

# 7. Phase 3 — Hybrid AI/ML Detection Ensemble

A key design decision is to avoid depending on one model.

The platform uses multiple detectors with different assumptions and failure modes.

## 7.1 XGBoost — Risk Classification

A supervised XGBoost classifier estimates the probability that a wallet belongs to the suspicious class.

Outputs include:

```text
risk_probability
risk_prediction
```

SHAP is used alongside the model to explain the contribution of individual features.

---

## 7.2 Isolation Forest — Behavioural Anomaly Detection

Isolation Forest detects observations that are unusual relative to the overall feature distribution.

It provides:

- raw anomaly scores
- normalized anomaly scores
- anomaly flags

This complements supervised classification by allowing the system to detect behaviour that may not be explicitly represented in the labelled examples.

---

## 7.3 Autoencoder — Reconstruction Anomaly Detection

A neural autoencoder learns to reconstruct normal feature patterns.

A larger reconstruction error indicates that the observed wallet behaviour differs from learned patterns.

Output:

```text
reconstruction_error
```

---

## 7.4 Deterministic Statistical Engine

A model-independent statistical layer provides transparent anomaly evidence using:

- robust Z-scores based on MAD
- empirical percentile ranks
- two-sided anomaly detection

The engine also identifies the most influential anomalous features for each wallet.

This adds a deterministic evidence layer that can be inspected independently of the ML models.

---

# 8. Data-Driven Risk Fusion

The detector outputs are normalised and combined into a single composite risk representation.

The fusion process uses:

### Empirical percentile normalisation

Detector outputs are transformed to comparable percentile-based scales.

### Correlation-aware redundancy handling

Spearman correlation is used to identify overlapping detector signals so that highly redundant evidence does not dominate the final score.

### Weighted aggregation

The final score is generated using correlation-adjusted, data-driven weighting rather than manually assigning arbitrary model weights.

Conceptually:

```text
XGBoost score
      │
Isolation Forest score
      ├────► Percentile Normalisation
Autoencoder score
      │
Statistical score
      │
      ▼
Correlation-aware Fusion
      │
      ▼
Composite Wallet Risk
```

---

# 9. Phase 4 — Risk Scoring & Explainability

The combined detector output is converted into investigator-facing risk information.

## Risk bands

Wallets are routed into empirical risk bands:

```text
LOW ───── MEDIUM ───── HIGH ───── CRITICAL
0–25%      25–50%       50–75%        75–100%
```

The platform can use configurable thresholds for review and hold workflows.

---

# 10. SHAP Explainability

A major design goal is to make the system **investigator-friendly rather than model-centric**.

For every flagged wallet, SHAP provides feature-level contributions from the XGBoost classifier.

The explanation layer:

- extracts top contributing factors
- identifies whether a factor increases or decreases risk
- maps internal feature names to human-readable labels
- attaches descriptions suitable for an investigation interface

Example:

```text
Wallet Risk: HIGH

Top contributing factors:
  ↑ High fan-out behaviour
  ↑ Unusual transaction burstiness
  ↑ High ASN diversity
  ↑ Rapid network-to-transaction activity
  ↓ Low long-term activity duration
```

GraphSAGE dimensions are intentionally treated as model features rather than surfaced as opaque investigator-facing explanations.

---

# 11. Pattern Tracking & Fund-Flow Forensics

Risk scores tell us **which wallets deserve attention**.

Pattern tracking helps answer **what happened**.

A BFS-based traversal traces fund flows across multiple hops.

The system can identify patterns such as:

```text
Direct Transfer
Fan-out
Consolidation
Layering
Round-trip
```

Each detected pattern can include:

- participating wallets
- hop sequence
- confidence
- transaction relationships
- animation-ready trace sequence

This converts a static graph into an investigation workflow.

---

# 12. Local LLM Summarisation

The final intelligence layer translates structured evidence into a concise narrative.

We use a local:

**Qwen2.5-1.5B-Instruct — GGUF Q4_K_M**

with CPU inference through:

**llama-cpp-python**

The model does not independently decide whether a wallet is suspicious.

Instead, it receives structured evidence assembled from the analytics pipeline.

### Context assembly

```text
Risk Verdict
     +
SHAP Factors
     +
Behavioural Profile
     +
Anomaly Signals
     ↓
Context Builder
     ↓
Local Qwen Model
     ↓
Human-readable forensic explanation
```

This design keeps the LLM in an **explanation and summarisation role**, while quantitative risk remains grounded in the deterministic and ML pipeline.

Additional implementation details include:

- lightweight local inference
- response caching in SQLite
- thread-safe model management
- compact wallet-level explanations

---

# 13. Phase 5 — API & Interactive Visualisation

## FastAPI backend

The backend exposes the analytical pipeline and investigator-facing evidence through REST APIs.

Representative endpoints include:

| Endpoint | Purpose |
|---|---|
| `/jobs/upload` | Upload a forensic dataset |
| `/jobs/{id}/status` | Track pipeline execution |
| `/jobs/{id}/alerts` | Retrieve generated alerts |
| `/graph/{id}/wallet/{wallet_id}` | Retrieve wallet graph/subgraph data |
| `/explainability/{id}/wallet/{wallet_id}` | Retrieve SHAP + statistical evidence |
| `/llm-explain/{id}/wallet/{wallet_id}` | Generate/retrieve LLM explanation |
| `/patterns/{id}/wallet/{wallet_id}` | Retrieve fund-flow traces |

The backend also provides:

- background job execution
- SQLite job/artifact management
- Pydantic validation
- CORS support
- asynchronous processing paths

---

# 14. Investigation Dashboard

The frontend is designed around rapid visual investigation rather than raw data tables.

### Core capabilities

- interactive Cytoscape.js transaction graphs
- risk-based node visualisation
- wallet detail sidebar
- SHAP evidence
- behavioural statistics
- LLM-generated explanations
- multi-file CSV upload
- real-time job status polling
- animated fund-flow tracing
- REST API-backed investigation views

An investigator can move from:

```text
Alert
  ↓
Wallet
  ↓
Evidence
  ↓
Graph Context
  ↓
Fund Flow Pattern
  ↓
Natural-language Case Summary
```

without manually switching between unrelated forensic tools.

---

# 15. Why This Architecture

The architecture is intentionally **layered and modular**.

### Multiple evidence types

Suspicious behaviour can appear as:

- unusual transaction volume
- unusual graph topology
- abnormal timing
- network infrastructure reuse
- anomalous feature combinations

Using multiple evidence sources reduces dependence on one signal.

### Explainability by design

Every major detection output has a corresponding evidence layer:

```text
ML prediction       → SHAP contributions
Anomaly detection   → anomaly score + features
Statistical engine  → robust statistical evidence
Graph analysis      → topology + fund-flow context
LLM                 → human-readable synthesis
```

### Separation of responsibilities

The system keeps the main components decoupled:

```text
Ingestion
   │
Graph / Feature Engineering
   │
Detection
   │
Risk Fusion
   │
Explainability
   │
API
   │
Visualisation
```

This makes individual components easier to test, replace, scale, and extend.

---

# 16. Synthetic / Demo Dataset

The current demonstration environment uses a compact synthetic forensic dataset designed to exercise the entire pipeline.

Representative data includes:

- **50 wallets**
- **196 transactions**
- **12 synthetic entities**
- **15 IP addresses**
- transaction input/output relationships
- network observations
- wallet/entity links

The pipeline is designed to operate from the canonical tables rather than relying on a specific hard-coded dataset size.

---

# 17. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Data processing | Pandas, NumPy |
| Graph analytics | NetworkX |
| Graph ML | PyTorch, PyTorch Geometric |
| Supervised ML | XGBoost |
| Anomaly detection | Scikit-learn |
| Explainability | SHAP |
| API | FastAPI, Pydantic |
| Persistence | SQLite |
| Frontend | React 19, Vite |
| Graph visualisation | Cytoscape.js |
| Local LLM | Qwen2.5-1.5B Instruct |
| LLM runtime | llama-cpp-python |
| Data interchange | CSV / JSON |

---

# 18. End-to-End Architecture

```text
                         ┌──────────────────────────────┐
                         │       RAW DATA SOURCES       │
                         │                              │
                         │ Bitcoin Metadata             │
                         │ Network Metadata             │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │  INGESTION & VALIDATION      │
                         │                              │
                         │ Schema • Integrity • Types   │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ IDENTITY RESOLUTION          │
                         │                              │
                         │ Wallet ↔ Entity ↔ Network   │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
              ┌─────────────────────────┴────────────────────────┐
              │                                                  │
              ▼                                                  ▼
   ┌────────────────────────┐                         ┌────────────────────────┐
   │ TRANSACTION GRAPH      │                         │ BEHAVIOURAL FEATURES   │
   │                        │                         │                        │
   │ NetworkX               │                         │ TX • Time • Network    │
   │ PageRank / Communities │                         │ Cross-domain signals   │
   └────────────┬───────────┘                         └────────────┬───────────┘
                │                                                  │
                ▼                                                  ▼
       ┌────────────────────┐                          ┌────────────────────┐
       │    GraphSAGE       │                          │    Feature Fusion   │
       │    Embeddings      │──────────────┬──────────►│                    │
       └────────────────────┘              │           └─────────┬──────────┘
                                           │                     │
                                           └─────────────────────┘
                                                         │
                                                         ▼
                              ┌─────────────────────────────────────────┐
                              │       HYBRID DETECTION ENSEMBLE         │
                              │                                         │
                              │ XGBoost • Isolation Forest              │
                              │ Autoencoder • Statistical Engine        │
                              └──────────────────┬──────────────────────┘
                                                 │
                                                 ▼
                              ┌─────────────────────────────────────────┐
                              │       CORRELATION-AWARE RISK FUSION     │
                              └──────────────────┬──────────────────────┘
                                                 │
                         ┌───────────────────────┼────────────────────────┐
                         │                       │                        │
                         ▼                       ▼                        ▼
                  ┌─────────────┐       ┌──────────────┐         ┌──────────────┐
                  │    SHAP     │       │   PATTERN    │         │ LOCAL QWEN   │
                  │ Explainable │       │   TRACKING   │         │  Summaries   │
                  └──────┬──────┘       └──────┬───────┘         └──────┬───────┘
                         │                     │                        │
                         └─────────────────────┼────────────────────────┘
                                               │
                                               ▼
                                ┌────────────────────────────┐
                                │       FASTAPI BACKEND      │
                                │                            │
                                │ Jobs • Alerts • Graphs    │
                                │ Evidence • Explanations   │
                                └──────────────┬─────────────┘
                                               │
                                               ▼
                                ┌────────────────────────────┐
                                │      REACT DASHBOARD       │
                                │                            │
                                │ Graph • Risk • Evidence    │
                                │ Patterns • LLM Summary     │
                                └────────────────────────────┘
```

---

# 19. Key Design Principles

### 1. Evidence before explanation

The system generates quantitative and structural evidence first. The LLM then converts that evidence into readable investigator context.

### 2. Multiple independent signals

Supervised classification, unsupervised anomaly detection, graph structure, and deterministic statistics provide complementary views of suspicious behaviour.

### 3. Human-in-the-loop investigation

The system surfaces and prioritises evidence; investigators remain responsible for interpreting the case and taking action.

### 4. Modular architecture

Each analytical stage can evolve independently without redesigning the entire platform.

### 5. Privacy-aware intelligence

The LLM summarisation layer can run locally, avoiding the need to send sensitive forensic context to an external model API.

---

# 20. Scalable Architecture

Although the current implementation focuses on Bitcoin Transaction Analysis, the architecture is designed around reusable abstractions:

```text
Chain-specific ingestion
          ↓
Canonical wallet/entity representation
          ↓
Graph construction
          ↓
Feature engineering
          ↓
Detection + Explainability
          ↓
Investigator Interface
```

This creates a path toward extending the system to additional blockchain ecosystems where account, transaction, and graph semantics can be mapped into the same analytical framework.

---

# 21. Summary

**Bitcoin Transaction Forensics** brings together data engineering, graph intelligence, machine learning, statistical analysis, explainability, and local language models into one investigation workflow.

The objective is not simply to produce an anomaly score.

It is to move from:

> **“This wallet looks suspicious.”**

to:

> **“This wallet is flagged because of specific behavioral, graph, temporal, and network evidence — here is how the value moved, why the signals matter, and a concise explanation an investigator can act on.”**

---
