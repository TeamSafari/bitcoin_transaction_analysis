# Bitcoin Transaction & Wallet Forensics --- Full Pipeline Implementation Plan

## 1. Objective

Make the complete Bitcoin forensic pipeline executable end-to-end with
one logical flow:

``` text
Raw CSV / JSON / XML
        ↓
Data Ingestion
(Parse → Validate → Normalize)
        ↓
Feature Engineering
(Transaction + Temporal + Network + Correlation)
        ↓
NetworkX
(Transaction graph + graph metrics)
        ↓
GraphSAGE
(32-D wallet embeddings)
        ↓
 ┌───────────────┬────────────────┐
 ↓               ↓                ↓
Isolation      Autoencoder     GraphSAGE
Forest          Anomaly         Embeddings
 ↓               ↓                ↓
 └───────────────┴────────────────┘
                ↓
      Deterministic Risk Layer
      (statistical, no labels)
                ↓
      Risk Feature Fusion
                ↓
        XGBoost Risk Model
                ↓
              SHAP
                ↓
        Alert Generation
                ↓
             FastAPI
                ↓
        Investigation Output
```

The core principle is that **every downstream component consumes a
defined artifact produced by the previous stage**. No component should
silently read a different dataset, bypass ingestion, recreate features
independently, or depend on stale output files.

------------------------------------------------------------------------

# 2. Repository Structure

The repository should follow this structure:

``` text
BITCOIN_TRANSACTION_ANALYSIS/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── api_server.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── csv_loader.py
│   │   ├── json_loader.py
│   │   ├── xml_loader.py
│   │   ├── normalizer.py
│   │   ├── validator.py
│   │   └── ingestion_pipeline.py
│   │
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   ├── feature_pipeline.py
│   │   ├── transaction_features.py
│   │   ├── temporal_features.py
│   │   ├── network_features.py
│   │   └── correlation_features.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── graph_engine.py
│   │   ├── networkx_graph.py
│   │   ├── graph_features.py
│   │   └── graphsage.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── isolation_forest.py
│   │   ├── autoencoder.py
│   │   ├── risk_model.py
│   │   └── artifacts/
│   │       ├── feature_scaler.pkl
│   │       ├── graphsage.pt
│   │       ├── isolation_forest.pkl
│   │       ├── isolation_forest_scaler.pkl
│   │       ├── autoencoder.pt
│   │       ├── autoencoder_scaler.pkl
│   │       └── risk_model.pkl
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── deterministic_engine.py
│   │   ├── statistical_thresholds.py
│   │   ├── evaluate_deterministic.py
│   │   ├── run_deterministic.py
│   │   └── risk_fusion.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_engine.py
│   │
│   ├── alerts/
│   │   ├── __init__.py
│   │   └── alert_service.py
│   │
│   └── agent/
│       ├── __init__.py
│       ├── agent.py
│       ├── prompts.py
│       └── tools.py
│
├── data/
│   ├── raw/
│   ├── exports/
│   ├── geoip/
│   └── ground_truth/
│
├── outputs/
│   ├── features/
│   ├── graphs/
│   ├── models/
│   ├── explainability/
│   ├── alerts/
│   └── reports/
│
├── .gitignore
├── README.md
└── requirements.txt
```

Remove `__pycache__/` and generated Python bytecode from Git.

------------------------------------------------------------------------

# 3. Canonical Input Dataset

The canonical raw dataset currently consists of these tables:

  -------------------------------------------------------------------------------
  File                                        Expected rows Purpose
  ---------------------------- ---------------------------- ---------------------
  `entities.csv`                                        100 Entity information

  `wallets.csv`                                         500 Wallet master records

  `wallet_entity_links.csv`                             543 Wallet ↔ entity
                                                            relationships

  `ip_metadata.csv`                                     100 IP/network metadata

  `transactions.csv`                                 10,019 Transaction master
                                                            records

  `transaction_inputs.csv`                           10,019 Transaction input
                                                            relationships

  `transaction_outputs.csv`                          11,734 Transaction output
                                                            relationships

  `network_observations.csv`                         15,603 Network observations
  -------------------------------------------------------------------------------

Ground truth:

``` text
data/ground_truth/wallet_ground_truth.csv
```

Ground truth contains 500 wallet records:

``` text
wallet_id
primary_entity_id
label
behavior_type
scenario_count
scenario_ids
transaction_count
generator_wallet_type
```

Current label distribution:

``` text
normal      461
suspicious   39
```

### Important rule

Ground truth is **not** part of runtime anomaly feature generation or
the deterministic risk calculation.

It may be used for:

-   supervised XGBoost training
-   model evaluation
-   validation
-   benchmarking

It must not leak into the deterministic risk layer.

------------------------------------------------------------------------

# 4. Stage 1 --- Data Ingestion

## 4.1 Required behavior

Create a single ingestion entry point:

``` text
backend/ingestion/ingestion_pipeline.py
```

The ingestion pipeline should:

1.  Locate the canonical raw directory.
2.  Load all required CSV tables.
3.  Parse optional JSON/XML transaction exports when explicitly
    requested.
4.  Normalize datatypes.
5.  Normalize timestamps.
6.  Normalize list-like fields.
7.  Validate required columns.
8.  Validate unique identifiers.
9.  Validate foreign-key/reference relationships.
10. Validate transaction accounting relationships.
11. Return one normalized in-memory dataset object.

The feature pipeline must call this ingestion layer.

It must **not** directly call:

``` python
pd.read_csv(...)
```

on the raw files.

That was previously a wiring problem.

------------------------------------------------------------------------

## 4.2 Loader responsibilities

### CSV loader

`csv_loader.py`

Responsible for loading the canonical eight raw CSV tables.

It should not require:

``` text
transactions_display.csv
```

because that is an export/display file rather than the canonical source.

### JSON loader

`json_loader.py`

Optional transaction-export loader.

### XML loader

`xml_loader.py`

Optional transaction-export loader.

JSON/XML are alternate transaction representations, not replacements for
the complete eight-table canonical dataset.

------------------------------------------------------------------------

# 5. Normalization

`normalizer.py` should normalize:

-   timestamps
-   numeric columns
-   nullable values
-   list-like fields
-   identifiers
-   categorical text

Important implementation detail:

When parsing list-like values, check for Python lists **before** calling
`pd.isna()`.

Correct logical order:

``` python
if isinstance(value, list):
    return value

if pd.isna(value):
    return []

# then parse strings
```

Calling `pd.isna()` first on a list can produce an array and lead to an
ambiguous truth-value error.

------------------------------------------------------------------------

# 6. Validation

`validator.py` must validate:

### Schema

Every required table contains the expected columns.

### IDs

Primary identifiers are unique.

### References

Examples:

``` text
transaction_inputs.transaction_id
        → transactions.transaction_id

transaction_outputs.transaction_id
        → transactions.transaction_id

wallet_entity_links.wallet_id
        → wallets.wallet_id

wallet_entity_links.entity_id
        → entities.entity_id
```

### Accounting

Transaction input/output relationships should be internally consistent.

### Failure behavior

Invalid input should fail early with a clear error.

Do not allow malformed data to reach feature engineering.

------------------------------------------------------------------------

# 7. Stage 2 --- Feature Engineering

Entry point:

``` text
backend/feature_engineering/feature_pipeline.py
```

The feature pipeline must start with:

``` text
ingest_dataset(...)
```

and consume the normalized tables returned by ingestion.

It should produce wallet-level features.

Current expected result:

``` text
500 wallets
52 engineered wallet features
```

Output:

``` text
outputs/features/wallet_features.csv
outputs/features/wallet_features_scaled.csv
outputs/features/feature_names.json
```

------------------------------------------------------------------------

# 8. Feature Families

## 8.1 Transaction Features

`transaction_features.py`

Generate wallet-level behavior such as:

-   transaction counts
-   input/output counts
-   total input value
-   total output value
-   average transaction value
-   transaction value dispersion
-   input/output ratios
-   counterparty counts
-   unique counterparties
-   transaction frequency

### Important correction

`unique_input_counterparties` and `unique_output_counterparties` must
represent actual opposite-side wallet relationships.

Do not calculate them by simply grouping by the wallet itself.

The correct approach is to join same-transaction opposite-side wallets
and count unique counterparties.

------------------------------------------------------------------------

# 9. Temporal Features

`temporal_features.py`

Generate:

-   first activity timestamp
-   last activity timestamp
-   active duration
-   transaction frequency
-   inter-transaction timing statistics
-   burst/activity concentration
-   temporal dispersion

Avoid arbitrary hardcoded behavioral thresholds where possible.

Prefer statistics derived from the observed dataset.

------------------------------------------------------------------------

# 10. Network Features

`network_features.py`

Generate wallet-level network behavior from transaction relationships.

Examples:

-   counterparties
-   incoming/outgoing relationship counts
-   flow ratios
-   network interaction statistics
-   network observation relationships

These should be feature-engineering signals, not duplicated NetworkX
graph metrics.

------------------------------------------------------------------------

# 11. Correlation Features

`correlation_features.py`

Relate network observations to transaction activity.

Current intended outputs include:

``` text
avg_network_tx_time_difference
min_network_tx_time_difference
observations_per_tx
rapid_hop_count
```

The rapid-hop threshold should be statistically derived from observed
positive timestamp differences rather than a fixed arbitrary number of
seconds.

------------------------------------------------------------------------

# 12. Feature Schema Contract

The feature pipeline must save the exact feature names used to construct
the model input.

``` text
outputs/features/feature_names.json
```

Every downstream model should either:

1.  read this schema, or
2.  store its own exact feature column list in its artifact.

Never rely on implicit DataFrame column ordering.

------------------------------------------------------------------------

# 13. Stage 3 --- NetworkX Graph

Entry point:

``` text
backend/graph/graph_engine.py
backend/graph/networkx_graph.py
```

Input:

``` text
normalized transactions
normalized transaction_inputs
normalized transaction_outputs
```

Output:

``` text
outputs/graphs/graph_edges.csv
outputs/graphs/graph_features.csv
```

Current prototype result:

``` text
500 wallet nodes
10,975 aggregated edges
```

------------------------------------------------------------------------

# 14. Graph Construction

The current graph construction models wallet-to-wallet value flow.

Repeated wallet pairs should be aggregated.

Self-transfers should be excluded from ordinary wallet-to-wallet edges
where appropriate.

When a transaction has multiple inputs and outputs, proportional output
allocation may be used to estimate the contribution of each input wallet
to each output wallet.

This is a modeling assumption and must be documented.

The graph should contain stable wallet identifiers.

------------------------------------------------------------------------

# 15. NetworkX Graph Features

`graph_features.py`

Generate structural metrics such as:

``` text
degree
in_degree
out_degree
weighted_degree
weighted_in_degree
weighted_out_degree
pagerank
betweenness_centrality
clustering_coefficient
community_id
community_size
two_hop_neighbor_count
```

These are graph-derived features.

They must remain logically separate from the base wallet behavioral
feature set.

------------------------------------------------------------------------

# 16. Stage 4 --- GraphSAGE

Entry point:

``` text
backend/graph/graphsage.py
```

GraphSAGE input:

``` text
wallet_features.csv
+
graph_edges.csv
```

It should **not** use NetworkX-derived graph metrics as node features.

Exclude:

``` text
degree
in_degree
out_degree
weighted_degree
weighted_in_degree
weighted_out_degree
pagerank
betweenness_centrality
clustering_coefficient
community_id
community_size
two_hop_neighbor_count
```

This keeps GraphSAGE representation learning separate from explicit
NetworkX metrics.

------------------------------------------------------------------------

# 17. GraphSAGE Training

Use:

-   PyTorch
-   PyTorch Geometric
-   two-layer GraphSAGE
-   self-supervised link prediction
-   proper negative sampling

The prototype target is:

``` text
32-dimensional wallet embedding
```

Output:

``` text
outputs/graphs/graphsage_embeddings.csv
```

Schema:

``` text
wallet_id
embedding_0
embedding_1
...
embedding_31
```

Artifact:

``` text
backend/models/artifacts/graphsage.pt
```

The checkpoint should store:

-   model state
-   input dimension
-   hidden/embedding dimensions
-   exact node feature columns
-   scaler mean
-   scaler scale
-   random state

This makes inference reproducible.

------------------------------------------------------------------------

# 18. Stage 5 --- Isolation Forest

Entry point:

``` text
backend/models/isolation_forest.py
```

Inputs:

``` text
wallet_features.csv
graph_features.csv
graphsage_embeddings.csv
```

Use:

-   base behavioral features
-   NetworkX graph features
-   GraphSAGE embeddings

Do not include:

``` text
community_id
```

as a continuous model feature.

Do not include ground-truth labels.

------------------------------------------------------------------------

# 19. Isolation Forest Processing

Pipeline:

``` text
merge
  ↓
numeric conversion
  ↓
missing-value handling
  ↓
constant-feature removal
  ↓
StandardScaler
  ↓
IsolationForest
  ↓
wallet anomaly score
```

Current implementation target:

``` text
300 trees
random_state = 42
contamination = auto
```

Outputs:

``` text
outputs/models/isolation_forest_scores.csv
backend/models/artifacts/isolation_forest.pkl
backend/models/artifacts/isolation_forest_scaler.pkl
```

Score output:

``` text
wallet_id
isolation_forest_anomaly_score
isolation_forest_flag
```

------------------------------------------------------------------------

# 20. Stage 6 --- Autoencoder

Entry point:

``` text
backend/models/autoencoder.py
```

Inputs:

``` text
wallet_features.csv
graph_features.csv
graphsage_embeddings.csv
```

Do not use ground truth.

Pipeline:

``` text
merge
  ↓
numeric cleaning
  ↓
constant-feature removal
  ↓
StandardScaler
  ↓
PyTorch MLP Autoencoder
  ↓
reconstruction error
```

Current prototype configuration:

``` text
100 epochs
learning rate = 1e-3
weight decay = 1e-5
batch size = 256
```

Output:

``` text
outputs/models/autoencoder_scores.csv
```

Artifacts:

``` text
backend/models/artifacts/autoencoder.pt
backend/models/artifacts/autoencoder_scaler.pkl
```

Store exact model input columns in the checkpoint.

------------------------------------------------------------------------

# 21. Stage 7 --- Deterministic Statistical Risk

Entry point:

``` text
backend/risk/deterministic_engine.py
backend/risk/run_deterministic.py
```

This layer is intentionally independent from:

``` text
Isolation Forest
Autoencoder
GraphSAGE
ground truth
XGBoost
```

It should use:

``` text
wallet behavioral features
+
NetworkX graph metrics
```

------------------------------------------------------------------------

# 22. Deterministic Risk Logic

The deterministic engine should not contain manually chosen risk weights
such as:

``` text
0.4 * feature_a
+
0.3 * feature_b
+
0.3 * feature_c
```

Instead:

1.  Calculate empirical feature percentile ranks.
2.  Measure two-sided deviation from the feature median.
3.  Convert each feature to an anomaly signal.
4.  Aggregate signals statistically.
5.  Produce a wallet-level deterministic score.

The current design uses the mean of the feature anomaly scores.

Outputs:

``` text
outputs/models/deterministic_scores.csv
outputs/models/deterministic_statistics.json
```

Expected fields include:

``` text
wallet_id
deterministic_score
deterministic_percentile
deterministic_feature_count
```

------------------------------------------------------------------------

# 23. Deterministic Layer Rules

The deterministic layer must never consume:

``` text
ground_truth label
isolation_forest_anomaly_score
autoencoder_reconstruction_error
risk_probability
XGBoost prediction
```

Otherwise the supposedly independent statistical signal becomes
circular.

------------------------------------------------------------------------

# 24. Stage 8 --- Risk Feature Fusion

Entry point:

``` text
backend/risk/risk_fusion.py
```

The fusion stage creates two conceptual datasets.

## Base feature dataset

Contains:

``` text
wallet behavioral features
+
NetworkX graph metrics
+
GraphSAGE embeddings
+
Isolation Forest score
+
Autoencoder score
+
deterministic score
```

Output:

``` text
outputs/models/base_fused_features.csv
```

## Supervised training dataset

Adds:

``` text
label
```

Output:

``` text
outputs/models/risk_training_features.csv
```

------------------------------------------------------------------------

# 25. Ground Truth Handling

The supplied ground truth labels are strings:

``` text
normal
suspicious
```

Map them explicitly:

``` text
normal      → 0
suspicious  → 1
```

Reject unsupported labels.

Do not silently convert arbitrary strings to numeric values.

------------------------------------------------------------------------

# 26. Important Model-Evaluation Caveat

The current prototype can generate unsupervised detector outputs for the
complete wallet batch and then train/test XGBoost.

This is convenient for the prototype but can create optimistic
evaluation because detector transformations were not isolated inside
each training fold.

For rigorous evaluation, implement:

``` text
stratified train/test split
        ↓
fit preprocessing on train only
        ↓
fit unsupervised detector on train
        ↓
generate validation/test detector features
        ↓
train XGBoost
        ↓
evaluate
```

An even stronger implementation uses out-of-fold detector features for
the training set.

This should be implemented before presenting benchmark metrics as
production-grade.

------------------------------------------------------------------------

# 27. Stage 9 --- XGBoost Risk Model

Entry point:

``` text
backend/models/risk_model.py
```

Input:

``` text
outputs/models/risk_training_features.csv
```

Target:

``` text
label
```

Exclude:

``` text
wallet_id
label
```

from the model feature matrix.

------------------------------------------------------------------------

# 28. XGBoost Configuration

Current prototype configuration:

``` text
XGBClassifier
n_estimators = 300
max_depth = 5
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = 3
reg_alpha = 0.1
reg_lambda = 1
objective = binary:logistic
random_state = 42
```

Because the suspicious class is much smaller than the normal class, use:

``` text
scale_pos_weight = negative_count / positive_count
```

------------------------------------------------------------------------

# 29. XGBoost Outputs

Save:

``` text
outputs/models/risk_model_predictions.csv
outputs/models/risk_model_test_predictions.csv
outputs/models/risk_model_metrics.txt
```

Production/batch predictions should include:

``` text
wallet_id
risk_probability
risk_prediction
```

The test set should include:

``` text
wallet_id
actual_label
risk_probability
risk_prediction
```

Artifact:

``` text
backend/models/artifacts/risk_model.pkl
```

Store at minimum:

``` text
model
feature_columns
target_column
decision_threshold
model_type
random_state
```

------------------------------------------------------------------------

# 30. Model Artifact Integrity

Every model artifact must be version-compatible with the feature schema
that created it.

At inference time:

1.  Load the artifact.
2.  Read stored feature columns.
3.  Verify every expected feature exists.
4.  Reorder columns to the stored order.
5.  Fail clearly if a feature is missing.
6.  Do not silently train/use a different feature set.

This prevents one of the most common pipeline failures: model artifacts
being applied to a differently ordered or differently engineered feature
matrix.

------------------------------------------------------------------------

# 31. Stage 10 --- SHAP Explainability

Entry point:

``` text
backend/explainability/shap_engine.py
```

Input:

``` text
risk_model.pkl
risk_training_features.csv
```

The SHAP engine must use the **exact feature columns stored in the
XGBoost artifact**.

Generate:

``` text
outputs/explainability/
```

Recommended outputs:

``` text
global_feature_importance
shap_values
top_contributions
beeswarm
violin
heatmap
waterfall
decision
dependence
interaction
```

At minimum, each wallet should have a ranked list of feature
contributions.

------------------------------------------------------------------------

# 32. Alert Generation

Entry point:

``` text
backend/alerts/alert_service.py
```

Inputs:

``` text
risk_model_predictions.csv
deterministic_scores.csv
SHAP contributions
```

The alert service should not expect obsolete names such as:

``` text
deterministic_risk_score
```

The canonical field is:

``` text
deterministic_score
```

------------------------------------------------------------------------

# 33. Alert Logic

The alert layer is an operational routing layer.

It should:

1.  Rank wallets.
2.  Combine model evidence.
3.  Attach deterministic evidence.
4.  Attach SHAP explanations.
5.  Assign an alert severity.
6.  Generate investigation-ready output.

Example operational routing:

``` text
risk_probability >= 0.80 → HIGH
risk_probability >= 0.50 → MEDIUM
otherwise                  → LOW / no alert
```

These values are **operational routing thresholds**, not claims that the
underlying statistical model has discovered universal Bitcoin-risk
boundaries.

Outputs:

``` text
outputs/alerts/
```

------------------------------------------------------------------------

# 34. Final Alert Contract

Each alert should contain enough information for an analyst to
understand:

``` text
wallet_id
risk_probability
risk_prediction
deterministic_score
deterministic_percentile
isolation_forest_anomaly_score
autoencoder_reconstruction_error
alert_severity
top_shap_features
evidence
```

Optional:

``` text
behavior_type
scenario information
network context
transaction summary
```

Ground truth should not be exposed as operational evidence unless the
endpoint is explicitly an evaluation/debug endpoint.

------------------------------------------------------------------------

# 35. FastAPI Integration

FastAPI is already implemented in:

``` text
backend/api_server.py
```

Do not replace the existing FastAPI implementation unnecessarily.

The API should act as the service boundary around the pipeline.

It should not contain the actual feature engineering/model
implementation.

Recommended architecture:

``` text
FastAPI endpoint
      ↓
pipeline/orchestrator.py
      ↓
stage functions
      ↓
artifacts
      ↓
response
```

------------------------------------------------------------------------

# 36. Pipeline Orchestrator

Create/maintain:

``` text
backend/pipeline/orchestrator.py
```

The orchestrator is the single canonical execution path.

Conceptually:

``` python
def run_pipeline(raw_dir, output_dir):
    dataset = ingest_dataset(raw_dir)

    features = build_features(dataset)

    graph = build_networkx_graph(dataset)
    graph_features = build_graph_features(graph)

    embeddings = train_or_load_graphsage(
        features,
        graph
    )

    isolation_scores = run_isolation_forest(
        features,
        graph_features,
        embeddings
    )

    autoencoder_scores = run_autoencoder(
        features,
        graph_features,
        embeddings
    )

    deterministic_scores = run_deterministic(
        features,
        graph_features
    )

    fused = build_risk_features(
        features,
        graph_features,
        embeddings,
        isolation_scores,
        autoencoder_scores,
        deterministic_scores
    )

    risk_model = train_or_load_xgboost(fused)

    shap_output = run_shap(risk_model, fused)

    alerts = generate_alerts(
        risk_model,
        deterministic_scores,
        shap_output
    )

    return alerts
```

The exact function names can differ, but the dependency ordering must
remain.

------------------------------------------------------------------------

# 37. Do Not Mix Training and Inference Silently

The pipeline should support two modes.

## Training / rebuild mode

``` text
ingest
→ feature generation
→ graph
→ GraphSAGE training
→ IF training
→ AE training
→ deterministic statistics
→ XGBoost training
→ SHAP
→ alerts
```

## Inference / analysis mode

``` text
ingest
→ feature generation
→ graph construction
→ GraphSAGE inference
→ IF inference
→ AE inference
→ deterministic scoring
→ XGBoost inference
→ SHAP
→ alerts
```

The inference path should load compatible artifacts instead of
retraining models every time.

------------------------------------------------------------------------

# 38. FastAPI Endpoints

The exact endpoint names should match the existing implementation, but
the API should logically expose at least:

### Health

``` text
GET /health
```

Returns service/pipeline status.

### Full analysis

``` text
POST /analyze
```

Runs the complete pipeline for the supplied dataset or configured
raw-data location.

### Wallet investigation

``` text
GET /wallet/{wallet_id}
```

Returns:

-   risk probability
-   risk prediction
-   deterministic score
-   anomaly signals
-   graph information
-   SHAP evidence
-   alert status

### Alerts

``` text
GET /alerts
```

Returns ranked alerts.

### Pipeline status

``` text
GET /pipeline/status
```

Returns artifact availability and latest pipeline state.

If the current FastAPI implementation uses different paths, preserve its
public contract and wire those endpoints to the orchestrator.

------------------------------------------------------------------------

# 39. FastAPI Request Flow

A full analysis request should follow:

``` text
POST /analyze
      ↓
validate request
      ↓
resolve raw data
      ↓
orchestrator
      ↓
ingestion
      ↓
feature engineering
      ↓
graph
      ↓
embeddings
      ↓
anomaly models
      ↓
deterministic risk
      ↓
risk fusion
      ↓
XGBoost
      ↓
SHAP
      ↓
alerts
      ↓
JSON response
```

FastAPI must not independently recreate any of these stages.

------------------------------------------------------------------------

# 40. Artifact Readiness Check

Before allowing inference, FastAPI/orchestrator should verify:

``` text
graphsage.pt
isolation_forest.pkl
isolation_forest_scaler.pkl
autoencoder.pt
autoencoder_scaler.pkl
risk_model.pkl
```

If a required artifact is missing, return a clear error such as:

``` text
Model artifacts are incomplete. Run training/rebuild pipeline first.
```

Do not silently retrain unless explicitly requested.

------------------------------------------------------------------------

# 41. Output Directory Contract

The pipeline should always write to:

``` text
outputs/features/
outputs/graphs/
outputs/models/
outputs/explainability/
outputs/alerts/
outputs/reports/
```

Do not use:

``` text
outputs/graph/
```

The canonical directory is:

``` text
outputs/graphs/
```

This path mismatch existed previously and must not return.

------------------------------------------------------------------------

# 42. Main Entry Point

Use module execution from the repository root:

``` bash
python -m backend.main
```

For individual stages:

``` bash
python -m backend.ingestion.ingestion_pipeline

python -m backend.feature_engineering.feature_pipeline

python -m backend.graph.graphsage

python -m backend.models.isolation_forest

python -m backend.models.autoencoder

python -m backend.risk.run_deterministic

python -m backend.models.risk_model

python -m backend.explainability.shap_engine

python -m backend.alerts.alert_service
```

Avoid executing files directly like:

``` bash
python backend/feature_engineering/feature_pipeline.py
```

when the source uses package imports such as:

``` python
from backend....
```

Module execution avoids many import-path failures.

------------------------------------------------------------------------

# 43. Required `__init__.py` Files

For predictable package imports, include:

``` text
backend/__init__.py
backend/ingestion/__init__.py
backend/feature_engineering/__init__.py
backend/graph/__init__.py
backend/models/__init__.py
backend/risk/__init__.py
backend/explainability/__init__.py
backend/alerts/__init__.py
backend/agent/__init__.py
backend/pipeline/__init__.py
```

------------------------------------------------------------------------

# 44. Dependency Requirements

The environment needs the packages required by the actual
implementation, including:

``` text
pandas
numpy
scipy
scikit-learn
networkx
torch
torch-geometric
xgboost
shap
joblib
matplotlib
fastapi
uvicorn
pydantic
lxml
```

The GraphSAGE stage specifically requires a working:

``` text
torch
torch-geometric
```

installation compatible with the environment's Python and PyTorch
versions.

------------------------------------------------------------------------

# 45. Full Clean-Run Procedure

Before declaring the pipeline working, delete stale generated outputs:

``` bash
rm -rf outputs/features/*
rm -rf outputs/graphs/*
rm -rf outputs/models/*
rm -rf outputs/explainability/*
rm -rf outputs/alerts/*
```

Do not delete the source dataset.

Then run the stages in order.

------------------------------------------------------------------------

# 46. Stage-by-Stage Acceptance Tests

## Test 1 --- Ingestion

Expected:

``` text
8 canonical CSV tables loaded
schema validation passes
reference validation passes
normalization passes
```

Failure here stops the pipeline.

------------------------------------------------------------------------

## Test 2 --- Feature Engineering

Expected:

``` text
500 wallets
52 feature columns
wallet_features.csv created
wallet_features_scaled.csv created
feature_names.json created
```

Verify:

``` text
wallet_id is unique
no accidental NetworkX metrics are present
```

------------------------------------------------------------------------

## Test 3 --- NetworkX

Expected:

``` text
500 nodes
10,975 edges
graph_features.csv created
graph_edges.csv created
```

Verify:

``` text
wallet_id is unique in graph_features.csv
```

------------------------------------------------------------------------

## Test 4 --- GraphSAGE

Expected:

``` text
500 wallet embeddings
32 dimensions
graphsage_embeddings.csv created
graphsage.pt created
```

Verify:

``` text
wallet_id + 32 embedding columns
```

------------------------------------------------------------------------

## Test 5 --- Isolation Forest

Expected:

``` text
500 wallet scores
isolation_forest_scores.csv created
.pkl artifacts created
```

------------------------------------------------------------------------

## Test 6 --- Autoencoder

Expected:

``` text
500 reconstruction scores
autoencoder_scores.csv created
.pt + scaler artifacts created
```

------------------------------------------------------------------------

## Test 7 --- Deterministic Risk

Expected:

``` text
500 deterministic scores
deterministic_statistics.json created
```

Verify no ground-truth column is used.

------------------------------------------------------------------------

## Test 8 --- Fusion

Expected:

``` text
base_fused_features.csv created
risk_training_features.csv created
```

Verify:

``` text
normal → 0
suspicious → 1
```

------------------------------------------------------------------------

## Test 9 --- XGBoost

Expected:

``` text
risk_model.pkl
risk_model_predictions.csv
risk_model_test_predictions.csv
risk_model_metrics.txt
```

Verify the model's stored feature list matches the training feature
matrix.

------------------------------------------------------------------------

## Test 10 --- SHAP

Expected:

``` text
SHAP values
global importance
top contributions
visualizations
```

Verify SHAP uses the exact XGBoost feature schema.

------------------------------------------------------------------------

## Test 11 --- Alerts

Expected:

``` text
ranked wallet alerts
risk probability
deterministic evidence
SHAP evidence
severity
```

------------------------------------------------------------------------

## Test 12 --- FastAPI

Start:

``` bash
uvicorn backend.api_server:app --reload
```

Then test:

``` text
GET /health
POST /analyze
GET /alerts
GET /wallet/{wallet_id}
```

The exact routes must follow the existing API contract if they differ.

------------------------------------------------------------------------

# 47. End-to-End Smoke Test

The final smoke test should verify:

``` text
raw data
  ↓
ingestion succeeds
  ↓
features created
  ↓
graph created
  ↓
GraphSAGE embeddings created
  ↓
IF scores created
  ↓
AE scores created
  ↓
deterministic scores created
  ↓
fusion created
  ↓
XGBoost predictions created
  ↓
SHAP explanations created
  ↓
alerts created
  ↓
FastAPI returns the final wallet result
```

The test should fail if any intermediate artifact is missing.

------------------------------------------------------------------------

# 48. Data/Artifact Join Integrity

Every major output must be joined using:

``` text
wallet_id
```

Expected cardinality:

``` text
wallet_features              1 row / wallet
graph_features               1 row / wallet
graphsage_embeddings         1 row / wallet
isolation_forest_scores      1 row / wallet
autoencoder_scores           1 row / wallet
deterministic_scores         1 row / wallet
risk_predictions             1 row / wallet
```

All should resolve to the same 500-wallet population for the current
dataset.

Before fusion, explicitly check:

``` text
duplicate wallet_id = 0
missing wallet_id = 0
unexpected wallet_id = 0
```

------------------------------------------------------------------------

# 49. Avoiding Circular Dependencies

The dependency graph must remain:

``` text
RAW DATA
   ↓
FEATURES
   ↓
NETWORKX
   ↓
GRAPHSAGE
   ↓
IF / AE
   ↓
DETERMINISTIC
   ↓
FUSION
   ↓
XGBOOST
   ↓
SHAP
   ↓
ALERTS
```

Specifically:

### Deterministic risk must not depend on:

``` text
IF
AE
XGBoost
SHAP
ground truth
```

### GraphSAGE must not depend on:

``` text
XGBoost
IF
AE
SHAP
ground truth
```

### IF/AE must not depend on:

``` text
ground truth
XGBoost
SHAP
```

### SHAP depends on:

``` text
XGBoost
```

### Alerts depend on:

``` text
risk outputs
deterministic evidence
SHAP evidence
```

------------------------------------------------------------------------

# 50. Handling Stale Artifacts

A major operational risk is mixing artifacts generated from different
feature schemas.

Each artifact should contain metadata such as:

``` text
feature_columns
feature_count
dataset/schema version
model version
random state
training timestamp
```

At runtime, compare the current feature schema against the artifact
schema.

If they do not match:

``` text
FAIL
```

rather than silently producing a prediction.

------------------------------------------------------------------------

# 51. Recommended Pipeline Manifest

Generate:

``` text
outputs/reports/pipeline_manifest.json
```

Example structure:

``` json
{
  "dataset": {
    "wallet_count": 500,
    "transaction_count": 10019
  },
  "features": {
    "count": 52
  },
  "graph": {
    "nodes": 500,
    "edges": 10975
  },
  "graphsage": {
    "embedding_dimension": 32
  },
  "models": {
    "isolation_forest": true,
    "autoencoder": true,
    "xgboost": true
  },
  "explainability": {
    "shap": true
  },
  "alerts": true
}
```

This makes debugging and API health checks much easier.

------------------------------------------------------------------------

# 52. Logging

Every stage should log:

``` text
stage name
input row counts
output row counts
feature counts
artifact paths
execution time
warnings
errors
```

Example:

``` text
[INGESTION] Loaded 8 tables
[FEATURES] Generated 500 × 52
[NETWORKX] Built 500 nodes / 10975 edges
[GRAPHSAGE] Generated 500 × 32 embeddings
[IF] Scored 500 wallets
[AE] Scored 500 wallets
[DETERMINISTIC] Scored 500 wallets
[FUSION] Generated training matrix
[XGBOOST] Generated 500 predictions
[SHAP] Generated explanations
[ALERTS] Generated ranked alerts
```

------------------------------------------------------------------------

# 53. Error Handling

Errors should identify:

1.  stage
2.  input artifact
3.  expected schema
4.  actual schema
5.  corrective action

Example:

``` text
Risk model failed:
Missing feature 'avg_network_tx_time_difference'
Expected schema version: ...
Current feature schema: ...
Re-run feature engineering and retrain compatible artifacts.
```

Avoid generic:

``` text
Something went wrong.
```

------------------------------------------------------------------------

# 54. API Response Design

A wallet investigation response should logically look like:

``` json
{
  "wallet_id": "wallet_x",
  "risk": {
    "probability": 0.87,
    "prediction": 1
  },
  "anomaly": {
    "isolation_forest": 0.91,
    "autoencoder": 0.78
  },
  "deterministic": {
    "score": 0.84,
    "percentile": 0.96
  },
  "graph": {
    "degree": 12,
    "pagerank": 0.004
  },
  "explanation": [
    {
      "feature": "feature_name",
      "contribution": 0.21,
      "direction": "increases_risk"
    }
  ],
  "alert": {
    "severity": "HIGH"
  }
}
```

The actual numeric values must come from generated artifacts; do not
hardcode example values into the API.

------------------------------------------------------------------------

# 55. AI Agent Integration

The repository currently contains the agent layer:

``` text
backend/agent/
```

but it should be treated as a later investigation interface unless the
implementation is completed.

The agent should consume existing structured evidence:

``` text
wallet profile
transaction summary
graph metrics
GraphSAGE embedding-derived context
Isolation Forest result
Autoencoder result
deterministic score
XGBoost probability
SHAP evidence
alert severity
```

The agent should **not independently calculate another risk score**.

Its role should be:

``` text
investigation
reasoning
evidence summarization
natural-language explanation
```

------------------------------------------------------------------------

# 56. What Must Be Removed or Disabled

There is an additional statistical `risk_engine.py` in the repository.

It currently represents another aggregation layer and expects fields
such as:

``` text
risk_model_probability
```

while the main XGBoost contract uses:

``` text
risk_probability
```

Do not allow this second risk engine to silently become part of the main
pipeline.

Either:

1.  remove it from the active pipeline, or
2.  explicitly redesign it as a documented alternative/experimental risk
    aggregation component.

The main production flow should have one clear risk-model path.

------------------------------------------------------------------------

# 57. Final Canonical Data Contracts

## Features

``` text
outputs/features/wallet_features.csv
outputs/features/wallet_features_scaled.csv
outputs/features/feature_names.json
```

## Graph

``` text
outputs/graphs/graph_edges.csv
outputs/graphs/graph_features.csv
```

## GraphSAGE

``` text
outputs/graphs/graphsage_embeddings.csv
backend/models/artifacts/graphsage.pt
```

## Isolation Forest

``` text
outputs/models/isolation_forest_scores.csv
backend/models/artifacts/isolation_forest.pkl
backend/models/artifacts/isolation_forest_scaler.pkl
```

## Autoencoder

``` text
outputs/models/autoencoder_scores.csv
backend/models/artifacts/autoencoder.pt
backend/models/artifacts/autoencoder_scaler.pkl
```

## Deterministic

``` text
outputs/models/deterministic_scores.csv
outputs/models/deterministic_statistics.json
```

## Fusion

``` text
outputs/models/base_fused_features.csv
outputs/models/risk_training_features.csv
```

## XGBoost

``` text
outputs/models/risk_model_predictions.csv
outputs/models/risk_model_test_predictions.csv
outputs/models/risk_model_metrics.txt
backend/models/artifacts/risk_model.pkl
```

## Explainability

``` text
outputs/explainability/*
```

## Alerts

``` text
outputs/alerts/*
```

## Pipeline report

``` text
outputs/reports/pipeline_manifest.json
```

------------------------------------------------------------------------

# 58. Final Definition of Done

The pipeline is considered fully wired only when all of the following
are true:

-   [ ] FastAPI remains the API entry layer.
-   [ ] FastAPI delegates analysis to the orchestrator.
-   [ ] Orchestrator calls ingestion first.
-   [ ] Feature engineering consumes ingestion output.
-   [ ] No raw `pd.read_csv()` bypass remains inside feature
    engineering.
-   [ ] Ingestion validates schemas and references.
-   [ ] Normalization occurs once at the ingestion boundary.
-   [ ] Transaction features calculate actual counterparties.
-   [ ] Temporal features are generated.
-   [ ] Network features are generated.
-   [ ] Correlation features are generated.
-   [ ] Exactly the expected wallet feature schema is persisted.
-   [ ] NetworkX graph is created from transaction relationships.
-   [ ] NetworkX graph metrics are generated separately.
-   [ ] GraphSAGE uses base wallet features and graph edges.
-   [ ] GraphSAGE does not consume NetworkX metrics as node features.
-   [ ] GraphSAGE produces 32-D embeddings.
-   [ ] GraphSAGE artifact stores its feature schema.
-   [ ] Isolation Forest consumes compatible merged features.
-   [ ] Autoencoder consumes compatible merged features.
-   [ ] Both anomaly models save inference artifacts.
-   [ ] Deterministic risk uses only statistical wallet/graph features.
-   [ ] Deterministic risk does not consume ground truth.
-   [ ] Deterministic risk does not consume IF/AE/XGBoost outputs.
-   [ ] Fusion joins all outputs using `wallet_id`.
-   [ ] Ground-truth strings map explicitly to `0/1`.
-   [ ] XGBoost trains on the fused supervised dataset.
-   [ ] XGBoost stores its exact feature schema.
-   [ ] SHAP uses the exact XGBoost schema.
-   [ ] Alerts consume actual risk/deterministic/SHAP outputs.
-   [ ] Alert thresholds are documented as operational routing
    thresholds.
-   [ ] Stale artifact/schema mismatches cause explicit failures.
-   [ ] Full clean-run succeeds from raw data.
-   [ ] FastAPI can return the final wallet investigation result.
-   [ ] No stage silently retrains models during inference.
-   [ ] No component creates an independent competing risk score.
-   [ ] `__pycache__` and generated artifacts that should not be
    committed are ignored.

------------------------------------------------------------------------

# 59. One-Command Target

The final developer experience should be as close as possible to:

``` bash
python -m backend.pipeline.orchestrator
```

for a complete rebuild/training run, and:

``` bash
uvicorn backend.api_server:app --reload
```

for API serving.

The API then becomes the user-facing interface while the orchestrator
remains the single source of truth for the complete forensic pipeline.

------------------------------------------------------------------------

# 60. Critical Principle

The most important implementation rule is:

> **Do not make each module merely runnable in isolation. Make every
> module consume the exact artifact and schema produced by the previous
> module.**

A successful pipeline is not:

``` text
Ingestion works
+ Features work
+ Graph works
+ ML works
+ API works
```

A successful pipeline is:

``` text
Ingestion output
      ↓
exactly matches
      ↓
Feature input
      ↓
Feature output
      ↓
exactly matches
      ↓
Graph / model inputs
      ↓
...
      ↓
Alert output
      ↓
exactly matches
      ↓
FastAPI response
```

That is the standard the final implementation should be tested against.
