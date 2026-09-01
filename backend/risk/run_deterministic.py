from pathlib import Path
import pandas as pd

from deterministic_engine import DeterministicRiskEngine


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FEATURE_DIR = PROJECT_ROOT / "outputs" / "features"
GRAPH_DIR = PROJECT_ROOT / "outputs" / "graph"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

WALLET_FEATURES_FILE = (
    FEATURE_DIR / "wallet_features.csv"
)

GRAPH_FEATURES_FILE = (
    GRAPH_DIR / "graph_features.csv"
)

GRAPHSAGE_FILE = (
    GRAPH_DIR / "graphsage_embeddings.csv"
)

ISOLATION_FOREST_FILE = (
    MODEL_DIR / "isolation_forest_scores.csv"
)

AUTOENCODER_FILE = (
    MODEL_DIR / "autoencoder_scores.csv"
)


# ============================================================
# LOAD WALLET FEATURES
# ============================================================

if not WALLET_FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Missing wallet feature file:\n"
        f"{WALLET_FEATURES_FILE}"
    )

data = pd.read_csv(
    WALLET_FEATURES_FILE
)


# ============================================================
# MERGE GRAPH FEATURES
# ============================================================

if GRAPH_FEATURES_FILE.exists():

    graph_features = pd.read_csv(
        GRAPH_FEATURES_FILE
    )

    if "wallet_id" not in graph_features.columns:
        raise ValueError(
            "graph_features.csv must contain "
            "'wallet_id'"
        )

    data = data.merge(
        graph_features,
        on="wallet_id",
        how="left",
        suffixes=("", "_graph")
    )


# ============================================================
# MERGE GRAPHSAGE EMBEDDINGS
# ============================================================

if GRAPHSAGE_FILE.exists():

    embeddings = pd.read_csv(
        GRAPHSAGE_FILE
    )

    if "wallet_id" not in embeddings.columns:
        raise ValueError(
            "graphsage_embeddings.csv must contain "
            "'wallet_id'"
        )

    data = data.merge(
        embeddings,
        on="wallet_id",
        how="left"
    )


# ============================================================
# MERGE ISOLATION FOREST SCORES
# ============================================================

if ISOLATION_FOREST_FILE.exists():

    isolation_scores = pd.read_csv(
        ISOLATION_FOREST_FILE
    )

    if "wallet_id" not in isolation_scores.columns:
        raise ValueError(
            "isolation_forest_scores.csv must contain "
            "'wallet_id'"
        )

    data = data.merge(
        isolation_scores,
        on="wallet_id",
        how="left"
    )


# ============================================================
# MERGE AUTOENCODER SCORES
# ============================================================

if AUTOENCODER_FILE.exists():

    autoencoder_scores = pd.read_csv(
        AUTOENCODER_FILE
    )

    if "wallet_id" not in autoencoder_scores.columns:
        raise ValueError(
            "autoencoder_scores.csv must contain "
            "'wallet_id'"
        )

    data = data.merge(
        autoencoder_scores,
        on="wallet_id",
        how="left"
    )


# ============================================================
# VALIDATION
# ============================================================

if "wallet_id" not in data.columns:
    raise ValueError(
        "Final input must contain wallet_id"
    )

data["wallet_id"] = (
    data["wallet_id"]
    .astype(str)
)


# ============================================================
# RUN DETERMINISTIC ENGINE
# ============================================================

engine = DeterministicRiskEngine(
    tail_percentile=0.95
)

deterministic_results = engine.run(
    data
)


# ============================================================
# SAVE OUTPUT
# ============================================================

output_file = (
    MODEL_DIR /
    "deterministic_scores.csv"
)

deterministic_results.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("DETERMINISTIC ENGINE COMPLETE")
print("=" * 60)

print(
    f"Wallets processed: "
    f"{len(deterministic_results)}"
)

print(
    f"Wallets with signals: "
    f"{(deterministic_results['deterministic_score'] > 0).sum()}"
)

print(
    f"Output: {output_file}"
)