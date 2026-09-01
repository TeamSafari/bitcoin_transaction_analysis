from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs"
)

FEATURE_DIR = (
    OUTPUT_DIR / "features"
)

GRAPH_DIR = (
    OUTPUT_DIR / "graphs"
)

MODEL_DIR = (
    OUTPUT_DIR / "models"
)

GROUND_TRUTH_DIR = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
)


FEATURE_FILE = (
    FEATURE_DIR
    / "wallet_features.csv"
)

GRAPH_FILE = (
    GRAPH_DIR
    / "graph_features.csv"
)

GRAPHSAGE_FILE = (
    GRAPH_DIR
    / "graphsage_embeddings.csv"
)

IF_FILE = (
    MODEL_DIR
    / "isolation_forest_scores.csv"
)

AE_FILE = (
    MODEL_DIR
    / "autoencoder_scores.csv"
)

DETERMINISTIC_FILE = (
    MODEL_DIR
    / "deterministic_scores.csv"
)

GROUND_TRUTH_FILE = (
    GROUND_TRUTH_DIR
    / "wallet_ground_truth.csv"
)

BASE_OUTPUT = (
    MODEL_DIR
    / "base_fused_features.csv"
)

TRAINING_OUTPUT = (
    MODEL_DIR
    / "risk_training_features.csv"
)


# ============================================================
# LOAD
# ============================================================

def load_wallet_csv(
    path: Path,
    name: str,
):

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    if df.empty:

        raise ValueError(
            f"{name} is empty."
        )

    if "wallet_id" not in df.columns:

        raise ValueError(
            f"{name} must contain wallet_id."
        )

    df["wallet_id"] = (
        df["wallet_id"]
        .astype(str)
    )

    if df["wallet_id"].duplicated().any():

        raise ValueError(
            f"{name} contains duplicate wallet_id."
        )

    return df


# ============================================================
# KEEP NUMERIC FEATURES
# ============================================================

def numeric_only(
    df,
):

    result = pd.DataFrame(
        {
            "wallet_id":
                df["wallet_id"]
        }
    )

    for column in df.columns:

        if column == "wallet_id":
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric.notna().sum() > 0:

            result[column] = numeric

    return result


# ============================================================
# MERGE
# ============================================================

def merge_layer(
    base,
    layer,
    source_name,
):

    layer = numeric_only(
        layer
    )

    rename = {}

    for column in layer.columns:

        if (
            column != "wallet_id"
            and column in base.columns
        ):

            rename[column] = (
                f"{source_name}_{column}"
            )

    if rename:

        layer = layer.rename(
            columns=rename
        )

    return base.merge(
        layer,
        on="wallet_id",
        how="inner",
        validate="one_to_one",
    )


# ============================================================
# BUILD BASE FEATURES
# ============================================================

def build_base_features():

    base = numeric_only(
        load_wallet_csv(
            FEATURE_FILE,
            "wallet_features.csv",
        )
    )

    graph = load_wallet_csv(
        GRAPH_FILE,
        "graph_features.csv",
    )

    sage = load_wallet_csv(
        GRAPHSAGE_FILE,
        "graphsage_embeddings.csv",
    )

    base = merge_layer(
        base,
        graph,
        "networkx",
    )

    base = merge_layer(
        base,
        sage,
        "graphsage",
    )

    if base.empty:

        raise ValueError(
            "Base feature fusion produced "
            "zero wallets."
        )

    return base


# ============================================================
# ADD DETECTOR OUTPUTS
# ============================================================

def add_detector_outputs(
    base,
):

    result = base.copy()

    layers = [
        (
            IF_FILE,
            "Isolation Forest",
            "isolation_forest",
        ),
        (
            AE_FILE,
            "Autoencoder",
            "autoencoder",
        ),
        (
            DETERMINISTIC_FILE,
            "Deterministic risk",
            "deterministic",
        ),
    ]

    for path, name, prefix in layers:

        layer = load_wallet_csv(
            path,
            name,
        )

        result = merge_layer(
            result,
            layer,
            prefix,
        )

    return result


# ============================================================
# ADD GROUND TRUTH FOR TRAINING
# ============================================================

def add_training_label(
    features,
):

    truth = load_wallet_csv(
        GROUND_TRUTH_FILE,
        "wallet_ground_truth.csv",
    )

    if "label" not in truth.columns:

        raise ValueError(
            "wallet_ground_truth.csv "
            "must contain label."
        )

    truth["label"] = pd.to_numeric(
        truth["label"],
        errors="coerce",
    )

    if truth["label"].isna().any():

        raise ValueError(
            "Ground-truth label contains "
            "missing/non-numeric values."
        )

    if not set(
        truth["label"].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Ground-truth label must be 0/1."
        )

    training = features.merge(
        truth[
            [
                "wallet_id",
                "label",
            ]
        ],
        on="wallet_id",
        how="inner",
        validate="one_to_one",
    )

    if training.empty:

        raise ValueError(
            "Ground-truth fusion produced "
            "zero rows."
        )

    return training


# ============================================================
# CLEAN
# ============================================================

def clean_features(
    df,
):

    result = df.copy()

    feature_columns = [
        column
        for column in result.columns
        if column
        not in {
            "wallet_id",
            "label",
        }
    ]

    for column in feature_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result[feature_columns] = (
        result[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    for column in feature_columns:

        median = (
            result[column]
            .median()
        )

        if pd.isna(median):
            median = 0.0

        result[column] = (
            result[column]
            .fillna(median)
        )

    # Remove constant features.
    variable_columns = [
        column
        for column in feature_columns
        if result[column].nunique() > 1
    ]

    result = result[
        ["wallet_id"]
        + variable_columns
        + (
            ["label"]
            if "label" in result.columns
            else []
        )
    ]

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RISK FEATURE FUSION")
    print("=" * 60)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Base representation
    # --------------------------------------------------------

    base = build_base_features()

    base.to_csv(
        BASE_OUTPUT,
        index=False,
    )

    print(
        f"Base features: "
        f"{base.shape}"
    )

    # --------------------------------------------------------
    # Add anomaly/statistical outputs
    # --------------------------------------------------------

    fused = add_detector_outputs(
        base
    )

    # --------------------------------------------------------
    # Ground truth is added ONLY here,
    # for supervised training.
    # --------------------------------------------------------

    training = add_training_label(
        fused
    )

    training = clean_features(
        training
    )

    training.to_csv(
        TRAINING_OUTPUT,
        index=False,
    )

    print(
        f"Training dataset: "
        f"{training.shape}"
    )

    print(
        f"Label distribution:\n"
        f"{training['label'].value_counts()}"
    )

    print(
        f"\nSaved:\n"
        f"{TRAINING_OUTPUT}"
    )

    return training


if __name__ == "__main__":
    main()