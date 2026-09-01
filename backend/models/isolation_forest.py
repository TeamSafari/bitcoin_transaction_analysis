from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is:
# backend/models/isolation_forest.py
#
# parents[0] = models
# parents[1] = backend
# parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

FEATURE_DIR = OUTPUT_DIR / "features"

GRAPH_DIR = OUTPUT_DIR / "graphs"

MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "artifacts"
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


# ============================================================
# OUTPUT FILES
# ============================================================

SCORES_FILE = (
    MODEL_OUTPUT_DIR
    / "isolation_forest_scores.csv"
)

MODEL_FILE = (
    ARTIFACT_DIR
    / "isolation_forest.pkl"
)

SCALER_FILE = (
    ARTIFACT_DIR
    / "isolation_forest_scaler.pkl"
)


# ============================================================
# PARAMETERS
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(
    path: Path,
    name: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{name} is empty."
        )

    if "wallet_id" not in df.columns:
        raise ValueError(
            f"{name} must contain "
            "'wallet_id'."
        )

    df["wallet_id"] = (
        df["wallet_id"]
        .astype(str)
    )

    if df["wallet_id"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate "
            "wallet_id values."
        )

    return df


# ============================================================
# EXTRACT NUMERIC FEATURES
# ============================================================

def extract_numeric_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

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

        # Keep columns containing at least
        # one valid numeric value.
        if numeric.notna().sum() > 0:

            result[column] = numeric

    return result


# ============================================================
# FEATURE FUSION
# ============================================================

def build_fused_features():

    print("=" * 60)
    print("LOADING MODEL INPUTS")
    print("=" * 60)

    wallet = load_csv(
        WALLET_FEATURES_FILE,
        "wallet_features.csv",
    )

    graph = load_csv(
        GRAPH_FEATURES_FILE,
        "graph_features.csv",
    )

    graphsage = load_csv(
        GRAPHSAGE_FILE,
        "graphsage_embeddings.csv",
    )

    print(
        f"Wallet features : {len(wallet)} rows"
    )

    print(
        f"Graph features  : {len(graph)} rows"
    )

    print(
        f"GraphSAGE       : {len(graphsage)} rows"
    )

    wallet = extract_numeric_features(
        wallet
    )

    graph = extract_numeric_features(
        graph
    )

    graphsage = extract_numeric_features(
        graphsage
    )

    # --------------------------------------------------------
    # Fuse wallet + graph
    # --------------------------------------------------------

    fused = wallet.merge(
        graph,
        on="wallet_id",
        how="inner",
        suffixes=(
            "",
            "_graph",
        ),
    )

    # --------------------------------------------------------
    # Add GraphSAGE embeddings
    # --------------------------------------------------------

    fused = fused.merge(
        graphsage,
        on="wallet_id",
        how="inner",
        suffixes=(
            "",
            "_graphsage",
        ),
    )

    if fused.empty:
        raise ValueError(
            "Feature fusion produced zero rows. "
            "Check wallet_id consistency."
        )

    # Remove accidental duplicate column names.
    fused = fused.loc[
        :,
        ~fused.columns.duplicated()
    ]

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in fused.columns
        if column != "wallet_id"
    ]

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    fused[feature_columns] = (
        fused[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    for column in feature_columns:

        median = (
            fused[column]
            .median()
        )

        if pd.isna(median):
            median = 0.0

        fused[column] = (
            fused[column]
            .fillna(median)
        )

    # --------------------------------------------------------
    # Remove constant features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in feature_columns
        if fused[column].nunique() > 1
    ]

    if not feature_columns:
        raise ValueError(
            "No usable features remain."
        )

    fused = fused[
        ["wallet_id"]
        + feature_columns
    ]

    print(
        f"\nFused wallets   : {len(fused)}"
    )

    print(
        f"Fused features  : {len(feature_columns)}"
    )

    return fused, feature_columns


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

def train_isolation_forest(
    fused: pd.DataFrame,
    feature_columns: list[str],
):

    X = fused[
        feature_columns
    ].values

    # --------------------------------------------------------
    # Scaling
    #
    # Not mathematically required by Isolation Forest,
    # but retained for a consistent preprocessing pipeline.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_scaled
    )

    # sklearn decision_function:
    #
    # higher = more normal
    #
    # Negating gives:
    #
    # higher = more anomalous

    raw_anomaly_score = (
        -model.decision_function(
            X_scaled
        )
    )

    prediction = model.predict(
        X_scaled
    )

    anomaly_flag = (
        prediction == -1
    ).astype(int)

    # --------------------------------------------------------
    # Batch-relative normalized score
    #
    # This is for presentation/ranking only.
    # The raw score remains available.
    # --------------------------------------------------------

    minimum = raw_anomaly_score.min()

    maximum = raw_anomaly_score.max()

    if maximum > minimum:

        normalized_score = (
            raw_anomaly_score
            - minimum
        ) / (
            maximum
            - minimum
        )

    else:

        normalized_score = np.zeros(
            len(raw_anomaly_score)
        )

    return (
        model,
        scaler,
        raw_anomaly_score,
        normalized_score,
        anomaly_flag,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    fused: pd.DataFrame,
    feature_columns: list[str],
    model,
    scaler,
    raw_anomaly_score,
    normalized_score,
    anomaly_flag,
):

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    output = pd.DataFrame(
        {
            "wallet_id":
                fused["wallet_id"],

            "isolation_forest_raw_score":
                raw_anomaly_score,

            "isolation_forest_anomaly_score":
                normalized_score,

            "isolation_forest_flag":
                anomaly_flag,
        }
    )

    output.to_csv(
        SCORES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Model + metadata
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": model,
            "feature_columns":
                feature_columns,
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Exact scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    print("\nSaved:")
    print(
        f"Scores : {SCORES_FILE}"
    )
    print(
        f"Model  : {MODEL_FILE}"
    )
    print(
        f"Scaler : {SCALER_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 60)
    print("# ISOLATION FOREST")
    print("#" * 60)

    (
        fused,
        feature_columns,
    ) = build_fused_features()

    (
        model,
        scaler,
        raw_score,
        normalized_score,
        anomaly_flag,
    ) = train_isolation_forest(
        fused,
        feature_columns,
    )

    save_results(
        fused=fused,
        feature_columns=feature_columns,
        model=model,
        scaler=scaler,
        raw_anomaly_score=raw_score,
        normalized_score=normalized_score,
        anomaly_flag=anomaly_flag,
    )

    print("\n" + "=" * 60)
    print("ISOLATION FOREST COMPLETE")
    print("=" * 60)

    print(
        f"Wallets processed : {len(fused)}"
    )

    print(
        f"Anomalies flagged : "
        f"{anomaly_flag.sum()}"
    )


if __name__ == "__main__":
    main()