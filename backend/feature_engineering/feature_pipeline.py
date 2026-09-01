from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from transaction_features import (
    build_transaction_features
)

from temporal_features import (
    build_temporal_features
)

from network_features import (
    build_network_features
)

from correlation_features import (
    build_correlation_features
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DATASET_DIR = PROJECT_ROOT / "data"

RAW_DIR = (
    DATASET_DIR /
    "raw"
)

GEOIP_DIR = (
    DATASET_DIR /
    "geoip"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "features"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "backend" /
    "models" /
    "artifacts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data():

    wallets = pd.read_csv(
        RAW_DIR /
        "wallets.csv"
    )

    transactions = pd.read_csv(
        RAW_DIR /
        "transactions.csv"
    )

    transaction_inputs = pd.read_csv(
        RAW_DIR /
        "transaction_inputs.csv"
    )

    transaction_outputs = pd.read_csv(
        RAW_DIR /
        "transaction_outputs.csv"
    )

    network_observations = pd.read_csv(
        RAW_DIR /
        "network_observations.csv"
    )

    ip_metadata = pd.read_csv(
        RAW_DIR /
        "ip_metadata.csv"
    )

    return {
        "wallets": wallets,
        "transactions": transactions,
        "transaction_inputs":
            transaction_inputs,
        "transaction_outputs":
            transaction_outputs,
        "network_observations":
            network_observations,
        "ip_metadata":
            ip_metadata,
    }


# ============================================================
# PARSE LIST COLUMNS
# ============================================================

def parse_list_column(
    value
):

    if pd.isna(value):

        return []

    if isinstance(value, list):

        return value

    value = str(value).strip()

    if not value:

        return []

    # Handle CSV representation:
    # ['W0001', 'W0002']

    try:

        import ast

        parsed = ast.literal_eval(
            value
        )

        if isinstance(
            parsed,
            list
        ):

            return parsed

    except (
        ValueError,
        SyntaxError
    ):

        pass

    return []


def prepare_transactions(
    transactions
):

    transactions = (
        transactions.copy()
    )

    list_columns = [
        "input_addresses",
        "output_addresses",
        "input_amounts_sats",
        "output_amounts_sats",
    ]

    for column in list_columns:

        if column in transactions.columns:

            transactions[column] = (
                transactions[column]
                .apply(
                    parse_list_column
                )
            )

    transactions["timestamp"] = (
        pd.to_datetime(
            transactions["timestamp"],
            errors="coerce"
        )
    )

    numeric_columns = [
        "total_input_sats",
        "total_output_sats",
        "fee_sats",
    ]

    for column in numeric_columns:

        if column in transactions.columns:

            transactions[column] = (
                pd.to_numeric(
                    transactions[column],
                    errors="coerce"
                )
            )

    return transactions


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features():

    print(
        "Loading raw dataset..."
    )

    data = load_raw_data()

    wallets = data["wallets"]

    transactions = (
        prepare_transactions(
            data["transactions"]
        )
    )

    transaction_inputs = (
        data["transaction_inputs"]
    )

    transaction_outputs = (
        data["transaction_outputs"]
    )

    network_observations = (
        data["network_observations"]
    )

    ip_metadata = (
        data["ip_metadata"]
    )

    # --------------------------------------------------------
    # Transaction features
    # --------------------------------------------------------

    print(
        "Building transaction features..."
    )

    transaction_features = (
        build_transaction_features(
            wallets,
            transactions,
            transaction_inputs,
            transaction_outputs,
        )
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    print(
        "Building temporal features..."
    )

    temporal_features = (
        build_temporal_features(
            transactions,
            wallets,
        )
    )

    # --------------------------------------------------------
    # Network features
    # --------------------------------------------------------

    print(
        "Building network features..."
    )

    network_features = (
        build_network_features(
            network_observations,
            ip_metadata,
        )
    )

    # --------------------------------------------------------
    # Correlation features
    # --------------------------------------------------------

    print(
        "Building correlation features..."
    )

    correlation_features = (
        build_correlation_features(
            transactions,
            network_observations,
        )
    )

    # --------------------------------------------------------
    # Start with every wallet
    # --------------------------------------------------------

    result = pd.DataFrame({
        "wallet_id":
            wallets["wallet_id"]
            .astype(str)
    })

    # --------------------------------------------------------
    # Merge all feature groups
    # --------------------------------------------------------

    feature_frames = [
        transaction_features,
        temporal_features,
        network_features,
        correlation_features,
    ]

    for frame in feature_frames:

        if frame.empty:

            continue

        frame["wallet_id"] = (
            frame["wallet_id"]
            .astype(str)
        )

        result = result.merge(
            frame,
            on="wallet_id",
            how="left",
            suffixes=(
                "",
                "_duplicate"
            ),
        )

    # --------------------------------------------------------
    # Remove accidental duplicate columns
    # --------------------------------------------------------

    duplicate_columns = [
        col
        for col in result.columns
        if col.endswith(
            "_duplicate"
        )
    ]

    if duplicate_columns:

        result = result.drop(
            columns=duplicate_columns
        )

    # --------------------------------------------------------
    # Fill numeric missing values
    # --------------------------------------------------------

    numeric_columns = [
        col
        for col in result.columns
        if col != "wallet_id"
    ]

    result[numeric_columns] = (
        result[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .replace(
            [float("inf"), float("-inf")],
            0
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Save unscaled features
    # --------------------------------------------------------

    feature_file = (
        OUTPUT_DIR /
        "wallet_features.csv"
    )

    result.to_csv(
        feature_file,
        index=False
    )

    print(
        f"Saved: {feature_file}"
    )

    # --------------------------------------------------------
    # ML feature matrix
    # --------------------------------------------------------

    ml_features = result[
        [
            col
            for col in result.columns
            if col != "wallet_id"
        ]
    ].copy()

    # Feature names are preserved
    feature_names = (
        ml_features.columns.tolist()
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaled_values = (
        scaler.fit_transform(
            ml_features
        )
    )

    scaled = pd.DataFrame(
        scaled_values,
        columns=feature_names
    )

    scaled.insert(
        0,
        "wallet_id",
        result["wallet_id"]
    )

    scaled_file = (
        OUTPUT_DIR /
        "wallet_features_scaled.csv"
    )

    scaled.to_csv(
        scaled_file,
        index=False
    )

    print(
        f"Saved: {scaled_file}"
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    scaler_file = (
        MODEL_DIR /
        "feature_scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    print(
        f"Saved: {scaler_file}"
    )

    # --------------------------------------------------------
    # Save feature names
    # --------------------------------------------------------

    import json

    feature_names_file = (
        OUTPUT_DIR /
        "feature_names.json"
    )

    with open(
        feature_names_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_names,
            f,
            indent=2
        )

    print(
        f"Saved: {feature_names_file}"
    )

    return result


if __name__ == "__main__":

    build_features()