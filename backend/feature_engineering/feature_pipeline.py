from pathlib import Path
import ast
import json

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler

from backend.feature_engineering.transaction_features import (
    build_transaction_features,
)

from backend.feature_engineering.temporal_features import (
    build_temporal_features,
)

from backend.feature_engineering.network_features import (
    build_network_features,
)

from backend.feature_engineering.correlation_features import (
    build_correlation_features,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

DATASET_DIR = (
    PROJECT_ROOT / "data"
)

RAW_DIR = (
    DATASET_DIR / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "features"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "artifacts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data():

    files = {
        "wallets":
            RAW_DIR / "wallets.csv",

        "transactions":
            RAW_DIR / "transactions.csv",

        "transaction_inputs":
            RAW_DIR / "transaction_inputs.csv",

        "transaction_outputs":
            RAW_DIR / "transaction_outputs.csv",

        "network_observations":
            RAW_DIR / "network_observations.csv",

        "ip_metadata":
            RAW_DIR / "ip_metadata.csv",
    }

    data = {}

    for name, path in files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Required dataset file not found:\n"
                f"{path}"
            )

        data[name] = pd.read_csv(path)

    return data


# ============================================================
# LIST PARSER
# ============================================================

def parse_list_column(
    value,
):

    if value is None:
        return []

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(parsed, list):
            return parsed

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    return []


# ============================================================
# PREPARE TRANSACTIONS
# ============================================================

def prepare_transactions(
    transactions,
):

    transactions = transactions.copy()

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
                .apply(parse_list_column)
            )

    transactions["timestamp"] = (
        pd.to_datetime(
            transactions["timestamp"],
            errors="coerce",
            utc=True,
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
                    errors="coerce",
                )
            )

    return transactions


# ============================================================
# MAIN FEATURE PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    data = load_raw_data()

    wallets = data["wallets"].copy()

    wallets["wallet_id"] = (
        wallets["wallet_id"]
        .astype(str)
    )

    transactions = prepare_transactions(
        data["transactions"]
    )

    transaction_inputs = (
        data["transaction_inputs"]
        .copy()
    )

    transaction_outputs = (
        data["transaction_outputs"]
        .copy()
    )

    network_observations = (
        data["network_observations"]
        .copy()
    )

    ip_metadata = (
        data["ip_metadata"]
        .copy()
    )

    # --------------------------------------------------------
    # Transaction
    # --------------------------------------------------------

    transaction_features = (
        build_transaction_features(
            wallets,
            transactions,
            transaction_inputs,
            transaction_outputs,
        )
    )

    # --------------------------------------------------------
    # Temporal
    # --------------------------------------------------------

    temporal_features = (
        build_temporal_features(
            transactions,
            wallets,
        )
    )

    # --------------------------------------------------------
    # Network / GeoIP
    # --------------------------------------------------------

    network_features = (
        build_network_features(
            network_observations,
            ip_metadata,
        )
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlation_features = (
        build_correlation_features(
            transactions,
            network_observations,
        )
    )

    # --------------------------------------------------------
    # Base wallet table
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "wallet_id":
                wallets["wallet_id"]
        }
    )

    result["wallet_id"] = (
        result["wallet_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Merge feature groups
    # --------------------------------------------------------

    for frame in [
        transaction_features,
        temporal_features,
        network_features,
        correlation_features,
    ]:

        if frame is None or frame.empty:
            continue

        frame = frame.copy()

        frame["wallet_id"] = (
            frame["wallet_id"]
            .astype(str)
        )

        if frame["wallet_id"].duplicated().any():

            raise ValueError(
                "Feature group contains duplicate "
                "wallet_id values."
            )

        result = result.merge(
            frame,
            on="wallet_id",
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in result.columns
        if column != "wallet_id"
    ]

    for column in feature_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result[feature_columns] = (
        result[feature_columns]
        .replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Remove constant features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in feature_columns
        if result[column].nunique() > 1
    ]

    result = result[
        ["wallet_id"]
        + feature_columns
    ]

    # --------------------------------------------------------
    # Save raw engineered features
    # --------------------------------------------------------

    feature_file = (
        OUTPUT_DIR
        / "wallet_features.csv"
    )

    result.to_csv(
        feature_file,
        index=False,
    )

    # --------------------------------------------------------
    # Scaled features
    # --------------------------------------------------------

    scaler = StandardScaler()

    scaled_values = (
        scaler.fit_transform(
            result[feature_columns]
        )
    )

    scaled = pd.DataFrame(
        scaled_values,
        columns=feature_columns,
    )

    scaled.insert(
        0,
        "wallet_id",
        result["wallet_id"],
    )

    scaled_file = (
        OUTPUT_DIR
        / "wallet_features_scaled.csv"
    )

    scaled.to_csv(
        scaled_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    scaler_file = (
        ARTIFACT_DIR
        / "feature_scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_file,
    )

    # --------------------------------------------------------
    # Save feature names
    # --------------------------------------------------------

    feature_names_file = (
        OUTPUT_DIR
        / "feature_names.json"
    )

    with open(
        feature_names_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            feature_columns,
            file,
            indent=2,
        )

    print(
        f"Wallets: {len(result)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Saved: {feature_file}"
    )

    print(
        f"Saved: {scaled_file}"
    )

    print(
        f"Saved: {scaler_file}"
    )

    return result


if __name__ == "__main__":
    main()