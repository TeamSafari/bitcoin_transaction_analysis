import pandas as pd
import numpy as np


def build_temporal_features(
    transactions: pd.DataFrame,
    wallets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build wallet-level temporal behavior features.
    """

    tx = transactions.copy()

    tx["txid"] = tx["txid"].astype(str)

    tx["timestamp"] = pd.to_datetime(
        tx["timestamp"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Extract wallet from input/output addresses
    # ---------------------------------------------------------

    rows = []

    for _, row in tx.iterrows():

        timestamp = row["timestamp"]

        if pd.isna(timestamp):
            continue

        for wallet in row.get(
            "input_addresses",
            []
        ):

            rows.append(
                {
                    "wallet_id": str(wallet),
                    "timestamp": timestamp,
                }
            )

        for wallet in row.get(
            "output_addresses",
            []
        ):

            rows.append(
                {
                    "wallet_id": str(wallet),
                    "timestamp": timestamp,
                }
            )

    wallet_tx = pd.DataFrame(rows)

    if wallet_tx.empty:

        return pd.DataFrame({
            "wallet_id":
                wallets["wallet_id"].astype(str)
        })

    wallet_tx = (
        wallet_tx
        .drop_duplicates()
        .sort_values(
            [
                "wallet_id",
                "timestamp"
            ]
        )
    )

    # ---------------------------------------------------------
    # Inter-transaction times
    # ---------------------------------------------------------

    wallet_tx["inter_tx_seconds"] = (
        wallet_tx
        .groupby("wallet_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    temporal = (
        wallet_tx
        .groupby("wallet_id")
        .agg(
            avg_inter_tx_seconds=(
                "inter_tx_seconds",
                "mean"
            ),
            median_inter_tx_seconds=(
                "inter_tx_seconds",
                "median"
            ),
            min_inter_tx_seconds=(
                "inter_tx_seconds",
                "min"
            ),
            std_inter_tx_seconds=(
                "inter_tx_seconds",
                "std"
            ),
            first_seen=(
                "timestamp",
                "min"
            ),
            last_seen=(
                "timestamp",
                "max"
            ),
        )
    )

    # ---------------------------------------------------------
    # Active duration
    # ---------------------------------------------------------

    temporal["active_duration_seconds"] = (
        (
            temporal["last_seen"]
            -
            temporal["first_seen"]
        )
        .dt.total_seconds()
    )

    # ---------------------------------------------------------
    # Transaction velocity
    # ---------------------------------------------------------

    tx_counts = (
        wallet_tx
        .groupby("wallet_id")
        .size()
    )

    duration = (
        temporal[
            "active_duration_seconds"
        ]
        .replace(0, np.nan)
    )

    temporal["transaction_velocity"] = (
        tx_counts
        /
        duration
    )

    # ---------------------------------------------------------
    # Burstiness
    #
    # B = (std - mean) / (std + mean)
    # ---------------------------------------------------------

    mean_delta = (
        temporal["avg_inter_tx_seconds"]
    )

    std_delta = (
        temporal["std_inter_tx_seconds"]
    )

    temporal["burstiness"] = (
        (std_delta - mean_delta)
        /
        (std_delta + mean_delta)
    )

    # ---------------------------------------------------------
    # Clean
    # ---------------------------------------------------------

    temporal = temporal.reset_index()

    temporal["first_seen"] = pd.to_datetime(
        temporal["first_seen"]
    )

    temporal["last_seen"] = pd.to_datetime(
        temporal["last_seen"]
    )

    numeric_columns = [
        "avg_inter_tx_seconds",
        "median_inter_tx_seconds",
        "min_inter_tx_seconds",
        "std_inter_tx_seconds",
        "active_duration_seconds",
        "transaction_velocity",
        "burstiness",
    ]

    temporal[numeric_columns] = (
        temporal[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            0
        )
        .fillna(0)
    )

    return temporal