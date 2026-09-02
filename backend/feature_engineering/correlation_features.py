"""
Correlation Features

Relates network observations to transaction activity.
Generates:
- avg_network_tx_time_difference
- min_network_tx_time_difference
- observations_per_tx
- rapid_hop_count
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_correlation_features(
    transactions: pd.DataFrame,
    network_observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build correlation features linking network observations and transactions per wallet.
    """
    if network_observations is None or network_observations.empty:
        return pd.DataFrame(
            columns=[
                "wallet_id",
                "avg_network_tx_time_difference",
                "min_network_tx_time_difference",
                "observations_per_tx",
                "rapid_hop_count",
            ]
        )

    net = network_observations.copy()
    tx = transactions.copy()

    net["wallet_context"] = net["wallet_context"].astype(str)
    net["txid"] = net["txid"].astype(str)

    # Parse timestamps
    net["net_time"] = pd.to_datetime(net["timestamp"], errors="coerce", utc=True)
    tx["tx_time"] = pd.to_datetime(tx["timestamp"], errors="coerce", utc=True)
    tx["txid"] = tx["txid"].astype(str)

    # Merge network observations with transactions on txid
    merged = net.merge(
        tx[["txid", "tx_time"]],
        on="txid",
        how="inner",
    )

    if not merged.empty:
        merged["time_diff_sec"] = (
            (merged["net_time"] - merged["tx_time"]).dt.total_seconds().abs()
        )
    else:
        merged["time_diff_sec"] = 0.0

    # Derive rapid hop threshold statistically from positive time differences
    pos_diffs = merged.loc[merged["time_diff_sec"] > 0, "time_diff_sec"]
    if len(pos_diffs) > 10:
        rapid_threshold = float(pos_diffs.quantile(0.10))
    else:
        rapid_threshold = 2.0  # default 2 seconds

    # Group by wallet_context
    records = []
    for wallet_id, group in net.groupby("wallet_context"):
        obs_count = len(group)
        unique_txids = group["txid"].nunique()
        obs_per_tx = float(obs_count / max(1, unique_txids))

        wallet_merged = merged[merged["wallet_context"] == wallet_id]
        if not wallet_merged.empty and wallet_merged["time_diff_sec"].notna().any():
            avg_diff = float(wallet_merged["time_diff_sec"].mean())
            min_diff = float(wallet_merged["time_diff_sec"].min())
            rapid_hops = int((wallet_merged["time_diff_sec"] <= rapid_threshold).sum())
        else:
            avg_diff = 0.0
            min_diff = 0.0
            rapid_hops = 0

        records.append(
            {
                "wallet_id": wallet_id,
                "avg_network_tx_time_difference": avg_diff,
                "min_network_tx_time_difference": min_diff,
                "observations_per_tx": obs_per_tx,
                "rapid_hop_count": rapid_hops,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "wallet_id",
                "avg_network_tx_time_difference",
                "min_network_tx_time_difference",
                "observations_per_tx",
                "rapid_hop_count",
            ]
        )

    return pd.DataFrame(records)