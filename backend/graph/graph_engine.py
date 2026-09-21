"""Build aggregated wallet-to-wallet graph edges from transaction I/O tables."""

from __future__ import annotations

import pandas as pd


def build_graph_edges(
    transactions: pd.DataFrame,
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
) -> pd.DataFrame:
    """Construct directed wallet edges with proportional amount allocation per transaction."""
    transactions = transactions.copy()
    inputs = inputs.copy()
    outputs = outputs.copy()

    transactions["txid"] = transactions["txid"].astype(str)
    inputs["txid"] = inputs["txid"].astype(str)
    outputs["txid"] = outputs["txid"].astype(str)
    inputs["wallet_id"] = inputs["wallet_id"].astype(str)
    outputs["wallet_id"] = outputs["wallet_id"].astype(str)

    inputs["amount_sats"] = pd.to_numeric(inputs["amount_sats"], errors="coerce").fillna(0.0)
    outputs["amount_sats"] = pd.to_numeric(outputs["amount_sats"], errors="coerce").fillna(0.0)

    tx_times = transactions[["txid", "timestamp"]].drop_duplicates("txid")
    records: list[dict] = []

    for txid, tx_inputs in inputs.groupby("txid"):
        tx_outputs = outputs[outputs["txid"] == txid]
        if tx_outputs.empty:
            continue

        total_input = tx_inputs["amount_sats"].sum()
        if total_input <= 0:
            continue

        timestamp_row = tx_times[tx_times["txid"] == txid]
        timestamp = timestamp_row.iloc[0]["timestamp"] if not timestamp_row.empty else None

        for _, input_row in tx_inputs.iterrows():
            source = str(input_row["wallet_id"])
            input_share = float(input_row["amount_sats"]) / total_input

            for _, output_row in tx_outputs.iterrows():
                target = str(output_row["wallet_id"])
                if source == target:
                    continue

                allocated_amount = float(output_row["amount_sats"]) * input_share
                records.append(
                    {
                        "source_wallet_id": source,
                        "target_wallet_id": target,
                        "transaction_count": 1,
                        "total_amount_sats": allocated_amount,
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "source_wallet_id",
                "target_wallet_id",
                "transaction_count",
                "total_amount_sats",
                "first_seen",
                "last_seen",
            ]
        )

    edges = pd.DataFrame(records)
    return edges.groupby(["source_wallet_id", "target_wallet_id"], as_index=False).agg(
        transaction_count=("transaction_count", "sum"),
        total_amount_sats=("total_amount_sats", "sum"),
        first_seen=("first_seen", "min"),
        last_seen=("last_seen", "max"),
    )
