from pathlib import Path

import pandas as pd

from backend.graph.networkx_graph import (
    build_wallet_graph,
)

from backend.graph.graph_features import (
    calculate_graph_features,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

RAW_DIR = (
    PROJECT_ROOT / "data" / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
)


GRAPH_EDGES_FILE = (
    OUTPUT_DIR
    / "graph_edges.csv"
)

GRAPH_FEATURES_FILE = (
    OUTPUT_DIR
    / "graph_features.csv"
)


# ============================================================
# BUILD TRANSACTION GRAPH EDGES
# ============================================================

def build_graph_edges(
    transactions: pd.DataFrame,
    inputs: pd.DataFrame,
    outputs: pd.DataFrame,
) -> pd.DataFrame:

    transactions = transactions.copy()
    inputs = inputs.copy()
    outputs = outputs.copy()

    transactions["txid"] = (
        transactions["txid"]
        .astype(str)
    )

    inputs["txid"] = (
        inputs["txid"]
        .astype(str)
    )

    outputs["txid"] = (
        outputs["txid"]
        .astype(str)
    )

    inputs["wallet_id"] = (
        inputs["wallet_id"]
        .astype(str)
    )

    outputs["wallet_id"] = (
        outputs["wallet_id"]
        .astype(str)
    )

    inputs["amount_sats"] = pd.to_numeric(
        inputs["amount_sats"],
        errors="coerce",
    ).fillna(0.0)

    outputs["amount_sats"] = pd.to_numeric(
        outputs["amount_sats"],
        errors="coerce",
    ).fillna(0.0)

    tx_times = transactions[
        [
            "txid",
            "timestamp",
        ]
    ].drop_duplicates(
        "txid"
    )

    records = []

    # --------------------------------------------------------
    # Each input wallet contributes to each output wallet.
    #
    # Output value is allocated proportionally to the input
    # contribution. This avoids counting the entire transaction
    # output amount once for every input wallet.
    # --------------------------------------------------------

    for txid, tx_inputs in (
        inputs.groupby("txid")
    ):

        tx_outputs = outputs[
            outputs["txid"] == txid
        ]

        if tx_outputs.empty:
            continue

        total_input = (
            tx_inputs["amount_sats"]
            .sum()
        )

        if total_input <= 0:
            continue

        timestamp_row = tx_times[
            tx_times["txid"] == txid
        ]

        timestamp = None

        if not timestamp_row.empty:

            timestamp = timestamp_row.iloc[0][
                "timestamp"
            ]

        for _, input_row in (
            tx_inputs.iterrows()
        ):

            source = str(
                input_row["wallet_id"]
            )

            input_amount = float(
                input_row["amount_sats"]
            )

            input_share = (
                input_amount
                / total_input
            )

            for _, output_row in (
                tx_outputs.iterrows()
            ):

                target = str(
                    output_row["wallet_id"]
                )

                if source == target:
                    continue

                output_amount = float(
                    output_row["amount_sats"]
                )

                allocated_amount = (
                    output_amount
                    * input_share
                )

                records.append(
                    {
                        "source_wallet_id":
                            source,

                        "target_wallet_id":
                            target,

                        "transaction_count":
                            1,

                        "total_amount_sats":
                            allocated_amount,

                        "first_seen":
                            timestamp,

                        "last_seen":
                            timestamp,
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

    edges = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Aggregate repeated wallet pairs
    # --------------------------------------------------------

    edges = (
        edges.groupby(
            [
                "source_wallet_id",
                "target_wallet_id",
            ],
            as_index=False,
        )
        .agg(
            transaction_count=(
                "transaction_count",
                "sum",
            ),
            total_amount_sats=(
                "total_amount_sats",
                "sum",
            ),
            first_seen=(
                "first_seen",
                "min",
            ),
            last_seen=(
                "last_seen",
                "max",
            ),
        )
    )

    return edges


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NETWORKX GRAPH ENGINE")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    wallets = pd.read_csv(
        RAW_DIR / "wallets.csv"
    )

    transactions = pd.read_csv(
        RAW_DIR / "transactions.csv"
    )

    inputs = pd.read_csv(
        RAW_DIR / "transaction_inputs.csv"
    )

    outputs = pd.read_csv(
        RAW_DIR / "transaction_outputs.csv"
    )

    wallet_ids = (
        wallets["wallet_id"]
        .astype(str)
        .tolist()
    )

    edges = build_graph_edges(
        transactions,
        inputs,
        outputs,
    )

    edges.to_csv(
        GRAPH_EDGES_FILE,
        index=False,
    )

    graph = build_wallet_graph(
        edges,
        wallet_ids,
    )

    graph_features = (
        calculate_graph_features(
            graph
        )
    )

    graph_features[
        "wallet_id"
    ] = (
        graph_features[
            "wallet_id"
        ]
        .astype(str)
    )

    graph_features.to_csv(
        GRAPH_FEATURES_FILE,
        index=False,
    )

    print(
        f"Nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Graph edges saved:\n"
        f"{GRAPH_EDGES_FILE}"
    )

    print(
        f"Graph features saved:\n"
        f"{GRAPH_FEATURES_FILE}"
    )

    return graph_features


if __name__ == "__main__":
    main()