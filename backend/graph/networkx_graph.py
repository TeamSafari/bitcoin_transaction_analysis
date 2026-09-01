from pathlib import Path

import pandas as pd
import networkx as nx


REQUIRED_COLUMNS = [
    "source_wallet_id",
    "target_wallet_id",
    "transaction_count",
    "total_amount_sats",
    "first_seen",
    "last_seen",
]


def load_graph_edges(
    path: Path
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Graph edges not found: {path}"
        )

    df = pd.read_csv(path)

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing graph columns: {missing}"
        )

    df["source_wallet_id"] = (
        df["source_wallet_id"]
        .astype(str)
    )

    df["target_wallet_id"] = (
        df["target_wallet_id"]
        .astype(str)
    )

    return df


def build_wallet_graph(
    edges: pd.DataFrame,
    wallet_ids: list[str]
) -> nx.DiGraph:

    graph = nx.DiGraph()

    graph.add_nodes_from(
        wallet_ids
    )

    for _, row in edges.iterrows():

        source = row[
            "source_wallet_id"
        ]

        target = row[
            "target_wallet_id"
        ]

        if source == target:
            continue

        graph.add_edge(
            source,
            target,
            transaction_count=float(
                row["transaction_count"]
            ),
            total_amount_sats=float(
                row["total_amount_sats"]
            ),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )

    return graph