import ast
from typing import Any

import pandas as pd


LIST_COLUMNS = [
    "input_addresses",
    "output_addresses",
    "input_amounts_sats",
    "output_amounts_sats",
]


def parse_list(value: Any) -> list:
    """
    Convert list-like CSV strings into Python lists.

    Example:
        "['W0000258']"
        ->
        ['W0000258']
    """

    if value is None:
        return []

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:

        result = ast.literal_eval(
            str(value)
        )

        if isinstance(result, list):
            return result

        return [result]

    except (
        ValueError,
        SyntaxError
    ):

        return []


def normalize_transactions(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Parse list-like fields
    for column in LIST_COLUMNS:

        if column in df.columns:

            df[column] = (
                df[column]
                .apply(parse_list)
            )

    # Numeric satoshi fields
    numeric_columns = [
        "total_input_sats",
        "total_output_sats",
        "fee_sats",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def normalize_wallets(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    if "initial_balance_btc" in df.columns:

        df["initial_balance_btc"] = pd.to_numeric(
            df["initial_balance_btc"],
            errors="coerce"
        )

    return df


def normalize_entities(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    return df


def normalize_wallet_entity_links(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce"
    )

    df["valid_from"] = pd.to_datetime(
        df["valid_from"],
        errors="coerce"
    )

    df["valid_to"] = pd.to_datetime(
        df["valid_to"],
        errors="coerce"
    )

    return df


def normalize_transaction_edges(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["amount_sats"] = pd.to_numeric(
        df["amount_sats"],
        errors="coerce"
    )

    return df


def normalize_network(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    numeric_columns = [
        "src_port",
        "dst_port",
        "latency_ms",
        "bytes_sent",
        "bytes_received",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def normalize_ip_metadata(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    numeric_columns = [
        "latitude",
        "longitude",
        "asn",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def normalize_dataset(
    datasets: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:

    datasets = datasets.copy()

    datasets["transactions"] = (
        normalize_transactions(
            datasets["transactions"]
        )
    )

    datasets["wallets"] = (
        normalize_wallets(
            datasets["wallets"]
        )
    )

    if "entities" in datasets:
        datasets["entities"] = normalize_entities(datasets["entities"])

    if "wallet_entity_links" in datasets:
        datasets["wallet_entity_links"] = normalize_wallet_entity_links(
            datasets["wallet_entity_links"]
        )

    datasets["transaction_inputs"] = (
        normalize_transaction_edges(
            datasets["transaction_inputs"]
        )
    )

    datasets["transaction_outputs"] = (
        normalize_transaction_edges(
            datasets["transaction_outputs"]
        )
    )

    datasets["network_observations"] = (
        normalize_network(
            datasets["network_observations"]
        )
    )

    datasets["ip_metadata"] = (
        normalize_ip_metadata(
            datasets["ip_metadata"]
        )
    )

    return datasets