import pandas as pd


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    table_name: str
):

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{table_name}: missing columns: "
            f"{missing}"
        )


def validate_schema(
    datasets: dict[str, pd.DataFrame]
):

    require_columns(
        datasets["wallets"],
        [
            "wallet_id",
            "created_at",
            "script_type",
            "initial_balance_btc",
            "wallet_type",
            "primary_entity_id",
            "is_service",
        ],
        "wallets"
    )

    require_columns(
        datasets["transactions"],
        [
            "txid",
            "timestamp",
            "input_addresses",
            "output_addresses",
            "input_amounts_sats",
            "output_amounts_sats",
            "total_input_sats",
            "total_output_sats",
            "fee_sats",
            "script_type",
            "generator_behavior",
        ],
        "transactions"
    )

    require_columns(
        datasets["transaction_inputs"],
        [
            "txid",
            "wallet_id",
            "amount_sats",
        ],
        "transaction_inputs"
    )

    require_columns(
        datasets["transaction_outputs"],
        [
            "txid",
            "wallet_id",
            "amount_sats",
        ],
        "transaction_outputs"
    )

    require_columns(
        datasets["network_observations"],
        [
            "observation_id",
            "timestamp",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "txid",
            "message_type",
            "latency_ms",
            "bytes_sent",
            "bytes_received",
            "direction",
            "wallet_context",
        ],
        "network_observations"
    )

    require_columns(
        datasets["ip_metadata"],
        [
            "ip_id",
            "ip",
            "country_code",
            "country_name",
            "region_name",
            "city_name",
            "latitude",
            "longitude",
            "zip_code",
            "time_zone",
            "asn",
            "as_name",
            "network_type",
        ],
        "ip_metadata"
    )


def validate_unique_ids(
    datasets: dict[str, pd.DataFrame]
):

    # Wallet IDs
    wallets = datasets["wallets"]

    if wallets["wallet_id"].duplicated().any():

        raise ValueError(
            "Duplicate wallet_id detected."
        )

    # TXIDs
    transactions = datasets["transactions"]

    if transactions["txid"].duplicated().any():

        raise ValueError(
            "Duplicate txid detected."
        )

    # IPs
    ip_metadata = datasets["ip_metadata"]

    if ip_metadata["ip"].duplicated().any():

        raise ValueError(
            "Duplicate IP detected."
        )

    # Observation IDs
    network = datasets[
        "network_observations"
    ]

    if network[
        "observation_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate observation_id detected."
        )


def validate_references(
    datasets: dict[str, pd.DataFrame]
):

    transactions = datasets[
        "transactions"
    ]

    wallets = datasets[
        "wallets"
    ]

    inputs = datasets[
        "transaction_inputs"
    ]

    outputs = datasets[
        "transaction_outputs"
    ]

    network = datasets[
        "network_observations"
    ]

    ip_metadata = datasets[
        "ip_metadata"
    ]

    txids = set(
        transactions["txid"]
    )

    wallet_ids = set(
        wallets["wallet_id"]
    )

    ips = set(
        ip_metadata["ip"]
    )

    # Input TX references
    invalid_input_txids = set(
        inputs["txid"]
    ) - txids

    if invalid_input_txids:

        raise ValueError(
            "transaction_inputs contains "
            "unknown txids."
        )

    # Output TX references
    invalid_output_txids = set(
        outputs["txid"]
    ) - txids

    if invalid_output_txids:

        raise ValueError(
            "transaction_outputs contains "
            "unknown txids."
        )

    # Input wallets
    invalid_input_wallets = set(
        inputs["wallet_id"]
    ) - wallet_ids

    if invalid_input_wallets:

        raise ValueError(
            "transaction_inputs contains "
            "unknown wallet_ids."
        )

    # Output wallets
    invalid_output_wallets = set(
        outputs["wallet_id"]
    ) - wallet_ids

    if invalid_output_wallets:

        raise ValueError(
            "transaction_outputs contains "
            "unknown wallet_ids."
        )

    # Network TXIDs
    invalid_network_txids = set(
        network["txid"]
    ) - txids

    if invalid_network_txids:

        raise ValueError(
            "network_observations contains "
            "unknown txids."
        )

    # Network wallet context
    invalid_network_wallets = set(
        network["wallet_context"]
    ) - wallet_ids

    if invalid_network_wallets:

        raise ValueError(
            "network_observations contains "
            "unknown wallet_context values."
        )

    # Source IPs
    invalid_src_ips = set(
        network["src_ip"]
    ) - ips

    if invalid_src_ips:

        raise ValueError(
            "network_observations contains "
            "unknown src_ip values."
        )

    # Destination IPs
    invalid_dst_ips = set(
        network["dst_ip"]
    ) - ips

    if invalid_dst_ips:

        raise ValueError(
            "network_observations contains "
            "unknown dst_ip values."
        )


def validate_transaction_accounting(
    datasets: dict[str, pd.DataFrame]
):

    transactions = datasets[
        "transactions"
    ]

    calculated = (
        transactions["total_input_sats"]
        -
        transactions["total_output_sats"]
        -
        transactions["fee_sats"]
    )

    violations = (
        calculated != 0
    )

    if violations.any():

        count = int(
            violations.sum()
        )

        raise ValueError(
            f"{count} transaction accounting "
            f"violations detected."
        )


def validate_dataset(
    datasets: dict[str, pd.DataFrame]
):

    validate_schema(datasets)

    validate_unique_ids(datasets)

    validate_references(datasets)

    validate_transaction_accounting(
        datasets
    )

    return True