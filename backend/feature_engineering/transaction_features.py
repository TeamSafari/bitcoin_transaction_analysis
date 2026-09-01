import pandas as pd


def build_transaction_features(
    wallets: pd.DataFrame,
    transactions: pd.DataFrame,
    transaction_inputs: pd.DataFrame,
    transaction_outputs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build wallet-level transaction features.

    One output row = one wallet.
    """

    wallet_ids = wallets["wallet_id"].astype(str)

    tx = transactions.copy()
    inputs = transaction_inputs.copy()
    outputs = transaction_outputs.copy()

    tx["txid"] = tx["txid"].astype(str)
    inputs["txid"] = inputs["txid"].astype(str)
    outputs["txid"] = outputs["txid"].astype(str)

    inputs["wallet_id"] = (
        inputs["wallet_id"].astype(str)
    )

    outputs["wallet_id"] = (
        outputs["wallet_id"].astype(str)
    )

    # ---------------------------------------------------------
    # INPUT FEATURES
    # ---------------------------------------------------------

    input_stats = (
        inputs.groupby("wallet_id")
        .agg(
            incoming_tx_count=(
                "txid",
                "nunique"
            ),
            total_received_sats=(
                "amount_sats",
                "sum"
            ),
            avg_received_sats=(
                "amount_sats",
                "mean"
            ),
            median_received_sats=(
                "amount_sats",
                "median"
            ),
            max_received_sats=(
                "amount_sats",
                "max"
            ),
            std_received_sats=(
                "amount_sats",
                "std"
            ),
            unique_input_counterparties=(
                "wallet_id",
                "nunique"
            ),
        )
    )

    # ---------------------------------------------------------
    # OUTPUT FEATURES
    # ---------------------------------------------------------

    output_stats = (
        outputs.groupby("wallet_id")
        .agg(
            outgoing_tx_count=(
                "txid",
                "nunique"
            ),
            total_sent_sats=(
                "amount_sats",
                "sum"
            ),
            avg_sent_sats=(
                "amount_sats",
                "mean"
            ),
            median_sent_sats=(
                "amount_sats",
                "median"
            ),
            max_sent_sats=(
                "amount_sats",
                "max"
            ),
            std_sent_sats=(
                "amount_sats",
                "std"
            ),
            unique_output_counterparties=(
                "wallet_id",
                "nunique"
            ),
        )
    )

    # ---------------------------------------------------------
    # INPUT/OUTPUT COUNT FEATURES
    # ---------------------------------------------------------

    tx_input_counts = (
        inputs.groupby("txid")
        .size()
        .rename("input_count")
    )

    tx_output_counts = (
        outputs.groupby("txid")
        .size()
        .rename("output_count")
    )

    tx_counts = (
        tx[["txid"]]
        .drop_duplicates()
        .merge(
            inputs[
                ["txid", "wallet_id"]
            ].drop_duplicates(),
            on="txid",
            how="left",
        )
    )

    wallet_tx_counts = (
        tx_counts.groupby("wallet_id")
        .size()
        .rename("tx_count")
    )

    # ---------------------------------------------------------
    # INPUT/OUTPUTS PER TRANSACTION
    # ---------------------------------------------------------

    tx_structure = (
        pd.concat(
            [
                tx_input_counts,
                tx_output_counts,
            ],
            axis=1,
        )
        .fillna(0)
    )

    tx_structure["input_count"] = (
        tx_structure["input_count"]
        .astype(int)
    )

    tx_structure["output_count"] = (
        tx_structure["output_count"]
        .astype(int)
    )

    input_structure = (
        inputs[["wallet_id", "txid"]]
        .drop_duplicates()
        .merge(
            tx_structure,
            left_on="txid",
            right_index=True,
            how="left",
        )
    )

    output_structure = (
        outputs[["wallet_id", "txid"]]
        .drop_duplicates()
        .merge(
            tx_structure,
            left_on="txid",
            right_index=True,
            how="left",
        )
    )

    input_structure_stats = (
        input_structure.groupby("wallet_id")
        .agg(
            avg_inputs_per_tx=(
                "input_count",
                "mean"
            ),
            max_inputs_per_tx=(
                "input_count",
                "max"
            ),
        )
    )

    output_structure_stats = (
        output_structure.groupby("wallet_id")
        .agg(
            avg_outputs_per_tx=(
                "output_count",
                "mean"
            ),
            max_outputs_per_tx=(
                "output_count",
                "max"
            ),
        )
    )

    # ---------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------

    result = pd.DataFrame({
        "wallet_id": wallet_ids
    })

    for frame in [
        wallet_tx_counts,
        input_stats,
        output_stats,
        input_structure_stats,
        output_structure_stats,
    ]:

        result = result.merge(
            frame,
            left_on="wallet_id",
            right_index=True,
            how="left",
        )

    # ---------------------------------------------------------
    # DERIVED FEATURES
    # ---------------------------------------------------------

    result["unique_counterparties"] = (
        result[
            [
                "unique_input_counterparties",
                "unique_output_counterparties",
            ]
        ]
        .sum(axis=1)
    )

    result["fan_in"] = (
        result["unique_input_counterparties"]
    )

    result["fan_out"] = (
        result["unique_output_counterparties"]
    )

    result["in_out_ratio"] = (
        result["total_received_sats"]
        /
        result["total_sent_sats"].replace(
            0,
            1
        )
    )

    # ---------------------------------------------------------
    # CLEAN
    # ---------------------------------------------------------

    numeric_columns = [
        col
        for col in result.columns
        if col != "wallet_id"
    ]

    result[numeric_columns] = (
        result[numeric_columns]
        .replace(
            [float("inf"), float("-inf")],
            0
        )
        .fillna(0)
    )

    return result