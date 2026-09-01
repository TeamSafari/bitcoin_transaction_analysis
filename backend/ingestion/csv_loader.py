from pathlib import Path

import pandas as pd


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected CSV file: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"CSV is empty: {path}"
        )

    return df


# ============================================================
# RAW DATASET
# ============================================================

def load_raw_dataset(
    raw_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Load the canonical raw dataset.

    Expected:
        entities.csv
        wallets.csv
        wallet_entity_links.csv
        ip_metadata.csv
        transactions.csv
        transaction_inputs.csv
        transaction_outputs.csv
        network_observations.csv

    transactions_display.csv is an export/view and is
    intentionally NOT required here.
    """

    expected_files = [
        "entities.csv",
        "wallets.csv",
        "wallet_entity_links.csv",
        "ip_metadata.csv",
        "transactions.csv",
        "transaction_inputs.csv",
        "transaction_outputs.csv",
        "network_observations.csv",
    ]

    datasets = {}

    for filename in expected_files:

        path = raw_dir / filename

        datasets[
            Path(filename).stem
        ] = load_csv(path)

    return datasets