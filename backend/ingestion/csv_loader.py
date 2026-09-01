from pathlib import Path
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file without changing its schema.
    """

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected CSV file: {path}")

    df = pd.read_csv(path)

    return df


def load_raw_dataset(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load the complete raw dataset.

    Expected files:
        entities.csv
        wallets.csv
        wallet_entity_links.csv
        ip_metadata.csv
        transactions.csv
        transaction_inputs.csv
        transaction_outputs.csv
        network_observations.csv
        transactions_display.csv
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
        "transactions_display.csv",
    ]

    datasets = {}

    for filename in expected_files:

        path = raw_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required raw dataset file missing: {path}"
            )

        key = Path(filename).stem

        datasets[key] = load_csv(path)

    return datasets