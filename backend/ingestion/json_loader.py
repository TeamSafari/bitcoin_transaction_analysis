from pathlib import Path
import json
import pandas as pd


def load_transactions_json(
    path: Path
) -> pd.DataFrame:
    """
    Load exports/transactions.json.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Expected JSON file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            "transactions.json must contain a JSON array."
        )

    df = pd.DataFrame(records)

    return df