"""
Data Ingestion Layer — load, validate, and normalize raw transaction data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import OPTIONAL_RAW_FILES, REQUIRED_RAW_FILES
from backend.database import get_result
from backend.ingestion.csv_loader import load_csv
from backend.ingestion.normalizer import normalize_dataset
from backend.ingestion.validator import validate_dataset


def load_raw_dataset(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load required and optional raw CSV files from a directory."""
    raw_dir = Path(raw_dir)
    datasets: dict[str, pd.DataFrame] = {}

    for filename in REQUIRED_RAW_FILES:
        path = raw_dir / filename
        datasets[Path(filename).stem] = load_csv(path)

    for filename in OPTIONAL_RAW_FILES:
        path = raw_dir / filename
        if path.exists():
            datasets[Path(filename).stem] = load_csv(path)

    return datasets


def run_ingestion(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Execute the full ingestion pipeline:
    load → validate schema/references → normalize types.
    """
    datasets = load_raw_dataset(raw_dir)
    validate_dataset(datasets)
    return normalize_dataset(datasets)


def run_ingestion_from_db(job_id: str) -> dict[str, pd.DataFrame]:
    """Load, validate, and normalize an uploaded job stored in SQLite."""
    datasets: dict[str, pd.DataFrame] = {}
    for filename in REQUIRED_RAW_FILES + OPTIONAL_RAW_FILES:
        records = get_result(job_id, f"input/{Path(filename).stem}")
        if records is not None:
            datasets[Path(filename).stem] = pd.DataFrame(records)

    missing = [name for name in REQUIRED_RAW_FILES if Path(name).stem not in datasets]
    if missing:
        raise FileNotFoundError(f"Missing database input artifacts for job {job_id}: {missing}")

    validate_dataset(datasets)
    return normalize_dataset(datasets)


def ingestion_report(datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Summarize ingested dataset for manifest reporting."""
    return {
        "wallet_count": len(datasets["wallets"]),
        "transaction_count": len(datasets["transactions"]),
        "network_observation_count": len(datasets["network_observations"]),
        "ip_count": len(datasets["ip_metadata"]),
        "has_entities": "entities" in datasets,
        "has_wallet_entity_links": "wallet_entity_links" in datasets,
    }
