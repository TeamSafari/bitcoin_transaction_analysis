"""Central configuration for the Bitcoin forensics backend."""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
JOBS_DIR = OUTPUT_DIR / "jobs"
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "models" / "artifacts"
LLM_MODEL_DIR = ARTIFACT_DIR / "llm"
DB_PATH = OUTPUT_DIR / "jobs.db"

REQUIRED_RAW_FILES = [
    "wallets.csv",
    "transactions.csv",
    "transaction_inputs.csv",
    "transaction_outputs.csv",
    "network_observations.csv",
    "ip_metadata.csv",
]

OPTIONAL_RAW_FILES = [
    "entities.csv",
    "wallet_entity_links.csv",
]

ALL_RAW_FILES = REQUIRED_RAW_FILES + OPTIONAL_RAW_FILES

# Fusion thresholds for investigator console routing
REVIEW_THRESHOLD = 0.50
HOLD_THRESHOLD = 0.80
