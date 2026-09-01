from pathlib import Path

import json
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# File:
# backend/alerts/alert_service.py
#
# parents[0] -> alerts
# parents[1] -> backend
# parents[2] -> project root

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs"
)

MODEL_OUTPUT_DIR = (
    OUTPUT_DIR / "models"
)

EXPLAINABILITY_DIR = (
    OUTPUT_DIR / "explainability"
)

ALERT_OUTPUT_DIR = (
    OUTPUT_DIR / "alerts"
)


# ============================================================
# INPUT FILES
# ============================================================

RISK_FEATURES_FILE = (
    MODEL_OUTPUT_DIR
    / "risk_training_features.csv"
)

RISK_PREDICTIONS_FILE = (
    MODEL_OUTPUT_DIR
    / "risk_model_predictions.csv"
)

SHAP_CONTRIBUTIONS_FILE = (
    EXPLAINABILITY_DIR
    / "top_feature_contributions.csv"
)

GLOBAL_IMPORTANCE_FILE = (
    EXPLAINABILITY_DIR
    / "global_feature_importance.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

ALERTS_FILE = (
    ALERT_OUTPUT_DIR
    / "alerts.csv"
)

ALERTS_JSON_FILE = (
    ALERT_OUTPUT_DIR
    / "alerts.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# These thresholds are presentation / alert-routing
# thresholds, NOT model-training parameters.

HIGH_RISK_THRESHOLD = 0.80

MEDIUM_RISK_THRESHOLD = 0.50

# Number of SHAP factors to include per alert.

TOP_EVIDENCE_COUNT = 5

# Number of alerts to retain in the final
# ranked alert queue.

MAX_ALERTS = 100


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(
    path: Path,
    name: str,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    if df.empty:

        raise ValueError(
            f"{name} is empty."
        )

    return df


# ============================================================
# LOAD RISK PREDICTIONS
# ============================================================

def load_risk_predictions():

    df = load_csv(
        RISK_PREDICTIONS_FILE,
        "risk_model_predictions.csv",
    )

    required = {
        "risk_probability",
        "risk_prediction",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Risk prediction file is missing "
            f"columns: {sorted(missing)}"
        )

    return df


# ============================================================
# LOAD RISK FEATURES
# ============================================================

def load_risk_features():

    df = load_csv(
        RISK_FEATURES_FILE,
        "risk_training_features.csv",
    )

    if "wallet_id" not in df.columns:

        raise ValueError(
            "risk_training_features.csv "
            "must contain wallet_id."
        )

    df["wallet_id"] = (
        df["wallet_id"]
        .astype(str)
    )

    return df


# ============================================================
# LOAD SHAP EVIDENCE
# ============================================================

def load_shap_evidence():

    df = load_csv(
        SHAP_CONTRIBUTIONS_FILE,
        "top_feature_contributions.csv",
    )

    required = {
        "wallet_id",
        "rank",
        "feature",
        "feature_value",
        "shap_value",
        "absolute_shap_value",
        "direction",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "SHAP contribution file is missing "
            f"columns: {sorted(missing)}"
        )

    df["wallet_id"] = (
        df["wallet_id"]
        .astype(str)
    )

    return df


# ============================================================
# OPTIONAL DETERMINISTIC RISK
# ============================================================

def find_deterministic_risk_column(
    df: pd.DataFrame,
):

    possible_columns = [

        "deterministic_risk_score",

        "deterministic_risk",

        "statistical_risk_score",

        "statistical_risk",
    ]

    for column in possible_columns:

        if column in df.columns:

            return column

    return None


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk(
    probability: float,
):

    if probability >= HIGH_RISK_THRESHOLD:

        return "HIGH"

    if probability >= MEDIUM_RISK_THRESHOLD:

        return "MEDIUM"

    return "LOW"


# ============================================================
# ALERT PRIORITY
# ============================================================

def calculate_priority(
    probability: float,
    evidence_count: int,
):

    # Risk probability is the primary ranking factor.
    #
    # Evidence count is only used as a small tie-breaker.
    #
    # This prevents the alert system from artificially
    # increasing risk because a wallet happens to have
    # more explainability records.

    evidence_factor = min(
        evidence_count,
        TOP_EVIDENCE_COUNT,
    ) / TOP_EVIDENCE_COUNT

    priority_score = (
        0.90 * probability
        +
        0.10 * evidence_factor
    )

    return priority_score


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence(
    shap_df: pd.DataFrame,
    wallet_id: str,
):

    wallet_shap = (
        shap_df[
            shap_df["wallet_id"]
            == wallet_id
        ]
        .sort_values(
            "absolute_shap_value",
            ascending=False,
        )
        .head(
            TOP_EVIDENCE_COUNT
        )
    )

    evidence = []

    for _, row in wallet_shap.iterrows():

        shap_value = float(
            row["shap_value"]
        )

        feature_value = float(
            row["feature_value"]
        )

        if shap_value > 0:

            effect = (
                "increases_risk"
            )

        elif shap_value < 0:

            effect = (
                "decreases_risk"
            )

        else:

            effect = "neutral"

        evidence.append(
            {
                "feature":
                    str(row["feature"]),

                "feature_value":
                    feature_value,

                "shap_value":
                    shap_value,

                "effect":
                    effect,
            }
        )

    return evidence


# ============================================================
# BUILD ALERT
# ============================================================

def build_alert(
    wallet_id: str,
    risk_probability: float,
    risk_prediction: int,
    evidence: list,
    deterministic_risk,
):

    risk_level = classify_risk(
        risk_probability
    )

    priority_score = (
        calculate_priority(
            risk_probability,
            len(evidence),
        )
    )

    # --------------------------------------------------------
    # Alert decision
    #
    # Only medium/high risk enters the alert queue.
    # Low-risk wallets remain outside the alert stream.
    # --------------------------------------------------------

    alert_status = (
        "ALERT"
        if risk_probability
        >= MEDIUM_RISK_THRESHOLD
        else "NO_ALERT"
    )

    return {

        "wallet_id":
            wallet_id,

        "risk_probability":
            round(
                risk_probability,
                6,
            ),

        "risk_prediction":
            int(risk_prediction),

        "risk_level":
            risk_level,

        "deterministic_risk_score":
            (
                None
                if deterministic_risk is None
                else round(
                    float(
                        deterministic_risk
                    ),
                    6,
                )
            ),

        "priority_score":
            round(
                priority_score,
                6,
            ),

        "alert_status":
            alert_status,

        "evidence":
            evidence,
    }


# ============================================================
# BUILD ALERT QUEUE
# ============================================================

def build_alert_queue():

    print("=" * 60)
    print("BUILDING ALERT QUEUE")
    print("=" * 60)

    risk_predictions = (
        load_risk_predictions()
    )

    risk_features = (
        load_risk_features()
    )

    shap_df = (
        load_shap_evidence()
    )

    # --------------------------------------------------------
    # Important:
    #
    # risk_model_predictions.csv currently contains test-set
    # predictions from model training.
    #
    # It does NOT necessarily contain wallet_id.
    #
    # Therefore, if wallet-level production alerts are required,
    # the inference stage should produce wallet_id + probability.
    # --------------------------------------------------------

    if "wallet_id" not in (
        risk_predictions.columns
    ):

        if len(risk_predictions) != len(
            risk_features
        ):

            raise ValueError(
                "risk_model_predictions.csv "
                "does not contain wallet_id and "
                "cannot be aligned safely with "
                "risk_training_features.csv."
            )

        # This fallback is only safe when
        # row ordering is guaranteed.

        risk_predictions = (
            risk_predictions.copy()
        )

        risk_predictions.insert(
            0,
            "wallet_id",
            risk_features[
                "wallet_id"
            ].values,
        )

    risk_predictions[
        "wallet_id"
    ] = (
        risk_predictions[
            "wallet_id"
        ]
        .astype(str)
    )

    deterministic_column = (
        find_deterministic_risk_column(
            risk_features
        )
    )

    deterministic_lookup = {}

    if deterministic_column:

        deterministic_lookup = dict(
            zip(
                risk_features[
                    "wallet_id"
                ].astype(str),

                pd.to_numeric(
                    risk_features[
                        deterministic_column
                    ],
                    errors="coerce",
                ),
            )
        )

    alerts = []

    # --------------------------------------------------------
    # Build wallet alerts
    # --------------------------------------------------------

    for _, row in (
        risk_predictions.iterrows()
    ):

        wallet_id = str(
            row["wallet_id"]
        )

        risk_probability = float(
            row["risk_probability"]
        )

        risk_prediction = int(
            row["risk_prediction"]
        )

        evidence = build_evidence(
            shap_df,
            wallet_id,
        )

        deterministic_risk = (
            deterministic_lookup.get(
                wallet_id
            )
        )

        alert = build_alert(
            wallet_id=wallet_id,
            risk_probability=risk_probability,
            risk_prediction=risk_prediction,
            evidence=evidence,
            deterministic_risk=deterministic_risk,
        )

        if (
            alert["alert_status"]
            == "ALERT"
        ):

            alerts.append(
                alert
            )

    # --------------------------------------------------------
    # Rank alerts
    # --------------------------------------------------------

    alerts.sort(
        key=lambda alert:
            alert["priority_score"],
        reverse=True,
    )

    alerts = alerts[
        :MAX_ALERTS
    ]

    print(
        f"Alerts generated: "
        f"{len(alerts)}"
    )

    return alerts


# ============================================================
# SAVE CSV
# ============================================================

def save_alert_csv(
    alerts,
):

    rows = []

    for alert in alerts:

        evidence = (
            alert["evidence"]
        )

        top_evidence = (
            evidence[:TOP_EVIDENCE_COUNT]
        )

        evidence_text = " | ".join(
            [
                (
                    f"{item['feature']}: "
                    f"{item['effect']} "
                    f"(SHAP="
                    f"{item['shap_value']:.4f})"
                )

                for item in top_evidence
            ]
        )

        rows.append(
            {
                "wallet_id":
                    alert["wallet_id"],

                "risk_probability":
                    alert["risk_probability"],

                "risk_prediction":
                    alert["risk_prediction"],

                "risk_level":
                    alert["risk_level"],

                "deterministic_risk_score":
                    alert[
                        "deterministic_risk_score"
                    ],

                "priority_score":
                    alert["priority_score"],

                "alert_status":
                    alert["alert_status"],

                "top_evidence":
                    evidence_text,
            }
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        ALERTS_FILE,
        index=False,
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_alert_json(
    alerts,
):

    with open(
        ALERTS_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            alerts,
            file,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 60)
    print("# ALERT SERVICE")
    print("#" * 60)

    ALERT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    alerts = (
        build_alert_queue()
    )

    # --------------------------------------------------------
    # Save machine-readable outputs
    # --------------------------------------------------------

    save_alert_csv(
        alerts
    )

    save_alert_json(
        alerts
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALERT SERVICE COMPLETE")
    print("=" * 60)

    print(
        f"Alerts generated: "
        f"{len(alerts)}"
    )

    if alerts:

        high_count = sum(
            alert["risk_level"]
            == "HIGH"
            for alert in alerts
        )

        medium_count = sum(
            alert["risk_level"]
            == "MEDIUM"
            for alert in alerts
        )

        print(
            f"High-risk alerts: "
            f"{high_count}"
        )

        print(
            f"Medium-risk alerts: "
            f"{medium_count}"
        )

        print(
            "\nTop alert:"
        )

        top = alerts[0]

        print(
            f"Wallet: "
            f"{top['wallet_id']}"
        )

        print(
            f"Risk: "
            f"{top['risk_probability']:.4f}"
        )

        print(
            f"Level: "
            f"{top['risk_level']}"
        )

    print(
        f"\nCSV:\n"
        f"{ALERTS_FILE}"
    )

    print(
        f"\nJSON:\n"
        f"{ALERTS_JSON_FILE}"
    )


if __name__ == "__main__":
    main()