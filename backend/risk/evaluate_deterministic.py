from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SCORE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "deterministic_scores.csv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "wallet_ground_truth.csv"
)


# ============================================================
# LOAD
# ============================================================

scores = pd.read_csv(
    SCORE_FILE
)

truth = pd.read_csv(
    GROUND_TRUTH_FILE
)


# ============================================================
# JOIN
# ============================================================

truth["wallet_id"] = (
    truth["wallet_id"]
    .astype(str)
)

scores["wallet_id"] = (
    scores["wallet_id"]
    .astype(str)
)

evaluation = scores.merge(
    truth[
        [
            "wallet_id",
            "label",
        ]
    ],
    on="wallet_id",
    how="inner",
)


# ============================================================
# TARGET
# ============================================================

y_true = (
    evaluation["label"]
    .eq("suspicious")
    .astype(int)
)

y_score = (
    evaluation[
        "deterministic_score"
    ]
    .astype(float)
)


# ============================================================
# TOP-K EVALUATION
# ============================================================

def precision_at_k(
    y_true,
    y_score,
    k,
):

    ranked = (
        pd.DataFrame({
            "truth": y_true,
            "score": y_score,
        })
        .sort_values(
            "score",
            ascending=False,
        )
        .head(k)
    )

    return float(
        ranked["truth"].mean()
    )


# ============================================================
# METRICS
# ============================================================

print("=" * 60)
print("DETERMINISTIC ENGINE EVALUATION")
print("=" * 60)

print(
    "Wallets evaluated:",
    len(evaluation)
)

print(
    "Suspicious wallets:",
    int(y_true.sum())
)


print(
    "\nAverage Precision:",
    round(
        average_precision_score(
            y_true,
            y_score
        ),
        4
    )
)


print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_true,
            y_score
        ),
        4
    )
)


# ------------------------------------------------------------
# Threshold for binary evaluation
#
# We use the 95th percentile because the engine itself
# defines its tail statistically.
# ------------------------------------------------------------

threshold = y_score.quantile(
    0.95
)

y_pred = (
    y_score >= threshold
).astype(int)


print(
    "Precision:",
    round(
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        4
    )
)


# ------------------------------------------------------------
# Ranking quality
# ------------------------------------------------------------

for k in [
    10,
    25,
    50,
    100,
]:

    if k <= len(evaluation):

        print(
            f"Precision@{k}:",
            round(
                precision_at_k(
                    y_true,
                    y_score,
                    k
                ),
                4
            )
        )