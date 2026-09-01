from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

# File location:
# backend/models/risk_model.py
#
# parents[0] -> models
# parents[1] -> backend
# parents[2] -> project root

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

MODEL_OUTPUT_DIR = (
    OUTPUT_DIR
    / "models"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "artifacts"
)


# ============================================================
# INPUT
# ============================================================

TRAINING_FILE = (
    MODEL_OUTPUT_DIR
    / "risk_training_features.csv"
)


# ============================================================
# OUTPUT
# ============================================================

MODEL_FILE = (
    ARTIFACT_DIR
    / "risk_model.pkl"
)

PREDICTIONS_FILE = (
    MODEL_OUTPUT_DIR
    / "risk_model_predictions.csv"
)

METRICS_FILE = (
    MODEL_OUTPUT_DIR
    / "risk_model_metrics.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "label"


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

XGB_PARAMS = {

    "n_estimators": 300,

    "max_depth": 5,

    "learning_rate": 0.05,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "min_child_weight": 3,

    "reg_alpha": 0.1,

    "reg_lambda": 1.0,

    "objective": "binary:logistic",

    "eval_metric": "logloss",

    "random_state": RANDOM_STATE,

    "n_jobs": -1,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    if not TRAINING_FILE.exists():

        raise FileNotFoundError(
            f"\nTraining file not found:\n"
            f"{TRAINING_FILE}\n\n"
            "Expected:\n"
            "outputs/models/"
            "risk_training_features.csv"
        )

    df = pd.read_csv(
        TRAINING_FILE
    )

    if df.empty:

        raise ValueError(
            "risk_training_features.csv "
            "is empty."
        )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found.\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns loaded: "
        f"{len(df.columns)}"
    )

    return df


# ============================================================
# PREPARE TARGET
# ============================================================

def prepare_target(
    df: pd.DataFrame,
):

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    if y.isna().any():

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "contains non-numeric or missing values."
        )

    # --------------------------------------------------------
    # Binary target validation
    # --------------------------------------------------------

    unique_values = sorted(
        y.unique().tolist()
    )

    if not set(unique_values).issubset(
        {0, 1}
    ):

        raise ValueError(
            "The risk model currently expects "
            "a binary target encoded as 0/1.\n"
            f"Found: {unique_values}"
        )

    y = y.astype(int)

    if y.nunique() < 2:

        raise ValueError(
            "Training data contains only "
            "one target class."
        )

    return y


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df: pd.DataFrame,
):

    # --------------------------------------------------------
    # Columns that must never be used as model features
    #
    # These are identifiers / ground-truth metadata and
    # would create leakage or meaningless learning.
    # --------------------------------------------------------

    EXCLUDED_COLUMNS = {

        TARGET_COLUMN,

        "wallet_id",

        "behavior_type",

        "scenario_id",

        "wallet_type",

        "primary_entity_id",
    }

    feature_columns = []

    X = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # Select numeric features
    # --------------------------------------------------------

    for column in df.columns:

        if column in EXCLUDED_COLUMNS:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        # Keep columns that contain
        # actual numerical information.
        if numeric.notna().sum() > 0:

            X[column] = numeric

            feature_columns.append(
                column
            )

    if X.empty:

        raise ValueError(
            "No numeric model features "
            "were found."
        )

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    for column in X.columns:

        median = X[column].median()

        if pd.isna(median):

            median = 0.0

        X[column] = (
            X[column]
            .fillna(median)
        )

    # --------------------------------------------------------
    # Remove constant features
    # --------------------------------------------------------

    non_constant_columns = [
        column
        for column in X.columns
        if X[column].nunique() > 1
    ]

    X = X[
        non_constant_columns
    ]

    feature_columns = (
        non_constant_columns
    )

    if X.empty:

        raise ValueError(
            "No non-constant features "
            "remain."
        )

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    return X, feature_columns


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

def calculate_scale_pos_weight(
    y_train: pd.Series,
):

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    if positive_count == 0:

        raise ValueError(
            "No positive samples in "
            "training data."
        )

    return (
        negative_count
        /
        positive_count
    )


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_model(
    X_train,
    y_train,
):

    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_train
        )
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.4f}"
    )

    params = dict(
        XGB_PARAMS
    )

    params[
        "scale_pos_weight"
    ] = scale_pos_weight

    model = XGBClassifier(
        **params
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

    except ValueError:

        roc_auc = float("nan")

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    metrics = {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            roc_auc,
    }

    return (
        metrics,
        cm,
        report,
        probabilities,
        predictions,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_columns,
):

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {

        "model":
            model,

        "feature_columns":
            feature_columns,

        "target_column":
            TARGET_COLUMN,

        "model_type":
            "XGBClassifier",

        "random_state":
            RANDOM_STATE,
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
    )

    print(
        f"\nModel saved:\n"
        f"{MODEL_FILE}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    X_test,
    y_test,
    probabilities,
    predictions,
):

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = pd.DataFrame(
        {
            "actual_label":
                y_test.values,

            "risk_probability":
                probabilities,

            "risk_prediction":
                predictions,
        }
    )

    output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"Predictions saved:\n"
        f"{PREDICTIONS_FILE}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics,
    confusion,
    report,
):

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "XGBOOST RISK MODEL\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "Metrics\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for name, value in (
            metrics.items()
        ):

            file.write(
                f"{name}: "
                f"{value:.6f}\n"
            )

        file.write(
            "\nConfusion Matrix\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            str(confusion)
            + "\n"
        )

        file.write(
            "\nClassification Report\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            report
        )

    print(
        f"Metrics saved:\n"
        f"{METRICS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 60)
    print("# XGBOOST RISK MODEL")
    print("#" * 60)

    # --------------------------------------------------------
    # 1. Load final risk features
    # --------------------------------------------------------

    df = load_training_data()

    # --------------------------------------------------------
    # 2. Prepare target
    # --------------------------------------------------------

    y = prepare_target(
        df
    )

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 3. Prepare model features
    # --------------------------------------------------------

    X, feature_columns = (
        prepare_features(df)
    )

    # --------------------------------------------------------
    # 4. Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
    )

    # --------------------------------------------------------
    # 5. Train XGBoost
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    (
        metrics,
        confusion,
        report,
        probabilities,
        predictions,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    for name, value in (
        metrics.items()
    ):

        print(
            f"{name:10s}: "
            f"{value:.4f}"
        )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion
    )

    print(
        "\nClassification Report:"
    )

    print(
        report
    )

    # --------------------------------------------------------
    # 7. Save model
    # --------------------------------------------------------

    save_model(
        model,
        feature_columns,
    )

    # --------------------------------------------------------
    # 8. Save predictions
    # --------------------------------------------------------

    save_predictions(
        X_test=X_test,
        y_test=y_test,
        probabilities=probabilities,
        predictions=predictions,
    )

    # --------------------------------------------------------
    # 9. Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics,
        confusion,
        report,
    )

    print("\n" + "#" * 60)
    print("# XGBOOST RISK MODEL COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()