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

from sklearn.model_selection import (
    train_test_split,
)

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "artifacts"
)


TRAINING_FILE = (
    MODEL_OUTPUT_DIR
    / "risk_training_features.csv"
)

MODEL_FILE = (
    ARTIFACT_DIR
    / "risk_model.pkl"
)

ALL_PREDICTIONS_FILE = (
    MODEL_OUTPUT_DIR
    / "risk_model_predictions.csv"
)

TEST_PREDICTIONS_FILE = (
    MODEL_OUTPUT_DIR
    / "risk_model_test_predictions.csv"
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

    "n_estimators":
        300,

    "max_depth":
        5,

    "learning_rate":
        0.05,

    "subsample":
        0.8,

    "colsample_bytree":
        0.8,

    "min_child_weight":
        3,

    "reg_alpha":
        0.1,

    "reg_lambda":
        1.0,

    "objective":
        "binary:logistic",

    "eval_metric":
        "logloss",

    "random_state":
        RANDOM_STATE,

    "n_jobs":
        -1,
}


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    if not TRAINING_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found:\n"
            f"{TRAINING_FILE}"
        )

    df = pd.read_csv(
        TRAINING_FILE
    )

    if df.empty:

        raise ValueError(
            "Risk training dataset is empty."
        )

    required = {
        "wallet_id",
        TARGET_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Training dataset is missing "
            f"columns: {sorted(missing)}"
        )

    df["wallet_id"] = (
        df["wallet_id"]
        .astype(str)
    )

    if df["wallet_id"].duplicated().any():

        raise ValueError(
            "Duplicate wallet_id values "
            "found in training data."
        )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df,
):

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    if y.isna().any():

        raise ValueError(
            "Risk labels contain invalid "
            "or missing values."
        )

    y = y.astype(int)

    if not set(
        y.unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Risk labels must contain only "
            "0 and 1."
        )

    # --------------------------------------------------------
    # Model features
    #
    # wallet_id and label are never features.
    # --------------------------------------------------------

    excluded_columns = {
        "wallet_id",
        TARGET_COLUMN,
    }

    X = pd.DataFrame(
        index=df.index
    )

    for column in df.columns:

        if column in excluded_columns:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric.notna().sum() > 0:

            X[column] = numeric

    if X.empty:

        raise ValueError(
            "No numeric features available "
            "for XGBoost."
        )

    # --------------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column in X.columns:

        median = (
            X[column]
            .median()
        )

        if pd.isna(median):

            median = 0.0

        X[column] = (
            X[column]
            .fillna(median)
        )

    # --------------------------------------------------------
    # Remove constant features
    # --------------------------------------------------------

    variable_columns = [
        column
        for column in X.columns
        if X[column].nunique() > 1
    ]

    X = X[
        variable_columns
    ]

    if X.empty:

        raise ValueError(
            "All model features are constant."
        )

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
):

    positive_count = int(
        (y_train == 1).sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    if positive_count == 0:

        raise ValueError(
            "Training data contains no "
            "positive samples."
        )

    if negative_count == 0:

        raise ValueError(
            "Training data contains no "
            "negative samples."
        )

    params = dict(
        XGB_PARAMS
    )

    # --------------------------------------------------------
    # Class imbalance adjustment.
    #
    # This is statistically motivated by the observed
    # training class frequencies rather than a hardcoded
    # fraud weight.
    # --------------------------------------------------------

    params[
        "scale_pos_weight"
    ] = (
        negative_count
        / positive_count
    )

    model = XGBClassifier(
        **params
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    metrics = {}

    metrics["accuracy"] = (
        accuracy_score(
            y_test,
            prediction,
        )
    )

    metrics["precision"] = (
        precision_score(
            y_test,
            prediction,
            zero_division=0,
        )
    )

    metrics["recall"] = (
        recall_score(
            y_test,
            prediction,
            zero_division=0,
        )
    )

    metrics["f1"] = (
        f1_score(
            y_test,
            prediction,
            zero_division=0,
        )
    )

    # ROC-AUC requires both classes in y_test.
    if len(
        np.unique(y_test)
    ) == 2:

        metrics["roc_auc"] = (
            roc_auc_score(
                y_test,
                probability,
            )
        )

    else:

        metrics["roc_auc"] = np.nan

    return (
        probability,
        prediction,
        metrics,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("XGBOOST RISK MODEL")
    print("=" * 60)

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_training_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y = prepare_features(
        df
    )

    wallet_ids = (
        df["wallet_id"]
        .astype(str)
        .values
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_wallet_ids,
        test_wallet_ids,
    ) = train_test_split(
        X,
        y,
        wallet_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Total wallets: "
        f"{len(df)}"
    )

    print(
        f"Features: "
        f"{X.shape[1]}"
    )

    print(
        f"Training wallets: "
        f"{len(X_train)}"
    )

    print(
        f"Test wallets: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate on untouched test set
    # --------------------------------------------------------

    (
        test_probability,
        test_prediction,
        metrics,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_output = pd.DataFrame(
        {
            "wallet_id":
                test_wallet_ids,

            "actual_label":
                y_test.values,

            "risk_probability":
                test_probability,

            "risk_prediction":
                test_prediction,
        }
    )

    test_output.to_csv(
        TEST_PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # All-wallet inference
    #
    # This is the downstream production/batch output.
    # --------------------------------------------------------

    all_probability = (
        model.predict_proba(
            X
        )[:, 1]
    )

    all_prediction = (
        all_probability >= 0.5
    ).astype(int)

    all_output = pd.DataFrame(
        {
            "wallet_id":
                wallet_ids,

            "risk_probability":
                all_probability,

            "risk_prediction":
                all_prediction,
        }
    )

    all_output.to_csv(
        ALL_PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        {
            "model":
                model,

            "feature_columns":
                X.columns.tolist(),

            "target_column":
                TARGET_COLUMN,

            "model_type":
                "XGBClassifier",

            "random_state":
                RANDOM_STATE,
        },
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    confusion = confusion_matrix(
        y_test,
        test_prediction,
    )

    report = classification_report(
        y_test,
        test_prediction,
        zero_division=0,
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

        for name, value in (
            metrics.items()
        ):

            if pd.isna(value):

                file.write(
                    f"{name}: N/A\n"
                )

            else:

                file.write(
                    f"{name}: "
                    f"{value:.6f}\n"
                )

        file.write(
            "\nConfusion Matrix\n"
        )

        file.write(
            str(confusion)
            + "\n"
        )

        file.write(
            "\nClassification Report\n"
        )

        file.write(
            report
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("XGBOOST COMPLETE")
    print("=" * 60)

    for name, value in (
        metrics.items()
    ):

        if pd.isna(value):

            print(
                f"{name}: N/A"
            )

        else:

            print(
                f"{name}: "
                f"{value:.4f}"
            )

    print(
        f"\nModel:\n"
        f"{MODEL_FILE}"
    )

    print(
        f"\nAll-wallet predictions:\n"
        f"{ALL_PREDICTIONS_FILE}"
    )

    print(
        f"\nTest predictions:\n"
        f"{TEST_PREDICTIONS_FILE}"
    )

    print(
        f"\nMetrics:\n"
        f"{METRICS_FILE}"
    )

    return model


if __name__ == "__main__":
    main()