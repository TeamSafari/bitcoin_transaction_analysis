from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

# backend/explainability/shap_engine.py
#
# parents[0] -> explainability
# parents[1] -> backend
# parents[2] -> project root

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "artifacts"
    / "risk_model.pkl"
)

TRAINING_FILE = (
    OUTPUT_DIR
    / "models"
    / "risk_training_features.csv"
)

EXPLAINABILITY_DIR = (
    OUTPUT_DIR
    / "explainability"
)


# ============================================================
# OUTPUT FILES
# ============================================================

SHAP_VALUES_FILE = (
    EXPLAINABILITY_DIR
    / "shap_values.csv"
)

TOP_CONTRIBUTIONS_FILE = (
    EXPLAINABILITY_DIR
    / "top_feature_contributions.csv"
)

GLOBAL_IMPORTANCE_FILE = (
    EXPLAINABILITY_DIR
    / "global_feature_importance.csv"
)

GLOBAL_IMPORTANCE_PLOT = (
    EXPLAINABILITY_DIR
    / "global_feature_importance.png"
)

BEESWARM_PLOT = (
    EXPLAINABILITY_DIR
    / "shap_beeswarm.png"
)

VIOLIN_PLOT = (
    EXPLAINABILITY_DIR
    / "shap_violin.png"
)

HEATMAP_PLOT = (
    EXPLAINABILITY_DIR
    / "shap_heatmap.png"
)

WATERFALL_PLOT = (
    EXPLAINABILITY_DIR
    / "shap_waterfall_top_risk.png"
)

DECISION_PLOT = (
    EXPLAINABILITY_DIR
    / "shap_decision_top_risk.png"
)

DEPENDENCE_PLOT = (
    EXPLAINABILITY_DIR
    / "shap_dependence_top_features.png"
)

INTERACTION_PLOT = (
    EXPLAINABILITY_DIR
    / "shap_interaction_heatmap.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_N_FEATURES = 20

TOP_N_EXPLANATIONS = 10

TOP_N_DEPENDENCE_FEATURES = 6

TOP_N_INTERACTION_FEATURES = 12

RANDOM_STATE = 42


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Risk model not found:\n"
            f"{MODEL_FILE}"
        )

    artifact = joblib.load(
        MODEL_FILE
    )

    if not isinstance(
        artifact,
        dict,
    ):

        raise ValueError(
            "risk_model.pkl must contain "
            "a dictionary."
        )

    if "model" not in artifact:

        raise ValueError(
            "'model' missing from "
            "risk_model.pkl."
        )

    if "feature_columns" not in artifact:

        raise ValueError(
            "'feature_columns' missing from "
            "risk_model.pkl."
        )

    model = artifact["model"]

    feature_columns = list(
        artifact["feature_columns"]
    )

    return (
        model,
        feature_columns,
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not TRAINING_FILE.exists():

        raise FileNotFoundError(
            f"Training data not found:\n"
            f"{TRAINING_FILE}"
        )

    df = pd.read_csv(
        TRAINING_FILE
    )

    if df.empty:

        raise ValueError(
            "risk_training_features.csv "
            "is empty."
        )

    if "wallet_id" not in df.columns:

        raise ValueError(
            "wallet_id is required."
        )

    return df


# ============================================================
# PREPARE MODEL FEATURES
# ============================================================

def prepare_features(
    df,
    feature_columns,
):

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Features expected by the "
            "trained XGBoost model are missing:\n"
            f"{missing}"
        )

    X = df[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in feature_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove infinities
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    for column in feature_columns:

        median = X[column].median()

        if pd.isna(median):
            median = 0.0

        X[column] = (
            X[column]
            .fillna(median)
        )

    return X


# ============================================================
# SHAP CALCULATION
# ============================================================

def calculate_shap(
    model,
    X,
):

    print(
        "\nCalculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    explanation = explainer(
        X
    )

    shap_values = np.asarray(
        explanation.values
    )

    # --------------------------------------------------------
    # Binary classification compatibility
    # --------------------------------------------------------

    if shap_values.ndim == 3:

        shap_values = (
            shap_values[:, :, -1]
        )

    if shap_values.ndim != 2:

        raise ValueError(
            "Unexpected SHAP shape: "
            f"{shap_values.shape}"
        )

    return (
        explainer,
        explanation,
        shap_values,
    )


# ============================================================
# GLOBAL IMPORTANCE
# ============================================================

def create_global_importance(
    shap_values,
    feature_columns,
):

    mean_abs_shap = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    importance = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "mean_absolute_shap":
                mean_abs_shap,
        }
    )

    importance = (
        importance
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    importance.to_csv(
        GLOBAL_IMPORTANCE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_df = (
        importance
        .head(TOP_N_FEATURES)
        .sort_values(
            "mean_absolute_shap"
        )
    )

    plt.figure(
        figsize=(11, 8)
    )

    plt.barh(
        plot_df["feature"],
        plot_df[
            "mean_absolute_shap"
        ],
    )

    plt.xlabel(
        "Mean |SHAP value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Global SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        GLOBAL_IMPORTANCE_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return importance


# ============================================================
# BEESWARM
# ============================================================

def create_beeswarm(
    shap_values,
    X,
):

    explanation = shap.Explanation(
        values=shap_values,
        data=X.values,
        feature_names=X.columns.tolist(),
    )

    shap.plots.beeswarm(
        explanation,
        max_display=TOP_N_FEATURES,
        show=False,
    )

    plt.title(
        "SHAP Beeswarm: Risk Feature Effects"
    )

    plt.tight_layout()

    plt.savefig(
        BEESWARM_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# VIOLIN
# ============================================================

def create_violin(
    shap_values,
    X,
):

    explanation = shap.Explanation(
        values=shap_values,
        data=X.values,
        feature_names=X.columns.tolist(),
    )

    shap.plots.violin(
        explanation,
        max_display=TOP_N_FEATURES,
        show=False,
    )

    plt.title(
        "SHAP Violin: Distribution of Feature Effects"
    )

    plt.tight_layout()

    plt.savefig(
        VIOLIN_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# SHAP HEATMAP
# ============================================================

def create_heatmap(
    shap_values,
    X,
):

    # Limit the heatmap to the most important
    # features so it remains interpretable.

    importance = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    top_indices = np.argsort(
        importance
    )[::-1][
        :TOP_N_FEATURES
    ]

    top_features = [
        X.columns[index]
        for index in top_indices
    ]

    top_values = (
        shap_values[
            :,
            top_indices
        ]
    )

    heatmap_df = pd.DataFrame(
        top_values,
        columns=top_features,
    )

    # Limit number of rows for readability.
    max_rows = 150

    if len(heatmap_df) > max_rows:

        heatmap_df = (
            heatmap_df
            .iloc[:max_rows]
        )

    plt.figure(
        figsize=(14, 9)
    )

    plt.imshow(
        heatmap_df.T,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="SHAP value"
    )

    plt.yticks(
        range(
            len(top_features)
        ),
        top_features,
    )

    plt.xlabel(
        "Wallet samples"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SHAP Heatmap Across Wallets"
    )

    plt.tight_layout()

    plt.savefig(
        HEATMAP_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# TOP WALLET
# ============================================================

def find_highest_risk_wallet(
    model,
    df,
    X,
):

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    index = int(
        np.argmax(
            probabilities
        )
    )

    wallet_id = str(
        df.iloc[index][
            "wallet_id"
        ]
    )

    return (
        index,
        wallet_id,
        probabilities,
    )


# ============================================================
# WATERFALL
# ============================================================

def create_waterfall(
    explanation,
    row_index,
    wallet_id,
    risk_probability,
):

    row_explanation = (
        explanation[
            row_index
        ]
    )

    plt.figure(
        figsize=(12, 9)
    )

    shap.plots.waterfall(
        row_explanation,
        max_display=TOP_N_EXPLANATIONS,
        show=False,
    )

    plt.title(
        "Individual Risk Explanation\n"
        f"Wallet: {wallet_id} | "
        f"Risk probability: "
        f"{risk_probability:.4f}"
    )

    plt.tight_layout()

    plt.savefig(
        WATERFALL_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# DECISION PLOT
# ============================================================

def create_decision_plot(
    explanation,
    row_index,
    wallet_id,
    risk_probability,
):

    row_explanation = (
        explanation[
            row_index
        ]
    )

    plt.figure(
        figsize=(13, 8)
    )

    shap.decision_plot(
        row_explanation.base_values,
        row_explanation.values,
        row_explanation.data,
        feature_names=row_explanation.feature_names,
        show=False,
    )

    plt.title(
        "SHAP Decision Path\n"
        f"Wallet: {wallet_id} | "
        f"Risk probability: "
        f"{risk_probability:.4f}"
    )

    plt.tight_layout()

    plt.savefig(
        DECISION_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# DEPENDENCE PLOTS
# ============================================================

def create_dependence_plots(
    shap_values,
    X,
    importance,
):

    top_features = (
        importance
        .head(
            TOP_N_DEPENDENCE_FEATURES
        )["feature"]
        .tolist()
    )

    # --------------------------------------------------------
    # One combined multi-feature dependence figure
    # --------------------------------------------------------

    for feature in top_features:

        plt.figure(
            figsize=(9, 6)
        )

        shap.dependence_plot(
            feature,
            shap_values,
            X,
            interaction_index="auto",
            show=False,
        )

        plt.title(
            f"SHAP Dependence: {feature}"
        )

        plt.tight_layout()

        safe_name = (
            feature
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

        output_file = (
            EXPLAINABILITY_DIR
            / f"shap_dependence_{safe_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# SHAP INTERACTION VALUES
# ============================================================

def create_interaction_heatmap(
    model,
    X,
    importance,
):

    print(
        "\nCalculating SHAP interactions..."
    )

    top_features = (
        importance
        .head(
            TOP_N_INTERACTION_FEATURES
        )["feature"]
        .tolist()
    )

    # --------------------------------------------------------
    # TreeExplainer interaction values
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    try:

        interaction_values = (
            explainer
            .shap_interaction_values(
                X
            )
        )

    except Exception as error:

        print(
            "Interaction SHAP could not "
            "be calculated."
        )

        print(
            f"Reason: {error}"
        )

        return

    interaction_values = np.asarray(
        interaction_values
    )

    # Binary classification compatibility.
    if interaction_values.ndim == 4:

        interaction_values = (
            interaction_values[:, :, :, -1]
        )

    if interaction_values.ndim != 3:

        print(
            "Skipping interaction heatmap "
            "because of unexpected shape: "
            f"{interaction_values.shape}"
        )

        return

    feature_indices = [
        X.columns.get_loc(
            feature
        )
        for feature in top_features
    ]

    selected = (
        interaction_values[
            :,
            feature_indices,
            :,
        ]
    )

    selected = (
        selected[
            :,
            :,
            feature_indices
        ]
    )

    mean_interaction = (
        np.abs(selected)
        .mean(axis=0)
    )

    plt.figure(
        figsize=(11, 9)
    )

    plt.imshow(
        mean_interaction,
        interpolation="nearest",
        aspect="auto",
    )

    plt.colorbar(
        label="Mean |SHAP interaction|"
    )

    plt.xticks(
        range(
            len(top_features)
        ),
        top_features,
        rotation=90,
    )

    plt.yticks(
        range(
            len(top_features)
        ),
        top_features,
    )

    plt.xlabel(
        "Feature"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "SHAP Feature Interaction Strength"
    )

    plt.tight_layout()

    plt.savefig(
        INTERACTION_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# PER-WALLET CONTRIBUTIONS
# ============================================================

def save_top_contributions(
    df,
    X,
    shap_values,
):

    records = []

    for row_index in range(
        len(X)
    ):

        wallet_id = str(
            df.iloc[row_index][
                "wallet_id"
            ]
        )

        row_values = (
            X.iloc[row_index]
        )

        row_shap = (
            shap_values[row_index]
        )

        order = np.argsort(
            np.abs(row_shap)
        )[::-1]

        top_indices = order[
            :TOP_N_EXPLANATIONS
        ]

        for rank, feature_index in enumerate(
            top_indices,
            start=1,
        ):

            value = float(
                row_values.iloc[
                    feature_index
                ]
            )

            shap_value = float(
                row_shap[
                    feature_index
                ]
            )

            if shap_value > 0:

                direction = (
                    "increases_risk"
                )

            elif shap_value < 0:

                direction = (
                    "decreases_risk"
                )

            else:

                direction = (
                    "neutral"
                )

            records.append(
                {
                    "wallet_id":
                        wallet_id,

                    "rank":
                        rank,

                    "feature":
                        X.columns[
                            feature_index
                        ],

                    "feature_value":
                        value,

                    "shap_value":
                        shap_value,

                    "absolute_shap_value":
                        abs(shap_value),

                    "direction":
                        direction,
                }
            )

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        TOP_CONTRIBUTIONS_FILE,
        index=False,
    )

    return result


# ============================================================
# SAVE RAW SHAP VALUES
# ============================================================

def save_shap_values(
    df,
    shap_values,
    feature_columns,
):

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_columns,
    )

    output = pd.concat(
        [
            df[
                ["wallet_id"]
            ].reset_index(
                drop=True
            ),

            shap_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output.to_csv(
        SHAP_VALUES_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 60)
    print("# SHAP EXPLAINABILITY ENGINE")
    print("#" * 60)

    EXPLAINABILITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------

    (
        model,
        feature_columns,
    ) = load_model()

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # 2. Load final risk dataset
    # --------------------------------------------------------

    df = load_data()

    print(
        f"Rows: {len(df)}"
    )

    # --------------------------------------------------------
    # 3. Prepare exact model features
    # --------------------------------------------------------

    X = prepare_features(
        df,
        feature_columns,
    )

    # --------------------------------------------------------
    # 4. Calculate SHAP
    # --------------------------------------------------------

    (
        explainer,
        explanation,
        shap_values,
    ) = calculate_shap(
        model,
        X,
    )

    # --------------------------------------------------------
    # 5. Global importance
    # --------------------------------------------------------

    importance = (
        create_global_importance(
            shap_values,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # 6. Beeswarm
    # --------------------------------------------------------

    create_beeswarm(
        shap_values,
        X,
    )

    # --------------------------------------------------------
    # 7. Violin
    # --------------------------------------------------------

    create_violin(
        shap_values,
        X,
    )

    # --------------------------------------------------------
    # 8. Heatmap
    # --------------------------------------------------------

    create_heatmap(
        shap_values,
        X,
    )

    # --------------------------------------------------------
    # 9. Highest-risk wallet
    # --------------------------------------------------------

    (
        top_index,
        top_wallet,
        probabilities,
    ) = find_highest_risk_wallet(
        model,
        df,
        X,
    )

    top_probability = float(
        probabilities[top_index]
    )

    print(
        f"\nHighest-risk wallet: "
        f"{top_wallet}"
    )

    print(
        f"Risk probability: "
        f"{top_probability:.4f}"
    )

    # --------------------------------------------------------
    # 10. Waterfall
    # --------------------------------------------------------

    create_waterfall(
        explanation,
        top_index,
        top_wallet,
        top_probability,
    )

    # --------------------------------------------------------
    # 11. Decision plot
    # --------------------------------------------------------

    create_decision_plot(
        explanation,
        top_index,
        top_wallet,
        top_probability,
    )

    # --------------------------------------------------------
    # 12. Dependence plots
    # --------------------------------------------------------

    create_dependence_plots(
        shap_values,
        X,
        importance,
    )

    # --------------------------------------------------------
    # 13. Interaction heatmap
    # --------------------------------------------------------

    create_interaction_heatmap(
        model,
        X,
        importance,
    )

    # --------------------------------------------------------
    # 14. Per-wallet contributions
    # --------------------------------------------------------

    save_top_contributions(
        df,
        X,
        shap_values,
    )

    # --------------------------------------------------------
    # 15. Raw SHAP values
    # --------------------------------------------------------

    save_shap_values(
        df,
        shap_values,
        feature_columns,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 60)

    print(
        "\nGenerated:"
    )

    print(
        f"  {GLOBAL_IMPORTANCE_PLOT}"
    )

    print(
        f"  {BEESWARM_PLOT}"
    )

    print(
        f"  {VIOLIN_PLOT}"
    )

    print(
        f"  {HEATMAP_PLOT}"
    )

    print(
        f"  {WATERFALL_PLOT}"
    )

    print(
        f"  {DECISION_PLOT}"
    )

    print(
        "  shap_dependence_<feature>.png"
    )

    print(
        f"  {INTERACTION_PLOT}"
    )

    print(
        f"\nData:"
    )

    print(
        f"  {SHAP_VALUES_FILE}"
    )

    print(
        f"  {TOP_CONTRIBUTIONS_FILE}"
    )

    print(
        f"  {GLOBAL_IMPORTANCE_FILE}"
    )


if __name__ == "__main__":
    main()