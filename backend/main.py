from pathlib import Path
import sys
import traceback


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

# Allow backend packages to be imported correctly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PIPELINE MODULES
# ============================================================

from backend.feature_engineering.feature_pipeline import (
    main as run_feature_engineering,
)

from backend.graph.graph_engine import (
    main as run_graph_engine,
)

from backend.graph.graphsage import (
    main as run_graphsage,
)

from backend.models.isolation_forest import (
    main as run_isolation_forest,
)

from backend.models.autoencoder import (
    main as run_autoencoder,
)

from backend.risk.run_deterministic import (
    main as run_deterministic_risk,
)

from backend.risk.risk_fusion import (
    main as run_risk_fusion,
)

from backend.models.risk_model import (
    main as run_risk_model,
)

from backend.explainability.shap_engine import (
    main as run_shap,
)

from backend.alerts.alert_service import (
    main as run_alert_service,
)


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_stage(
    stage_name,
    stage_function,
):

    print("\n")
    print("=" * 70)
    print(f"STARTING: {stage_name}")
    print("=" * 70)

    try:

        stage_function()

    except Exception as error:

        print("\n")
        print("#" * 70)
        print(f"PIPELINE FAILED: {stage_name}")
        print("#" * 70)

        print(
            f"\nError:\n{error}"
        )

        print(
            "\nTraceback:"
        )

        traceback.print_exc()

        raise

    print("\n")
    print("=" * 70)
    print(f"COMPLETED: {stage_name}")
    print("=" * 70)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("#" * 70)
    print("# BITCOIN TRANSACTION RISK PIPELINE")
    print("#" * 70)

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    # ========================================================
    # 1. FEATURE ENGINEERING
    # ========================================================

    run_stage(
        "FEATURE ENGINEERING",
        run_feature_engineering,
    )

    # ========================================================
    # 2. NETWORKX GRAPH ENGINE
    # ========================================================

    run_stage(
        "NETWORKX GRAPH ENGINE",
        run_graph_engine,
    )

    # ========================================================
    # 3. GRAPHSAGE EMBEDDINGS
    # ========================================================

    run_stage(
        "GRAPHSAGE EMBEDDINGS",
        run_graphsage,
    )

    # ========================================================
    # 4. ISOLATION FOREST
    # ========================================================

    run_stage(
        "ISOLATION FOREST",
        run_isolation_forest,
    )

    # ========================================================
    # 5. AUTOENCODER
    # ========================================================

    run_stage(
        "AUTOENCODER",
        run_autoencoder,
    )

    # ========================================================
    # 6. DETERMINISTIC STATISTICAL RISK
    # ========================================================

    run_stage(
        "DETERMINISTIC STATISTICAL RISK",
        run_deterministic_risk,
    )

    # ========================================================
    # 7. RISK FEATURE FUSION
    # ========================================================

    run_stage(
        "RISK FEATURE FUSION",
        run_risk_fusion,
    )

    # ========================================================
    # 8. XGBOOST RISK MODEL
    # ========================================================

    run_stage(
        "XGBOOST RISK MODEL",
        run_risk_model,
    )

    # ========================================================
    # 9. SHAP EXPLAINABILITY
    # ========================================================

    run_stage(
        "SHAP EXPLAINABILITY",
        run_shap,
    )

    # ========================================================
    # 10. ALERT SERVICE
    # ========================================================

    run_stage(
        "ALERT SERVICE",
        run_alert_service,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("#" * 70)
    print("# PIPELINE COMPLETED SUCCESSFULLY")
    print("#" * 70)

    print(
        "\nFinal alert outputs:"
    )

    print(
        PROJECT_ROOT
        / "outputs"
        / "alerts"
    )

    print(
        "\nFinal explainability outputs:"
    )

    print(
        PROJECT_ROOT
        / "outputs"
        / "explainability"
    )

    print("\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()