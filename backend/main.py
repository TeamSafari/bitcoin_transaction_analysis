from pathlib import Path
import sys
import traceback


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PIPELINE IMPORTS
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
    main as run_deterministic,
)

from backend.risk.risk_fusion import (
    main as run_risk_fusion,
)

from backend.models.risk_model import (
    main as run_risk_model,
)


# ============================================================
# OPTIONAL LATER STAGES
# ============================================================

# These remain intentionally disabled until the complete
# upstream production chain has been verified.
#
# from backend.explainability.shap_engine import ...
# from backend.alerts.alert_service import ...


# ============================================================
# STAGE RUNNER
# ============================================================

def run_stage(
    name,
    function,
):

    print("\n")
    print("=" * 70)
    print(f"STARTING: {name}")
    print("=" * 70)

    try:

        function()

    except Exception as error:

        print("\n")
        print("#" * 70)
        print(f"FAILED: {name}")
        print("#" * 70)

        print(
            f"\n{error}"
        )

        traceback.print_exc()

        raise

    print(
        f"\nCOMPLETED: {name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 70)
    print("# BITCOIN TRANSACTION RISK PIPELINE")
    print("#" * 70)

    # --------------------------------------------------------
    # 1. Feature engineering
    # --------------------------------------------------------

    run_stage(
        "FEATURE ENGINEERING",
        run_feature_engineering,
    )

    # --------------------------------------------------------
    # 2. NetworkX
    # --------------------------------------------------------

    run_stage(
        "NETWORKX GRAPH ENGINE",
        run_graph_engine,
    )

    # --------------------------------------------------------
    # 3. GraphSAGE
    # --------------------------------------------------------

    run_stage(
        "GRAPHSAGE",
        run_graphsage,
    )

    # --------------------------------------------------------
    # 4. Isolation Forest
    # --------------------------------------------------------

    run_stage(
        "ISOLATION FOREST",
        run_isolation_forest,
    )

    # --------------------------------------------------------
    # 5. Autoencoder
    # --------------------------------------------------------

    run_stage(
        "AUTOENCODER",
        run_autoencoder,
    )

    # --------------------------------------------------------
    # 6. Deterministic statistical risk
    # --------------------------------------------------------

    run_stage(
        "DETERMINISTIC STATISTICAL RISK",
        run_deterministic,
    )

    # --------------------------------------------------------
    # 7. Risk feature fusion
    # --------------------------------------------------------

    run_stage(
        "RISK FEATURE FUSION",
        run_risk_fusion,
    )

    # --------------------------------------------------------
    # 8. XGBoost
    # --------------------------------------------------------

    run_stage(
        "XGBOOST RISK MODEL",
        run_risk_model,
    )

    print("\n")
    print("#" * 70)
    print("# CORE PIPELINE COMPLETED")
    print("#" * 70)

    print(
        "\nNext stages:"
    )

    print(
        "SHAP → Alert Service → AI Agent"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()