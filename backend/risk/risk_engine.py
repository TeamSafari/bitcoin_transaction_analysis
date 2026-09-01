from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


class StatisticalRiskEngine:
    """
    Purely statistical risk aggregation engine.

    No:
        - manually assigned model weights
        - hardcoded feature thresholds
        - ground-truth leakage
        - heuristic risk bonuses

    Method:
        1. Convert every available detector score to an
           empirical percentile.
        2. Measure redundancy using absolute Spearman
           correlation.
        3. Assign inverse-redundancy weights.
        4. Calculate the weighted mean percentile.
        5. Convert the final score to an empirical percentile.
        6. Assign risk bands from empirical quartiles.
    """

    SCORE_COLUMNS = {
        "risk_model": "risk_model_probability",
        "isolation_forest":
            "isolation_forest_anomaly_score",
        "autoencoder":
            "autoencoder_reconstruction_error",
        "deterministic":
            "deterministic_score",
    }

    def __init__(self):

        self.source_weights = {}
        self.correlation_matrix = None

    # ========================================================
    # CLEAN NUMERIC SERIES
    # ========================================================

    @staticmethod
    def clean_series(
        series: pd.Series,
    ) -> pd.Series:

        return (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    # ========================================================
    # EMPIRICAL PERCENTILE
    # ========================================================

    @staticmethod
    def empirical_percentile(
        series: pd.Series,
    ) -> pd.Series:
        """
        Empirical CDF / percentile rank.

        No distributional assumption is made.

        0.50 ≈ median
        0.95 ≈ upper 5% of the batch
        """

        values = (
            StatisticalRiskEngine
            .clean_series(series)
        )

        result = pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

        valid = values.notna()

        if valid.sum() == 0:
            return result

        result.loc[valid] = (
            values.loc[valid]
            .rank(
                method="average",
                pct=True,
            )
        )

        return result

    # ========================================================
    # BUILD SCORE MATRIX
    # ========================================================

    def build_score_matrix(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        matrix = pd.DataFrame(
            index=df.index
        )

        for source, column in (
            self.SCORE_COLUMNS.items()
        ):

            if column not in df.columns:
                continue

            matrix[source] = (
                self.clean_series(
                    df[column]
                )
            )

        if matrix.empty:

            raise ValueError(
                "No supported risk-score columns "
                "were found."
            )

        return matrix

    # ========================================================
    # CONVERT TO EMPIRICAL PERCENTILES
    # ========================================================

    def build_percentile_matrix(
        self,
        score_matrix: pd.DataFrame,
    ) -> pd.DataFrame:

        result = pd.DataFrame(
            index=score_matrix.index
        )

        for column in score_matrix.columns:

            result[column] = (
                self.empirical_percentile(
                    score_matrix[column]
                )
            )

        return result

    # ========================================================
    # CORRELATION-ADJUSTED WEIGHTS
    # ========================================================

    def calculate_source_weights(
        self,
        percentile_matrix: pd.DataFrame,
    ) -> dict:
        """
        Calculate weights entirely from observed data.

        Highly correlated detectors are treated as more
        redundant.

        For each detector:

            redundancy =
                mean absolute Spearman correlation
                with the other detectors

            raw_weight =
                1 / (1 + redundancy)

        Finally, weights are normalized to sum to 1.

        No manually specified weights are used.
        """

        columns = list(
            percentile_matrix.columns
        )

        if len(columns) == 1:

            return {
                columns[0]: 1.0
            }

        correlation = (
            percentile_matrix
            .corr(
                method="spearman"
            )
            .abs()
        )

        self.correlation_matrix = (
            correlation
        )

        raw_weights = {}

        for source in columns:

            others = [
                c
                for c in columns
                if c != source
            ]

            correlations = (
                correlation.loc[
                    source,
                    others
                ]
                .dropna()
            )

            if correlations.empty:

                redundancy = 0.0

            else:

                redundancy = float(
                    correlations.mean()
                )

            raw_weights[source] = (
                1.0
                /
                (
                    1.0
                    +
                    redundancy
                )
            )

        total = sum(
            raw_weights.values()
        )

        if total == 0:

            equal_weight = (
                1.0 / len(columns)
            )

            return {
                source: equal_weight
                for source in columns
            }

        return {
            source:
                weight / total
            for source, weight
            in raw_weights.items()
        }

    # ========================================================
    # STATISTICAL FUSION
    # ========================================================

    def aggregate_scores(
        self,
        percentile_matrix: pd.DataFrame,
        weights: dict,
    ) -> pd.Series:
        """
        Correlation-adjusted weighted mean of empirical
        percentiles.

        This produces a relative anomaly/risk score
        within the current batch.
        """

        numerator = pd.Series(
            0.0,
            index=percentile_matrix.index,
        )

        denominator = pd.Series(
            0.0,
            index=percentile_matrix.index,
        )

        for source in percentile_matrix.columns:

            values = percentile_matrix[
                source
            ]

            weight = weights[
                source
            ]

            valid = values.notna()

            numerator.loc[valid] += (
                weight
                * values.loc[valid]
            )

            denominator.loc[valid] += (
                weight
            )

        result = (
            numerator
            /
            denominator.replace(
                0,
                np.nan,
            )
        )

        return result.fillna(0.0)

    # ========================================================
    # RISK PERCENTILE
    # ========================================================

    @staticmethod
    def calculate_risk_percentile(
        score: pd.Series,
    ) -> pd.Series:

        return (
            score
            .rank(
                method="average",
                pct=True,
            )
        )

    # ========================================================
    # EMPIRICAL RISK BAND
    # ========================================================

    @staticmethod
    def assign_risk_level(
        percentile: float,
    ) -> str:
        """
        Risk bands are based entirely on empirical
        position within the current batch.

        0.00 - 0.25 -> LOW
        0.25 - 0.50 -> MEDIUM
        0.50 - 0.75 -> HIGH
        0.75 - 1.00 -> CRITICAL

        These are rank bands, NOT probabilities.
        """

        if percentile <= 0.25:
            return "LOW"

        if percentile <= 0.50:
            return "MEDIUM"

        if percentile <= 0.75:
            return "HIGH"

        return "CRITICAL"

    # ========================================================
    # RUN ENGINE
    # ========================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if "wallet_id" not in df.columns:

            raise ValueError(
                "wallet_id column is required."
            )

        # ----------------------------------------------------
        # 1. Build detector matrix
        # ----------------------------------------------------

        score_matrix = (
            self.build_score_matrix(
                df
            )
        )

        # ----------------------------------------------------
        # 2. Convert different score scales into
        #    empirical percentiles
        # ----------------------------------------------------

        percentile_matrix = (
            self.build_percentile_matrix(
                score_matrix
            )
        )

        # ----------------------------------------------------
        # 3. Calculate data-driven weights
        # ----------------------------------------------------

        weights = (
            self.calculate_source_weights(
                percentile_matrix
            )
        )

        self.source_weights = weights

        # ----------------------------------------------------
        # 4. Statistical fusion
        # ----------------------------------------------------

        final_score = (
            self.aggregate_scores(
                percentile_matrix,
                weights,
            )
        )

        # ----------------------------------------------------
        # 5. Position in current batch
        # ----------------------------------------------------

        risk_percentile = (
            self.calculate_risk_percentile(
                final_score
            )
        )

        # ----------------------------------------------------
        # 6. Statistical risk band
        # ----------------------------------------------------

        risk_level = (
            risk_percentile
            .apply(
                self.assign_risk_level
            )
        )

        # ----------------------------------------------------
        # 7. Output
        # ----------------------------------------------------

        result = pd.DataFrame({

            "wallet_id":
                df["wallet_id"]
                .astype(str),

            "risk_model_percentile":
                percentile_matrix.get(
                    "risk_model",
                    pd.Series(
                        np.nan,
                        index=df.index,
                    ),
                ),

            "isolation_forest_percentile":
                percentile_matrix.get(
                    "isolation_forest",
                    pd.Series(
                        np.nan,
                        index=df.index,
                    ),
                ),

            "autoencoder_percentile":
                percentile_matrix.get(
                    "autoencoder",
                    pd.Series(
                        np.nan,
                        index=df.index,
                    ),
                ),

            "deterministic_percentile":
                percentile_matrix.get(
                    "deterministic",
                    pd.Series(
                        np.nan,
                        index=df.index,
                    ),
                ),

            "statistical_aggregate_score":
                final_score,

            "risk_percentile":
                risk_percentile,

            "risk_level":
                risk_level,
        })

        return result

    # ========================================================
    # SAVE STATISTICAL METADATA
    # ========================================================

    def save_metadata(
        self,
        output_file: Path,
    ):

        metadata = {

            "engine_type":
                "pure_statistical",

            "normalization":
                "empirical_percentile_rank",

            "correlation_method":
                "absolute_spearman",

            "weighting_method":
                "inverse_redundancy",

            "aggregation":
                "correlation_adjusted_weighted_mean",

            "risk_band_method":
                "empirical_quartile_rank",

            "source_weights":
                self.source_weights,
        }

        if self.correlation_matrix is not None:

            metadata[
                "correlation_matrix"
            ] = (
                self.correlation_matrix
                .round(6)
                .to_dict()
            )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
            )


# ============================================================
# SCRIPT ENTRY
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    MODEL_DIR = (
        PROJECT_ROOT
        / "outputs"
        / "models"
    )

    INPUT_FILE = (
        MODEL_DIR
        / "risk_engine_input.csv"
    )

    OUTPUT_FILE = (
        MODEL_DIR
        / "risk_scores.csv"
    )

    METADATA_FILE = (
        MODEL_DIR
        / "risk_engine_metadata.json"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Expected columns are any available "
            "combination of:\n"
            "risk_model_probability\n"
            "isolation_forest_anomaly_score\n"
            "autoencoder_reconstruction_error\n"
            "deterministic_score"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    engine = StatisticalRiskEngine()

    result = engine.run(
        data
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    engine.save_metadata(
        METADATA_FILE
    )

    print("=" * 60)
    print("STATISTICAL RISK ENGINE COMPLETE")
    print("=" * 60)

    print(
        f"Wallets processed: {len(result)}"
    )

    print(
        "\nStatistical source weights:"
    )

    for source, weight in (
        engine.source_weights.items()
    ):

        print(
            f"{source}: {weight:.4f}"
        )

    print(
        "\nRisk distribution:"
    )

    print(
        result[
            "risk_level"
        ].value_counts()
    )

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )