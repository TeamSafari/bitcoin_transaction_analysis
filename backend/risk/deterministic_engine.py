from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class DeterministicRiskEngine:
    """
    Pure statistical anomaly engine.

    No ground truth is used.

    The engine:
        1. Identifies numeric behavioral features.
        2. Calculates robust statistics using median and MAD.
        3. Converts each feature into an empirical percentile.
        4. Calculates a robust anomaly score.
        5. Aggregates feature-level evidence.
        6. Produces a deterministic statistical score.

    Important:
        The output is an anomaly/risk ranking, NOT a probability
        of criminal activity.
    """

    # Columns that are identifiers or metadata rather than
    # behavioral measurements.
    EXCLUDED_COLUMNS = {
        "wallet_id",
        "txid",
        "address",
        "src_ip",
        "dst_ip",
        "timestamp",
        "label",
        "scenario_id",
        "behavior_type",
        "wallet_type",
        "primary_entity_id",
        "geo_country",
        "asn",
    }

    def __init__(
        self,
        mad_epsilon: float = 1e-9,
    ):
        self.mad_epsilon = mad_epsilon

        self.feature_statistics = {}
        self.feature_columns = []

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    @staticmethod
    def _numeric_series(
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
    # FEATURE DISCOVERY
    # ========================================================

    def select_features(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Select numeric behavioral features.

        Identifiers, labels and categorical metadata are excluded.
        """

        candidates = []

        for column in df.columns:

            if column in self.EXCLUDED_COLUMNS:
                continue

            numeric = self._numeric_series(
                df[column]
            )

            # Require at least two valid observations.
            if numeric.notna().sum() < 2:
                continue

            # Constant columns carry no anomaly information.
            if numeric.dropna().nunique() <= 1:
                continue

            candidates.append(column)

        if not candidates:

            raise ValueError(
                "No valid numeric behavioral features "
                "were found for deterministic analysis."
            )

        return candidates

    # ========================================================
    # EMPIRICAL PERCENTILE
    # ========================================================

    @staticmethod
    def empirical_percentile(
        series: pd.Series,
    ) -> pd.Series:
        """
        Empirical CDF represented as percentile rank.

        No Gaussian or other distributional assumption.
        """

        values = (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
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
    # ROBUST Z-SCORE
    # ========================================================

    def robust_z_score(
        self,
        series: pd.Series,
    ) -> tuple[pd.Series, dict]:
        """
        Robust standardized deviation:

            robust_z =
                0.6745 * (x - median) / MAD

        0.6745 makes the statistic comparable to a normal
        z-score when the underlying distribution is normal.

        MAD is used instead of standard deviation because
        transaction/network data can contain extreme values.
        """

        values = self._numeric_series(
            series
        )

        median = values.median()

        absolute_deviation = (
            values - median
        ).abs()

        mad = absolute_deviation.median()

        if pd.isna(mad) or mad < self.mad_epsilon:

            # If MAD collapses, use percentile distance
            # rather than inventing a numerical scale.
            robust_z = pd.Series(
                0.0,
                index=series.index,
            )

        else:

            robust_z = (
                0.6745
                * (values - median)
                / mad
            )

        statistics = {
            "median": (
                float(median)
                if pd.notna(median)
                else None
            ),
            "mad": (
                float(mad)
                if pd.notna(mad)
                else None
            ),
        }

        return robust_z, statistics

    # ========================================================
    # FEATURE ANOMALY SCORE
    # ========================================================

    def feature_anomaly_score(
        self,
        series: pd.Series,
    ) -> tuple[pd.Series, dict]:
        """
        Convert a feature into a two-sided anomaly score.

        Both unusually low and unusually high observations
        are considered anomalous.

        This avoids arbitrary assumptions such as:
            "high transaction count = suspicious"

        unless the data itself demonstrates that behavior.
        """

        values = self._numeric_series(
            series
        )

        percentile = (
            self.empirical_percentile(
                values
            )
        )

        robust_z, statistics = (
            self.robust_z_score(
                values
            )
        )

        # Two-sided empirical deviation from the median.
        #
        # percentile = 0.5 represents the distribution midpoint.
        #
        # Distance from midpoint is normalized to [0, 1].
        percentile_anomaly = (
            2.0
            * (
                percentile
                - 0.5
            )
            .abs()
        )

        # Robust-z magnitude is useful as supporting evidence,
        # but percentile anomaly remains bounded and comparable
        # across heterogeneous features.
        robust_magnitude = (
            robust_z.abs()
        )

        statistics[
            "robust_z_median"
        ] = (
            float(
                robust_magnitude
                .median()
            )
            if robust_magnitude.notna().any()
            else 0.0
        )

        return (
            percentile_anomaly,
            statistics,
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df.empty:

            raise ValueError(
                "Input dataframe is empty."
            )

        if "wallet_id" not in df.columns:

            raise ValueError(
                "wallet_id column is required."
            )

        self.feature_columns = (
            self.select_features(df)
        )

        scores = pd.DataFrame(
            index=df.index
        )

        self.feature_statistics = {}

        # ----------------------------------------------------
        # Calculate feature-level statistical anomaly scores
        # ----------------------------------------------------

        for feature in self.feature_columns:

            feature_score, statistics = (
                self.feature_anomaly_score(
                    df[feature]
                )
            )

            scores[feature] = (
                feature_score
            )

            self.feature_statistics[
                feature
            ] = statistics

        # ----------------------------------------------------
        # Statistical aggregate
        # ----------------------------------------------------

        # Equal weighting is used here because the deterministic
        # engine should not manually privilege one feature.
        #
        # Features are first transformed into the same bounded
        # empirical scale [0, 1].

        deterministic_score = (
            scores.mean(
                axis=1,
                skipna=True,
            )
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        # ----------------------------------------------------
        # Feature count
        # ----------------------------------------------------

        feature_count = (
            scores.notna()
            .sum(axis=1)
        )

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        evidence = []

        for index in scores.index:

            row_scores = (
                scores.loc[index]
                .dropna()
                .sort_values(
                    ascending=False
                )
            )

            row_evidence = []

            # Keep strongest feature contributors.
            # This is output formatting, not a risk threshold.
            for feature, score in (
                row_scores.head(5).items()
            ):

                raw_value = df.loc[
                    index,
                    feature
                ]

                stats = (
                    self.feature_statistics[
                        feature
                    ]
                )

                row_evidence.append({

                    "feature":
                        feature,

                    "value":
                        self._safe_value(
                            raw_value
                        ),

                    "empirical_anomaly_score":
                        round(
                            float(score),
                            6,
                        ),

                    "median":
                        stats[
                            "median"
                        ],

                    "mad":
                        stats[
                            "mad"
                        ],
                })

            evidence.append(
                row_evidence
            )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        result = pd.DataFrame({

            "wallet_id":
                df["wallet_id"]
                .astype(str),

            "deterministic_score":
                deterministic_score,

            "deterministic_feature_count":
                feature_count,

            "deterministic_evidence":
                [
                    json.dumps(item)
                    for item in evidence
                ],
        })

        return result

    # ========================================================
    # SAFE VALUE
    # ========================================================

    @staticmethod
    def _safe_value(
        value,
    ):

        if pd.isna(value):
            return None

        if isinstance(
            value,
            (np.integer,),
        ):
            return int(value)

        if isinstance(
            value,
            (np.floating,),
        ):
            return float(value)

        return value

    # ========================================================
    # SAVE STATISTICS
    # ========================================================

    def save_statistics(
        self,
        output_file: Path,
    ):

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
                {
                    "method":
                        "empirical_percentile_plus_MAD",

                    "feature_count":
                        len(
                            self.feature_columns
                        ),

                    "features":
                        self.feature_columns,

                    "feature_statistics":
                        self.feature_statistics,
                },
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

    INPUT_FILE = (
        PROJECT_ROOT
        / "outputs"
        / "features"
        / "wallet_features.csv"
    )

    OUTPUT_DIR = (
        PROJECT_ROOT
        / "outputs"
        / "models"
    )

    OUTPUT_FILE = (
        OUTPUT_DIR
        / "deterministic_scores.csv"
    )

    STATISTICS_FILE = (
        OUTPUT_DIR
        / "deterministic_statistics.json"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    engine = DeterministicRiskEngine()

    result = engine.run(
        data
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    engine.save_statistics(
        STATISTICS_FILE
    )

    print("=" * 60)
    print("DETERMINISTIC STATISTICAL ENGINE")
    print("=" * 60)

    print(
        f"Wallets processed: {len(result)}"
    )

    print(
        f"Features analyzed: "
        f"{len(engine.feature_columns)}"
    )

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )

    print(
        f"Statistics: {STATISTICS_FILE}"
    )