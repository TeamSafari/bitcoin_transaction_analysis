from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_rank(
    series: pd.Series
) -> pd.Series:
    """
    Empirical percentile rank.

    0 = relatively low
    1 = relatively high
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    return values.rank(
        method="average",
        pct=True
    )


def robust_statistics(
    series: pd.Series
) -> dict:
    """
    Calculate robust distribution statistics.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if values.empty:

        return {
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "median": float(
            values.median()
        ),

        "q1": float(
            values.quantile(0.25)
        ),

        "q3": float(
            values.quantile(0.75)
        ),

        "iqr": float(
            values.quantile(0.75)
            -
            values.quantile(0.25)
        ),

        "p90": float(
            values.quantile(0.90)
        ),

        "p95": float(
            values.quantile(0.95)
        ),

        "p99": float(
            values.quantile(0.99)
        ),
    }


def robust_zscore(
    series: pd.Series
) -> pd.Series:
    """
    Robust z-score using median and MAD.

    z = 0.6745 * (x - median) / MAD

    This is less sensitive to extreme outliers
    than the ordinary mean/std z-score.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    median = values.median()

    mad = np.median(
        np.abs(
            values - median
        )
    )

    if mad == 0:

        return pd.Series(
            np.zeros(len(values)),
            index=values.index
        )

    return (
        0.6745
        *
        (values - median)
        /
        mad
    )


def upper_tail_flag(
    series: pd.Series,
    percentile: float = 0.95
) -> pd.Series:
    """
    Flag values in the upper statistical tail.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    threshold = values.quantile(
        percentile
    )

    return values >= threshold


def lower_tail_flag(
    series: pd.Series,
    percentile: float = 0.05
) -> pd.Series:
    """
    Flag values in the lower statistical tail.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    threshold = values.quantile(
        percentile
    )

    return values <= threshold