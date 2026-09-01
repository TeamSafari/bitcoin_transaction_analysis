import numpy as np
import pandas as pd


def entropy(series: pd.Series) -> float:

    counts = series.value_counts(
        normalize=True
    )

    return float(
        -(counts * np.log2(counts)).sum()
    )


def build_network_features(
    network_observations: pd.DataFrame,
    ip_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build wallet-level network and GeoIP features.
    """

    network = (
        network_observations.copy()
    )

    metadata = (
        ip_metadata.copy()
    )

    network["wallet_context"] = (
        network["wallet_context"]
        .astype(str)
    )

    network["src_ip"] = (
        network["src_ip"]
        .astype(str)
    )

    network["dst_ip"] = (
        network["dst_ip"]
        .astype(str)
    )

    # ---------------------------------------------------------
    # Basic network statistics
    # ---------------------------------------------------------

    grouped = (
        network.groupby("wallet_context")
        .agg(
            observation_count=(
                "observation_id",
                "nunique"
            ),
            unique_src_ips=(
                "src_ip",
                "nunique"
            ),
            unique_dst_ips=(
                "dst_ip",
                "nunique"
            ),
            unique_src_ports=(
                "src_port",
                "nunique"
            ),
            unique_dst_ports=(
                "dst_port",
                "nunique"
            ),
            mean_latency_ms=(
                "latency_ms",
                "mean"
            ),
            median_latency_ms=(
                "latency_ms",
                "median"
            ),
            std_latency_ms=(
                "latency_ms",
                "std"
            ),
            total_bytes_sent=(
                "bytes_sent",
                "sum"
            ),
            total_bytes_received=(
                "bytes_received",
                "sum"
            ),
        )
    )

    grouped["inbound_ratio"] = (
        grouped["total_bytes_received"]
        /
        (
            grouped["total_bytes_sent"]
            +
            grouped["total_bytes_received"]
        ).replace(0, 1)
    )

    grouped["outbound_ratio"] = (
        grouped["total_bytes_sent"]
        /
        (
            grouped["total_bytes_sent"]
            +
            grouped["total_bytes_received"]
        ).replace(0, 1)
    )

    # ---------------------------------------------------------
    # GeoIP enrichment
    # ---------------------------------------------------------

    src_geo = metadata[
        [
            "ip",
            "country_code",
            "asn",
            "network_type",
        ]
    ].copy()

    src_geo = src_geo.rename(
        columns={
            "ip": "src_ip",
            "country_code": "src_country_code",
            "asn": "src_asn",
            "network_type":
                "src_network_type",
        }
    )

    dst_geo = metadata[
        [
            "ip",
            "country_code",
            "asn",
            "network_type",
        ]
    ].copy()

    dst_geo = dst_geo.rename(
        columns={
            "ip": "dst_ip",
            "country_code": "dst_country_code",
            "asn": "dst_asn",
            "network_type":
                "dst_network_type",
        }
    )

    enriched = (
        network
        .merge(
            src_geo,
            on="src_ip",
            how="left",
        )
        .merge(
            dst_geo,
            on="dst_ip",
            how="left",
        )
    )

    # ---------------------------------------------------------
    # Wallet-level GeoIP metrics
    # ---------------------------------------------------------

    geo_rows = []

    for wallet, group in (
        enriched.groupby(
            "wallet_context"
        )
    ):

        countries = pd.concat(
            [
                group[
                    "src_country_code"
                ],
                group[
                    "dst_country_code"
                ],
            ]
        ).dropna()

        asns = pd.concat(
            [
                group["src_asn"],
                group["dst_asn"],
            ]
        ).dropna()

        network_types = pd.concat(
            [
                group[
                    "src_network_type"
                ],
                group[
                    "dst_network_type"
                ],
            ]
        ).dropna()

        geo_rows.append(
            {
                "wallet_id": wallet,

                "unique_countries":
                    countries.nunique(),

                "unique_asns":
                    asns.nunique(),

                "unique_network_types":
                    network_types.nunique(),

                "country_entropy":
                    entropy(countries)
                    if len(countries)
                    else 0.0,

                "asn_entropy":
                    entropy(asns)
                    if len(asns)
                    else 0.0,
            }
        )

    geo_features = pd.DataFrame(
        geo_rows
    )

    if not geo_features.empty:

        geo_features = (
            geo_features
            .set_index("wallet_id")
        )

    # ---------------------------------------------------------
    # Shared IP / ASN features
    # ---------------------------------------------------------

    ip_wallets = pd.concat(
        [
            network[
                [
                    "wallet_context",
                    "src_ip",
                ]
            ].rename(
                columns={
                    "src_ip": "ip"
                }
            ),

            network[
                [
                    "wallet_context",
                    "dst_ip",
                ]
            ].rename(
                columns={
                    "dst_ip": "ip"
                }
            ),
        ]
    )

    ip_counts = (
        ip_wallets
        .groupby("ip")[
            "wallet_context"
        ]
        .nunique()
    )

    shared_ip = (
        ip_wallets
        .assign(
            shared=lambda x:
                x["ip"].map(ip_counts)
        )
        .groupby(
            "wallet_context"
        )["shared"]
        .max()
        .rename(
            "shared_ip_wallet_count"
        )
    )

    # ASN → wallet count
    asn_rows = enriched[
        [
            "wallet_context",
            "src_asn",
            "dst_asn",
        ]
    ]

    asn_long = pd.concat(
        [
            asn_rows[
                [
                    "wallet_context",
                    "src_asn",
                ]
            ].rename(
                columns={
                    "src_asn": "asn"
                }
            ),

            asn_rows[
                [
                    "wallet_context",
                    "dst_asn",
                ]
            ].rename(
                columns={
                    "dst_asn": "asn"
                }
            ),
        ]
    ).dropna()

    asn_counts = (
        asn_long
        .groupby("asn")[
            "wallet_context"
        ]
        .nunique()
    )

    shared_asn = (
        asn_long
        .assign(
            shared=lambda x:
                x["asn"].map(asn_counts)
        )
        .groupby(
            "wallet_context"
        )["shared"]
        .max()
        .rename(
            "shared_asn_wallet_count"
        )
    )

    # ---------------------------------------------------------
    # Final merge
    # ---------------------------------------------------------

    result = grouped.copy()

    if not geo_features.empty:

        result = result.join(
            geo_features,
            how="left",
        )

    result = result.join(
        shared_ip,
        how="left",
    )

    result = result.join(
        shared_asn,
        how="left",
    )

    result = result.reset_index()

    result = result.rename(
        columns={
            "wallet_context":
                "wallet_id"
        }
    )

    numeric_columns = [
        col
        for col in result.columns
        if col != "wallet_id"
    ]

    result[numeric_columns] = (
        result[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            0
        )
        .fillna(0)
    )

    return result