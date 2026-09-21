"""
Identity Resolution Layer — link wallets to entities and network fingerprints.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def resolve_identities(datasets: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Resolve wallet ↔ entity links and build device/IP fingerprint profiles.

    Returns enriched datasets and an identity resolution table keyed by wallet_id.
    """
    data = datasets.copy()
    wallets = data["wallets"].copy()
    wallets["wallet_id"] = wallets["wallet_id"].astype(str)

    resolution_rows: list[dict[str, Any]] = []

    entities = data.get("entities")
    links = data.get("wallet_entity_links")
    net_obs = data["network_observations"].copy()
    ip_meta = data["ip_metadata"].copy()

    wallet_to_entity: dict[str, str] = {}
    entity_confidence: dict[str, float] = {}

    if links is not None and not links.empty:
        link_df = links.copy()
        link_df["wallet_id"] = link_df["wallet_id"].astype(str)
        link_df["entity_id"] = link_df["entity_id"].astype(str)
        if "confidence" in link_df.columns:
            link_df["confidence"] = pd.to_numeric(link_df["confidence"], errors="coerce").fillna(0.0)
        else:
            link_df["confidence"] = 1.0

        best_links = (
            link_df.sort_values("confidence", ascending=False)
            .drop_duplicates(subset=["wallet_id"], keep="first")
        )
        wallet_to_entity = dict(zip(best_links["wallet_id"], best_links["entity_id"]))
        entity_confidence = dict(zip(best_links["wallet_id"], best_links["confidence"]))

    elif "primary_entity_id" in wallets.columns:
        for _, row in wallets.iterrows():
            wid = str(row["wallet_id"])
            eid = row.get("primary_entity_id")
            if pd.notna(eid) and str(eid).strip():
                wallet_to_entity[wid] = str(eid)
                entity_confidence[wid] = 1.0

    net_obs["wallet_context"] = net_obs["wallet_context"].astype(str)
    ip_meta["ip"] = ip_meta["ip"].astype(str)

    wallet_ips: dict[str, set[str]] = {}
    wallet_asns: dict[str, set[str]] = {}

    for _, row in net_obs.iterrows():
        wid = str(row["wallet_context"])
        wallet_ips.setdefault(wid, set()).update(
            {str(row["src_ip"]), str(row["dst_ip"])}
        )

    ip_to_asn = dict(zip(ip_meta["ip"].astype(str), ip_meta.get("asn", pd.Series(dtype=float))))
    for wid, ips in wallet_ips.items():
        wallet_asns[wid] = {str(ip_to_asn.get(ip, "")) for ip in ips if ip in ip_to_asn}

    entity_fingerprints: dict[str, set[str]] = {}
    for wid, eid in wallet_to_entity.items():
        fingerprint = frozenset(wallet_ips.get(wid, set()))
        entity_fingerprints.setdefault(eid, set()).update(fingerprint)

    duplicate_entity_flags: dict[str, bool] = {}
    for eid, fingerprints in entity_fingerprints.items():
        duplicate_entity_flags[eid] = len(fingerprints) > 1

    for _, row in wallets.iterrows():
        wid = str(row["wallet_id"])
        eid = wallet_to_entity.get(wid)
        ips = wallet_ips.get(wid, set())
        asns = wallet_asns.get(wid, set())

        resolution_rows.append(
            {
                "wallet_id": wid,
                "resolved_entity_id": eid,
                "entity_link_confidence": entity_confidence.get(wid),
                "unique_ip_count": len(ips),
                "unique_asn_count": len({a for a in asns if a and a != "nan"}),
                "shared_ip_fingerprint": len(ips) > 0,
                "duplicate_entity_flag": duplicate_entity_flags.get(eid, False) if eid else False,
            }
        )

    resolution_df = pd.DataFrame(resolution_rows)

    wallets["resolved_entity_id"] = wallets["wallet_id"].map(wallet_to_entity)
    wallets["entity_link_confidence"] = wallets["wallet_id"].map(entity_confidence)
    wallets["unique_ip_count"] = wallets["wallet_id"].map(
        lambda w: len(wallet_ips.get(str(w), set()))
    )
    data["wallets"] = wallets
    data["identity_resolution"] = resolution_df

    return data, resolution_df
