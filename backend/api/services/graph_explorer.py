"""Build Cytoscape-compatible graph payloads from pipeline outputs."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from backend.api.schemas import Edge, Node
from backend.api.services.data_loader import DataLoader
from backend.api.services.gnn_influence import GNNInfluenceCalculator


class GraphExplorer:
    def __init__(self, loader: DataLoader, influence_calc: GNNInfluenceCalculator):
        self.loader = loader
        self.influence_calc = influence_calc
        self.graph: dict[str, set[str]] = defaultdict(set)
        self.reverse_graph: dict[str, set[str]] = defaultdict(set)
        self.edge_data: dict[tuple[str, str], dict] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        edges = self.loader.graph_edges
        for _, row in edges.iterrows():
            src = str(row["source_wallet_id"])
            tgt = str(row["target_wallet_id"])
            self.graph[src].add(tgt)
            self.reverse_graph[tgt].add(src)
            self.edge_data[(src, tgt)] = {
                "tx_count": int(row.get("transaction_count", 1)),
                "amount": float(row.get("total_amount_sats", 0.0)),
            }

    def _node_features(self, nid: str, preds: pd.DataFrame, graph_feats: pd.DataFrame, wallet_feats: pd.DataFrame) -> Node:
        if nid in preds.index:
            row = preds.loc[nid]
            risk_score = float(row["risk_probability"]) if "risk_probability" in preds.columns else 0.5
            # Use the model's binary classification label as ground truth for is_anomaly
            if "risk_prediction" in preds.columns:
                is_anomaly = bool(row["risk_prediction"] == 1)
            else:
                # Fallback only if column missing: use 0.7 threshold
                is_anomaly = risk_score >= 0.7
        else:
            risk_score = 0.5
            is_anomaly = False

        feat_dict: dict = {}
        if nid in graph_feats.index:
            g_row = graph_feats.loc[nid]
            feat_dict.update(
                {
                    "degree": int(g_row.get("degree", 0)),
                    "in_degree": int(g_row.get("in_degree", 0)),
                    "out_degree": int(g_row.get("out_degree", 0)),
                    "pagerank": float(g_row.get("pagerank", 0.0)),
                    "betweenness_centrality": float(g_row.get("betweenness_centrality", 0.0)),
                    "clustering_coefficient": float(g_row.get("clustering_coefficient", 0.0)),
                    "community_id": str(g_row.get("community_id", "0")),
                }
            )

        if nid in wallet_feats.index:
            w_row = wallet_feats.loc[nid]
            feat_dict.update(
                {
                    "tx_count": int(w_row.get("tx_count", 0)),
                    "total_sent_sats": float(w_row.get("total_sent_sats", 0.0)),
                    "total_received_sats": float(w_row.get("total_received_sats", 0.0)),
                    "fan_out": float(w_row.get("fan_out", 0.0)),
                    "fan_in": float(w_row.get("fan_in", 0.0)),
                }
            )

        return Node(
            id=nid,
            risk_score=round(risk_score, 4),
            is_anomaly=is_anomaly,
            degree=feat_dict.get("degree", 0),
            pagerank=feat_dict.get("pagerank", 0.0),
            community_id=feat_dict.get("community_id", "0"),
            wallet_features=feat_dict,
        )

    def get_ego_graph(self, wallet_id: str, hops: int = 2) -> dict:
        nodes = {wallet_id}
        current_level = {wallet_id}

        for _ in range(hops):
            next_level: set[str] = set()
            for node in current_level:
                next_level.update(self.graph[node])
                next_level.update(self.reverse_graph[node])
            current_level = next_level - nodes
            nodes.update(current_level)

        preds = self.loader.risk_predictions
        graph_feats = self.loader.graph_features
        wallet_feats = self.loader.wallet_features

        node_list = [self._node_features(nid, preds, graph_feats, wallet_feats) for nid in nodes]
        edge_list = []

        for src, tgts in self.graph.items():
            if src not in nodes:
                continue
            for tgt in tgts:
                if tgt not in nodes:
                    continue
                edata = self.edge_data.get((src, tgt), {})
                tx_count = edata.get("tx_count", 1)
                edge_list.append(
                    Edge(
                        source=src,
                        target=tgt,
                        transaction_count=tx_count,
                        total_amount_sats=edata.get("amount", 0.0),
                        gnn_influence_weight=self.influence_calc.calculate_edge_influence(src, tgt),
                        frequency_score=round(min(1.0, np.log1p(tx_count) / 5), 4),
                    )
                )

        return {"wallet_id": wallet_id, "nodes": node_list, "edges": edge_list, "ego_hops": hops}

    def get_overview_graph(self, max_nodes: int = 500) -> dict:
        preds = self.loader.risk_predictions
        edges_df = self.loader.graph_edges
        graph_feats = self.loader.graph_features
        wallet_feats = self.loader.wallet_features

        selected = set(str(w) for w in preds.sort_values("risk_probability", ascending=False).index[:max_nodes])
        src_col = edges_df["source_wallet_id"].astype(str)
        tgt_col = edges_df["target_wallet_id"].astype(str)
        filtered_edges = edges_df[src_col.isin(selected) & tgt_col.isin(selected)]

        connected = set(filtered_edges["source_wallet_id"].astype(str)).union(
            set(filtered_edges["target_wallet_id"].astype(str))
        )
        all_nodes = connected.union(selected)

        nodes = []
        for nid in all_nodes:
            node = self._node_features(nid, preds, graph_feats, wallet_feats)
            vis_out = int((filtered_edges["source_wallet_id"].astype(str) == nid).sum())
            vis_in = int((filtered_edges["target_wallet_id"].astype(str) == nid).sum())
            node.wallet_features["visible_degree"] = vis_in + vis_out
            node.wallet_features["visible_in_degree"] = vis_in
            node.wallet_features["visible_out_degree"] = vis_out
            nodes.append(node)

        edges = []
        for _, row in filtered_edges.iterrows():
            src, tgt = str(row["source_wallet_id"]), str(row["target_wallet_id"])
            tx_c = int(row.get("transaction_count", 1))
            edges.append(
                Edge(
                    source=src,
                    target=tgt,
                    transaction_count=tx_c,
                    total_amount_sats=float(row.get("total_amount_sats", 0.0)),
                    gnn_influence_weight=self.influence_calc.calculate_edge_influence(src, tgt),
                    frequency_score=round(min(1.0, np.log1p(tx_c) / 5), 4),
                )
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }
