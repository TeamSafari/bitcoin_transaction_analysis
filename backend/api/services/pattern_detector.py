"""Fund-flow pattern detection for trace animations."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple

import logging

from backend.api.services.data_loader import DataLoader

logger = logging.getLogger(__name__)


class PatternDetector:
    def __init__(self, loader: DataLoader, raw_dir=None):
        self.loader = loader
        self.raw_dir = raw_dir
        self._build_graph()

    def _build_graph(self) -> None:
        try:
            edges = self.loader.graph_edges
            self.graph: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for _, row in edges.iterrows():
                src = str(row["source_wallet_id"])
                tgt = str(row["target_wallet_id"])
                self.graph[src][tgt].append(
                    {
                        "txid": str(row.get("first_txid", f"edge_{src}_{tgt}")),
                        "amount": float(row.get("total_amount_sats", 0.0)),
                        "timestamp": str(row.get("first_seen", "")),
                        "count": int(row.get("transaction_count", 1)),
                    }
                )
        except Exception as exc:
            logger.warning("PatternDetector graph build failed: %s", exc)
            self.graph = defaultdict(lambda: defaultdict(list))

    def detect_pattern(self, wallet_id: str, max_hops: int = 5) -> Tuple[str, float, List[Dict]]:
        sequence = self._trace_funds_bfs(wallet_id, max_hops=max_hops)
        if not sequence:
            return "direct_transfer", 0.5, []
        pattern, confidence = self._analyze_pattern(wallet_id, sequence)
        return pattern, confidence, sequence

    def _trace_funds_bfs(self, wallet_id: str, max_hops: int = 5) -> List[Dict]:
        sequence: List[Dict] = []
        visited: set[str] = set()
        queue = deque([(wallet_id, 0)])

        while queue:
            current_wallet, hop_count = queue.popleft()
            if hop_count >= max_hops or current_wallet in visited:
                continue
            visited.add(current_wallet)

            for target_wallet, txs in self.graph[current_wallet].items():
                for tx in txs:
                    sequence.append(
                        {
                            "step": len(sequence) + 1,
                            "source_wallet": current_wallet,
                            "source": current_wallet,
                            "target_wallet": target_wallet,
                            "target": target_wallet,
                            "txid": tx["txid"],
                            "amount_sats": tx["amount"],
                            "timestamp": tx["timestamp"],
                        }
                    )
                    queue.append((target_wallet, hop_count + 1))

        return sorted(sequence, key=lambda x: x["amount_sats"], reverse=True)[:10]

    def _analyze_pattern(self, wallet_id: str, sequence: List[Dict]) -> Tuple[str, float]:
        if not sequence:
            return "direct_transfer", 0.5

        out_degree = len({tx["target_wallet"] for tx in sequence})
        in_degree = len({tx["source_wallet"] for tx in sequence if tx["source_wallet"] != wallet_id})

        if out_degree > 5 and len(sequence) > 1:
            return "fan_out", min(0.95, 0.6 + out_degree / 50)
        if in_degree > 5:
            return "fan_in", min(0.90, 0.5 + in_degree / 50)
        if len(sequence) >= 3 and out_degree <= 3:
            amounts = [tx["amount_sats"] for tx in sequence[:5]]
            if all(amounts[i] >= amounts[i + 1] * 0.9 for i in range(len(amounts) - 1)):
                return "peeling_chain", min(0.92, 0.65 + len(sequence) / 20)

        return "direct_transfer", min(0.85, 0.5 + len(sequence) / 20)
