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
            # Forward graph: src -> {tgt: [txs]}
            self.graph: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
            # Reverse graph: tgt -> {src: [txs]} — needed for fan_in detection
            self.reverse_graph: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

            for _, row in edges.iterrows():
                src = str(row["source_wallet_id"])
                tgt = str(row["target_wallet_id"])
                tx_data = {
                    "txid": str(row.get("first_txid", f"edge_{src}_{tgt}")),
                    "amount": float(row.get("total_amount_sats", 0.0)),
                    "timestamp": str(row.get("first_seen", "")),
                    "count": int(row.get("transaction_count", 1)),
                }
                self.graph[src][tgt].append(tx_data)
                self.reverse_graph[tgt][src].append(tx_data)

        except Exception as exc:
            logger.warning("PatternDetector graph build failed: %s", exc)
            self.graph = defaultdict(lambda: defaultdict(list))
            self.reverse_graph = defaultdict(lambda: defaultdict(list))

    def _get_graph_degrees(self, wallet_id: str) -> Tuple[int, int]:
        """Return (in_degree, out_degree) from graph_features CSV if available."""
        try:
            gf = self.loader.graph_features
            if wallet_id in gf.index:
                row = gf.loc[wallet_id]
                return int(row.get("in_degree", 0)), int(row.get("out_degree", 0))
        except Exception:
            pass
        # Fallback: count from adjacency lists directly
        out_deg = len(self.graph.get(wallet_id, {}))
        in_deg = len(self.reverse_graph.get(wallet_id, {}))
        return in_deg, out_deg

    def _detect_cycle(self, wallet_id: str) -> bool:
        """
        Strict cycle detection: ONLY flags as cycle if there is a direct 2-hop back-edge
        (A → B → A). This avoids marking every node in a dense graph as 'cycle'
        when longer paths happen to loop back.
        """
        for neighbor in self.graph.get(wallet_id, {}):
            if wallet_id in self.graph.get(neighbor, {}):
                return True  # Direct A→B→A back-edge found
        return False

    def _find_cycle_path(self, wallet_id: str) -> list:
        """
        Returns the edge IDs forming the FIRST direct 2-hop cycle found.
        Returns empty list if none exists.
        """
        fwd_graph = self.graph
        for neighbor, txs_ab in fwd_graph.get(wallet_id, {}).items():
            if wallet_id in fwd_graph.get(neighbor, {}):
                # Build edge path: wallet_id→neighbor→wallet_id
                txid_ab = txs_ab[0]["txid"] if txs_ab else f"{wallet_id}_{neighbor}"
                txs_ba = fwd_graph[neighbor][wallet_id]
                txid_ba = txs_ba[0]["txid"] if txs_ba else f"{neighbor}_{wallet_id}"
                return [
                    {"src": wallet_id, "tgt": neighbor, "txid": txid_ab},
                    {"src": neighbor, "tgt": wallet_id, "txid": txid_ba},
                ]
        return []


    def detect_pattern(self, wallet_id: str, max_hops: int = 5) -> Tuple[str, float, List[Dict]]:
        """
        Detect the dominant fund-flow pattern using real in/out degree from graph_features.

        Priority order:
          1. fan_out     — out dominates strongly (out >= 5 and out > in*2)
          2. fan_in      — in dominates strongly (in >= 5 and in > out*2)
          3. cycle       — balanced degrees with a direct 2-hop back-edge
          4. peeling_chain — moderate traffic both sides
          5. direct_transfer — simple low-degree flow
        """
        in_deg, out_deg = self._get_graph_degrees(wallet_id)
        total = in_deg + out_deg

        # 1. Fan-out (strong): out clearly dominates (>2x)
        if out_deg >= 4 and out_deg > in_deg * 2.0:
            sequence = self._trace_forward(wallet_id, max_hops)
            return "fan_out", min(0.97, 0.65 + out_deg / 40), sequence

        # 2. Fan-in (strong): in clearly dominates (>2x)
        if in_deg >= 4 and in_deg > out_deg * 2.0:
            sequence = self._trace_backward(wallet_id, max_hops)
            return "fan_in", min(0.95, 0.60 + in_deg / 40), sequence

        # 3. Fan-out (moderate): out noticeably higher than in (>1.3x)
        if out_deg >= 3 and out_deg > in_deg * 1.3:
            sequence = self._trace_forward(wallet_id, max_hops)
            return "fan_out", min(0.88, 0.58 + out_deg / 40), sequence

        # 4. Fan-in (moderate): in noticeably higher than out (>1.3x)
        if in_deg >= 3 and in_deg > out_deg * 1.3:
            sequence = self._trace_backward(wallet_id, max_hops)
            return "fan_in", min(0.86, 0.55 + in_deg / 40), sequence

        # 5. Cycle: ONLY when degrees are nearly balanced (within 20% of each other)
        #    AND there is a confirmed direct A→B→A back-edge. Rare & meaningful.
        max_deg = max(in_deg, out_deg) if max(in_deg, out_deg) > 0 else 1
        is_balanced = abs(in_deg - out_deg) / max_deg <= 0.20
        if is_balanced and total >= 4 and self._detect_cycle(wallet_id):
            sequence = self._trace_forward(wallet_id, max_hops)
            return "cycle", min(0.94, 0.70 + total / 120), sequence

        # 6. Peeling chain / layering: moderate traffic both sides
        if in_deg >= 2 and out_deg >= 2:
            sequence = self._trace_forward(wallet_id, max_hops)
            return "peeling_chain", min(0.82, 0.52 + (in_deg + out_deg) / 80), sequence

        # 7. Fallback: simple direct transfer
        sequence = self._trace_forward(wallet_id, max_hops)
        return "direct_transfer", min(0.75, 0.40 + total / 20), sequence


    def _trace_forward(self, wallet_id: str, max_hops: int) -> List[Dict]:
        """BFS outward from wallet_id collecting ordered transaction steps."""
        sequence: List[Dict] = []
        visited: set[str] = {wallet_id}
        queue = deque([(wallet_id, 0)])

        while queue and len(sequence) < 15:
            current, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for tgt, txs in self.graph[current].items():
                if tgt not in visited:
                    visited.add(tgt)
                    queue.append((tgt, hop + 1))
                for tx in txs:
                    sequence.append({
                        "step": len(sequence) + 1,
                        "source_wallet": current,
                        "source": current,
                        "target_wallet": tgt,
                        "target": tgt,
                        "txid": tx["txid"],
                        "amount_sats": tx["amount"],
                        "timestamp": tx["timestamp"],
                    })
                    if len(sequence) >= 15:
                        break

        return sequence

    def _trace_backward(self, wallet_id: str, max_hops: int) -> List[Dict]:
        """BFS inward to wallet_id (reverse edges) collecting ordered transaction steps."""
        sequence: List[Dict] = []
        visited: set[str] = {wallet_id}
        queue = deque([(wallet_id, 0)])

        while queue and len(sequence) < 15:
            current, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for src, txs in self.reverse_graph[current].items():
                if src not in visited:
                    visited.add(src)
                    queue.append((src, hop + 1))
                for tx in txs:
                    sequence.append({
                        "step": len(sequence) + 1,
                        "source_wallet": src,
                        "source": src,
                        "target_wallet": current,
                        "target": current,
                        "txid": tx["txid"],
                        "amount_sats": tx["amount"],
                        "timestamp": tx["timestamp"],
                    })
                    if len(sequence) >= 15:
                        break

        return sequence
