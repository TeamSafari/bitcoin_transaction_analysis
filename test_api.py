#!/usr/bin/env python3
"""
API Test Script - Test all three endpoints

Usage:
    python test_api.py
    python test_api.py W0000001        # Test specific wallet
    python test_api.py W0000001 W0000002 W0000003  # Test multiple wallets
"""

import requests
import json
import sys
from typing import List

API_BASE = "http://localhost:8000/api/v1"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def test_health():
    """Test health endpoint"""
    print(f"\n{Colors.HEADER}{'='*70}")
    print("HEALTH CHECK")
    print(f"{'='*70}{Colors.END}\n")
    
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        data = resp.json()
        
        if resp.status_code == 200:
            print(f"{Colors.GREEN}✓ API is healthy{Colors.END}")
            print(f"  Status: {data['status']}")
            print(f"  Graph Edges: {data['data_sources']['graph_edges']:,}")
            print(f"  Wallets: {data['data_sources']['wallets']:,}")
            print(f"  Embeddings: {data['data_sources']['embeddings']:,}")
            return True
        else:
            print(f"{Colors.RED}✗ API returned status {resp.status_code}{Colors.END}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}✗ Cannot connect to API at {API_BASE}{Colors.END}")
        print(f"  Start the API server: bash run_api.sh")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")
        return False

def test_traceability(wallet_id: str):
    """Test traceability endpoint"""
    print(f"\n{Colors.HEADER}{'='*70}")
    print(f"TRACEABILITY TEST: {wallet_id}")
    print(f"{'='*70}{Colors.END}\n")
    
    try:
        resp = requests.get(f"{API_BASE}/patterns/trace/{wallet_id}?max_hops=5", timeout=10)
        
        if resp.status_code != 200:
            print(f"{Colors.RED}✗ Request failed: {resp.status_code}{Colors.END}")
            print(f"  {resp.json()}")
            return False
        
        data = resp.json()
        
        print(f"{Colors.CYAN}Pattern Detected:{Colors.END} {data['pattern_detected']}")
        print(f"{Colors.CYAN}Confidence Score:{Colors.END} {data['confidence_score']:.2%}")
        print(f"{Colors.CYAN}Total Hops:{Colors.END} {data['hop_count']}")
        print(f"{Colors.CYAN}Total Amount:{Colors.END} {data['total_amount_sats']:,.0f} sats")
        
        print(f"\n{Colors.BLUE}Transaction Sequence:{Colors.END}")
        for tx in data['sequence'][:5]:  # Show first 5
            print(f"  Step {tx['step']:2d}: {tx['source_wallet']} → {tx['target_wallet']}")
            print(f"           Amount: {tx['amount_sats']:15,.0f} sats")
            print(f"           TxID: {tx['txid']}")
        
        if len(data['sequence']) > 5:
            print(f"  ... and {len(data['sequence']) - 5} more transactions")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")
        return False

def test_explainability(wallet_id: str):
    """Test explainability endpoint"""
    print(f"\n{Colors.HEADER}{'='*70}")
    print(f"EXPLAINABILITY TEST: {wallet_id}")
    print(f"{'='*70}{Colors.END}\n")
    
    try:
        resp = requests.get(f"{API_BASE}/explainability/wallet/{wallet_id}", timeout=10)
        
        if resp.status_code != 200:
            print(f"{Colors.RED}✗ Request failed: {resp.status_code}{Colors.END}")
            print(f"  {resp.json()}")
            return False
        
        data = resp.json()
        
        # Risk level coloring
        risk_score = data['overall_risk_score']
        if risk_score < 0.3:
            risk_color = Colors.GREEN
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_color = Colors.YELLOW
            risk_level = "MEDIUM"
        elif risk_score < 0.8:
            risk_color = Colors.YELLOW
            risk_level = "HIGH"
        else:
            risk_color = Colors.RED
            risk_level = "CRITICAL"
        
        print(f"{Colors.CYAN}Overall Risk Score:{Colors.END} {risk_color}{risk_score:.3f} ({risk_level}){Colors.END}")
        print(f"{Colors.CYAN}Model Used:{Colors.END} {data['model_used']}")
        if data.get('deterministic_score'):
            print(f"{Colors.CYAN}Deterministic Score:{Colors.END} {data['deterministic_score']:.3f}")
        
        print(f"\n{Colors.BLUE}Top Risk Factors:{Colors.END}")
        for i, factor in enumerate(data['top_risk_factors'][:5], 1):
            print(f"  {i}. {Colors.BOLD}{factor['feature_name']}{Colors.END}")
            print(f"     Raw Value: {factor['raw_value']:.6f}")
            print(f"     SHAP Impact: {factor['shap_impact']:.3f} ({factor['percentile_rank']:.0f}th percentile)")
            print(f"     Description: {factor['description']}")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")
        return False

def test_graph(wallet_id: str, hops: int = 2):
    """Test graph endpoint"""
    print(f"\n{Colors.HEADER}{'='*70}")
    print(f"GRAPH TEST: {wallet_id} ({hops}-hop ego-graph)")
    print(f"{'='*70}{Colors.END}\n")
    
    try:
        resp = requests.get(f"{API_BASE}/graph/wallet/{wallet_id}?hops={hops}", timeout=10)
        
        if resp.status_code != 200:
            print(f"{Colors.RED}✗ Request failed: {resp.status_code}{Colors.END}")
            print(f"  {resp.json()}")
            return False
        
        data = resp.json()
        
        print(f"{Colors.CYAN}Nodes:{Colors.END} {len(data['nodes'])}")
        print(f"{Colors.CYAN}Edges:{Colors.END} {len(data['edges'])}")
        print(f"{Colors.CYAN}Ego Hops:{Colors.END} {data['ego_hops']}")
        
        print(f"\n{Colors.BLUE}Top Nodes (by risk):{Colors.END}")
        for node in sorted(data['nodes'], key=lambda n: n['risk_score'], reverse=True)[:3]:
            print(f"  {node['id']}: risk={node['risk_score']:.3f}")
        
        print(f"\n{Colors.BLUE}Edges with GNN Influence Weights:{Colors.END}")
        for edge in sorted(data['edges'], key=lambda e: e['gnn_influence_weight'], reverse=True)[:5]:
            influence_bar = "█" * int(edge['gnn_influence_weight'] * 20)
            print(f"  {edge['source']} → {edge['target']}")
            print(f"    Influence: {influence_bar} {edge['gnn_influence_weight']:.3f}")
            print(f"    Frequency: {edge['frequency_score']:.3f} ({edge['transaction_count']} tx)")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.END}")
        return False

def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}Bitcoin Transaction Analysis API - Test Suite{Colors.END}\n")
    
    # Get wallet IDs to test
    if len(sys.argv) > 1:
        wallet_ids = sys.argv[1:]
    else:
        wallet_ids = ["W0000001", "W0000002", "W0000004"]
    
    # Test health
    if not test_health():
        return 1
    
    # Test endpoints
    passed = 0
    failed = 0
    
    for wallet_id in wallet_ids:
        tests = [
            ("Traceability", lambda: test_traceability(wallet_id)),
            ("Explainability", lambda: test_explainability(wallet_id)),
            ("Graph", lambda: test_graph(wallet_id, hops=2)),
        ]
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"{Colors.RED}✗ {test_name} failed: {e}{Colors.END}")
                failed += 1
    
    # Summary
    print(f"\n{Colors.HEADER}{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}{Colors.END}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"{Colors.RED}Failed: {failed}{Colors.END}")
    print()
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
