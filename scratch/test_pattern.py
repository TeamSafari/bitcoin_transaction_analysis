import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, r'c:\Kamlesh\bitcoin_transaction_analysis')

from backend.api_server import DataLoader, PatternDetector, ExplainabilityEngine, PROJECT_ROOT

try:
    loader = DataLoader(PROJECT_ROOT / "outputs")
    pattern_engine = PatternDetector(loader)
    
    print(f"Graph in pattern engine has {len(pattern_engine.graph)} nodes.")
    print(f"Edges out from W0000182: {len(pattern_engine.graph['W0000182'])}")
    
    pattern, conf, seq = pattern_engine.detect_pattern('W0000182', max_hops=5)
    print(f"Pattern for W0000182: {pattern}, Conf: {conf}, Seq len: {len(seq)}")
    for x in seq:
        print(f"  {x['source_wallet']} -> {x['target_wallet']} (amt: {x['amount_sats']})")

    print("\nTesting SHAP:")
    explain_engine = ExplainabilityEngine(loader)
    shap_res = explain_engine.explain_wallet_risk('W0000182')
    print("SHAP result:", shap_res)

except Exception as e:
    import traceback
    traceback.print_exc()
