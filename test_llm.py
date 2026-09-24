"""Quick test of the LLM explanation generator."""
from backend.api.services.data_loader import DataLoader
from backend.llm_agent.explanation_generator import ExplanationGenerator
from pathlib import Path

loader = DataLoader(Path("outputs/jobs/job-0c49abe5/results"))
gen = ExplanationGenerator(loader, job_id="job-0c49abe5")

print("Generating explanation for W0000182 (highest risk)...")
print()
result = gen.explain("W0000182")
print("=== RESULT ===")
print(f"Wallet: {result['wallet_id']}")
print(f"Cached: {result['cached']}")
print(f"Model: {result['model_used']}")
print()
print("Explanation:")
print(result["explanation"])
print()

# Test cache hit
print("--- Testing cache hit ---")
result2 = gen.explain("W0000182")
print(f"Cached: {result2['cached']}")
print()

# Test a low-risk wallet
print("--- Testing low-risk wallet W0000001 ---")
result3 = gen.explain("W0000001")
print(f"Wallet: {result3['wallet_id']}")
print(f"Cached: {result3['cached']}")
print()
print("Explanation:")
print(result3["explanation"])
