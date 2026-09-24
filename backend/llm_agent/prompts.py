"""
Prompt templates for the LLM explanation generator.

Edit these templates to tune the model's output style without touching logic.
"""

SYSTEM_PROMPT = (
    "You are a cryptocurrency forensics analyst. Analyze the following wallet data "
    "and explain why it was flagged as suspicious. Be specific about which behaviors "
    "are concerning and why. Reference the actual values provided. "
    "Write 3-5 sentences in clear, professional language."
)

USER_PROMPT_TEMPLATE = """\
WALLET: {wallet_id}
RISK: probability={risk_probability}, severity={severity}, level={risk_level}, action={routing_action}
SCORES: fusion={fusion_score}, deterministic={deterministic_score}

TOP RISK FACTORS (SHAP analysis):
{shap_factors}

BEHAVIOR: {behavior_summary}

ANOMALY SIGNALS: {anomaly_signals}

ANALYSIS:"""
