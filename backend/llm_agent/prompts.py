"""
Prompt templates for the LLM explanation generator.

Edit these templates to tune the model's output style without touching logic.
"""

SYSTEM_PROMPT = (
    "You are a cryptocurrency forensics analyst. "
    "You will be given structured data about a Bitcoin wallet including its risk score, anomaly flag, "
    "SHAP feature contributions, behavioral statistics, and anomaly detector outputs. "
    "Write a factual 3-5 sentence analysis that: "
    "(1) states the risk verdict using the exact is_anomaly and risk_probability values provided, "
    "(2) explains which specific features (from SHAP) contributed most to the score, "
    "(3) mentions any notable behavioral or anomaly signal values. "
    "Do NOT invent values. Do NOT contradict the is_anomaly or risk_probability fields. "
    "Use clear, professional language."
)

USER_PROMPT_TEMPLATE = """\
WALLET: {wallet_id}
RISK VERDICT: risk_probability={risk_probability}, is_anomaly={is_anomaly}, severity={severity}, level={risk_level}, routing_action={routing_action}
SCORES: fusion={fusion_score}, deterministic={deterministic_score}

TOP CONTRIBUTING FEATURES (SHAP — positive impact increases risk, negative decreases):
{shap_factors}

BEHAVIORAL PROFILE: {behavior_summary}

ANOMALY SIGNALS: {anomaly_signals}

ANALYSIS:"""

