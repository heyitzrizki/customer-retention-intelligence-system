import numpy as np
import pandas as pd


def risk_level_from_probability(churn_probability: float) -> str:
    if churn_probability >= 0.65:
        return "High"
    if churn_probability >= 0.35:
        return "Medium"
    return "Low"


def priority_from_score(score: float) -> str:
    if score >= 80:
        return "P1"
    if score >= 60:
        return "P2"
    if score >= 40:
        return "P3"
    return "P4"


def _value_score(
    customer_value: float,
    thresholds: dict | None = None,
    tier: str | None = None,
) -> float:
    thresholds = thresholds or {}
    min_value = thresholds.get("min")
    max_value = thresholds.get("max")
    if min_value is not None and max_value is not None and float(max_value) > float(min_value):
        scaled_value = (customer_value - float(min_value)) / (
            float(max_value) - float(min_value)
        )
        return float(np.clip(scaled_value * 100, 0, 100))

    tier_scores = {"Low": 30, "Medium": 65, "High": 100}
    return float(tier_scores.get(str(tier), 50))


def service_issue_score(row: pd.Series, failed_call_rate_high: float = 0.20) -> float:
    score = 0
    if float(row.get("complains", 0)) >= 1:
        score += 60
    if float(row.get("failed_call_rate", 0)) >= failed_call_rate_high:
        score += 40
    return float(np.clip(score, 0, 100))


def retention_priority_score(row: pd.Series, metadata: dict | None = None) -> int:
    metadata = metadata or {}
    value_thresholds = metadata.get("customer_value_quantile_thresholds", {})
    risk_thresholds = metadata.get("risk_driver_thresholds", {})

    churn_risk_score = float(row.get("churn_probability", 0)) * 100
    customer_value_score = _value_score(
        float(row.get("customer_value", 0)),
        thresholds=value_thresholds,
        tier=row.get("customer_value_tier"),
    )
    issue_score = service_issue_score(
        row,
        failed_call_rate_high=float(risk_thresholds.get("failed_call_rate_high", 0.20)),
    )

    score = 0.50 * churn_risk_score + 0.30 * customer_value_score + 0.20 * issue_score
    return int(round(float(np.clip(score, 0, 100))))


def recommended_retention_action(row: pd.Series, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    risk_thresholds = metadata.get("risk_driver_thresholds", {})
    failed_call_rate_high = float(risk_thresholds.get("failed_call_rate_high", 0.20))
    frequency_of_use_low = float(risk_thresholds.get("frequency_of_use_low", 20))

    risk_level = row.get("risk_level", "Low")
    value_tier = row.get("customer_value_tier", "Medium")
    has_service_issue = (
        float(row.get("complains", 0)) >= 1
        or float(row.get("failed_call_rate", 0)) >= failed_call_rate_high
    )

    if risk_level == "High" and value_tier == "High":
        if has_service_issue:
            return "Service recovery call"
        return "Priority retention call"

    if risk_level == "High" and value_tier == "Medium":
        if float(row.get("frequency_of_use", 0)) <= frequency_of_use_low:
            return "Re-engagement offer"
        return "Personalized retention offer"

    if risk_level == "High" and value_tier == "Low":
        return "Automated SMS/email campaign"

    if risk_level == "Medium" and value_tier == "High":
        return "Monitor and send loyalty message"

    if risk_level == "Medium":
        return "Low-cost retention campaign"

    return "No immediate action"


def add_retention_recommendations(data: pd.DataFrame, metadata: dict | None = None) -> pd.DataFrame:
    data = data.copy()
    data["retention_priority_score"] = data.apply(
        lambda row: retention_priority_score(row, metadata),
        axis=1,
    )
    data["priority"] = data["retention_priority_score"].apply(priority_from_score)
    data["recommended_action"] = data.apply(
        lambda row: recommended_retention_action(row, metadata),
        axis=1,
    )
    return data
