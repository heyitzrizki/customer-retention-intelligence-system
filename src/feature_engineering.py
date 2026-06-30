import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "age",
    "customer_value",
]

TARGET_COLUMN = "churn"

ENGINEERED_FEATURE_COLUMNS = [
    "usage_minutes",
    "failed_call_rate",
    "sms_share",
    "value_per_usage_minute",
    "value_per_frequency",
    "calls_per_distinct_number",
    "engagement_score",
    "usage_intensity",
]

MODEL_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS


def clean_column_name(column: object) -> str:
    column = str(column).strip().lower()
    return re.sub(r"[^0-9a-zA-Z]+", "_", column).strip("_")


def clean_column_names(columns: Iterable[object]) -> list[str]:
    return [clean_column_name(column) for column in columns]


def clean_dataframe_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = clean_column_names(data.columns)
    return data


def ensure_customer_id(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "customer_id" not in data.columns:
        customer_ids = [f"CUST_{index:04d}" for index in range(1, len(data) + 1)]
        data.insert(0, "customer_id", customer_ids)
    return data


def coerce_raw_features(
    data: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    data = clean_dataframe_columns(data)
    required_columns = required_columns or RAW_FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.copy()
    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    return data


def customer_value_thresholds(data: pd.DataFrame) -> Dict[str, float]:
    values = pd.to_numeric(data["customer_value"], errors="coerce")
    return {
        "low": float(values.quantile(0.33)),
        "high": float(values.quantile(0.67)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def apply_customer_value_tier(data: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    low = float(thresholds.get("low", 0))
    high = float(thresholds.get("high", low))

    def tier(value: float) -> str:
        if value >= high:
            return "High"
        if value >= low:
            return "Medium"
        return "Low"

    return pd.to_numeric(data["customer_value"], errors="coerce").fillna(0).apply(tier)


def _safe_minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_value) / (max_value - min_value)


def add_engineered_features(
    data: pd.DataFrame,
    value_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    data = coerce_raw_features(data)
    thresholds = value_thresholds or customer_value_thresholds(data)

    data["usage_minutes"] = data["seconds_of_use"] / 60
    data["failed_call_rate"] = data["call_failure"] / (data["frequency_of_use"] + 1)
    data["sms_share"] = data["frequency_of_sms"] / (
        data["frequency_of_sms"] + data["frequency_of_use"] + 1
    )
    data["value_per_usage_minute"] = data["customer_value"] / (data["usage_minutes"] + 1)
    data["value_per_frequency"] = data["customer_value"] / (data["frequency_of_use"] + 1)
    data["calls_per_distinct_number"] = data["frequency_of_use"] / (
        data["distinct_called_numbers"] + 1
    )

    usage_component = _safe_minmax(data["frequency_of_use"])
    duration_component = _safe_minmax(data["usage_minutes"])
    sms_component = _safe_minmax(data["frequency_of_sms"])
    subscription_component = _safe_minmax(data["subscription_length"])
    complaint_penalty = np.where(data["complains"] > 0, 0.25, 0)

    data["engagement_score"] = (
        (
            0.35 * usage_component
            + 0.30 * duration_component
            + 0.20 * sms_component
            + 0.15 * subscription_component
        )
        - complaint_penalty
    ).clip(0, 1) * 100
    data["usage_intensity"] = (
        0.60 * _safe_minmax(data["frequency_of_use"]) + 0.40 * _safe_minmax(data["usage_minutes"])
    ) * 100
    data["customer_value_tier"] = apply_customer_value_tier(data, thresholds)
    return data, thresholds


def prepare_model_features(data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    value_thresholds = metadata.get("customer_value_quantile_thresholds")
    data, _ = add_engineered_features(data, value_thresholds=value_thresholds)
    model_feature_columns = metadata.get("model_feature_columns", MODEL_FEATURE_COLUMNS)
    missing_columns = [column for column in model_feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing model feature columns after feature engineering: {missing_columns}"
        )
    return data[model_feature_columns]


def prepare_training_frame(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    data = clean_dataframe_columns(data)
    data = ensure_customer_id(data)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must include target column '{TARGET_COLUMN}'.")
    data, thresholds = add_engineered_features(data)
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)
    return data, thresholds
