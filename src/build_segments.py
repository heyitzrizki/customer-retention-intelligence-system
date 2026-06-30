from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from feature_engineering import RAW_FEATURE_COLUMNS, ensure_customer_id, prepare_training_frame


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "Customer Churn.csv"
FALLBACK_DATA_PATH = BASE_DIR / "Customer Churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SEGMENT_MODEL_PATH = ARTIFACTS_DIR / "segmentation_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
SEGMENTS_PATH = PROCESSED_DIR / "customer_segments.csv"

SEGMENT_FEATURES = [
    "customer_value",
    "subscription_length",
    "frequency_of_use",
    "frequency_of_sms",
    "failed_call_rate",
    "complains",
    "usage_minutes",
    "value_per_usage_minute",
]


def load_dataset() -> pd.DataFrame:
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    return pd.read_csv(path)


def describe_segment(row: pd.Series) -> tuple[str, str, str]:
    high_value = row["customer_value"] >= row["global_value_median"]
    high_usage = row["frequency_of_use"] >= row["global_usage_median"]
    high_complaints = row["complains"] >= row["global_complaint_mean"]
    high_failure = row["failed_call_rate"] >= row["global_failure_median"]
    high_churn = row["churn"] >= row["global_churn_mean"]

    if high_value and not high_churn:
        return (
            "High-Value Loyal Customers",
            "Valuable customers with comparatively healthy retention signals.",
            "Maintain loyalty benefits and proactive relationship management.",
        )
    if high_value and (high_churn or high_complaints or high_failure):
        return (
            "At-Risk Heavy Users",
            "Important customers showing service issues or elevated churn behavior.",
            "Prioritize service recovery and personalized retention outreach.",
        )
    if not high_usage:
        return (
            "Low Engagement Customers",
            "Customers with limited usage and weaker day-to-day engagement.",
            "Use re-engagement offers and education campaigns.",
        )
    if not high_value:
        return (
            "Price-Sensitive / Low-Value Users",
            "Lower-value customers who may respond better to low-cost campaigns.",
            "Use automated SMS/email offers and price-sensitive bundles.",
        )
    return (
        "Stable Low-Risk Customers",
        "Customers with steady behavior and no major warning signal.",
        "Monitor routinely and maintain standard lifecycle communications.",
    )


def build_segment_profiles(data: pd.DataFrame) -> dict:
    global_values = {
        "global_value_median": data["customer_value"].median(),
        "global_usage_median": data["frequency_of_use"].median(),
        "global_complaint_mean": data["complains"].mean(),
        "global_failure_median": data["failed_call_rate"].median(),
        "global_churn_mean": data["churn"].mean(),
    }

    profiles = (
        data.assign(**global_values)
        .groupby("segment_id")
        .agg(
            customer_value=("customer_value", "mean"),
            frequency_of_use=("frequency_of_use", "mean"),
            complains=("complains", "mean"),
            failed_call_rate=("failed_call_rate", "mean"),
            churn=("churn", "mean"),
            global_value_median=("global_value_median", "first"),
            global_usage_median=("global_usage_median", "first"),
            global_complaint_mean=("global_complaint_mean", "first"),
            global_failure_median=("global_failure_median", "first"),
            global_churn_mean=("global_churn_mean", "first"),
        )
        .reset_index()
    )

    label_map = {}
    used_names = set()
    for _, row in profiles.iterrows():
        name, description, strategy = describe_segment(row)
        if name in used_names:
            if row["customer_value"] >= row["global_value_median"]:
                name = "Stable Low-Risk Customers"
                description = "Customers with solid value and manageable risk signals."
                strategy = "Maintain regular engagement and monitor for risk changes."
            else:
                name = "Price-Sensitive / Low-Value Users"
                description = "Lower-value customers suited for efficient lifecycle campaigns."
                strategy = "Use low-cost automated offers and monitor response."
        used_names.add(name)
        label_map[int(row["segment_id"])] = {
            "segment": name,
            "segment_description": description,
            "suggested_strategy": strategy,
        }
    return label_map


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = ensure_customer_id(load_dataset())
    data, _ = prepare_training_frame(raw_data)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[SEGMENT_FEATURES])
    cluster_count = 5 if len(data) >= 5 else max(2, len(data))
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
    data["segment_id"] = model.fit_predict(scaled_features)

    label_map = build_segment_profiles(data)
    data["segment"] = data["segment_id"].map(lambda value: label_map[int(value)]["segment"])
    data["segment_description"] = data["segment_id"].map(
        lambda value: label_map[int(value)]["segment_description"]
    )
    data["suggested_strategy"] = data["segment_id"].map(
        lambda value: label_map[int(value)]["suggested_strategy"]
    )

    output_columns = [
        "customer_id",
        *RAW_FEATURE_COLUMNS,
        "churn",
        "usage_minutes",
        "failed_call_rate",
        "sms_share",
        "value_per_usage_minute",
        "value_per_frequency",
        "calls_per_distinct_number",
        "engagement_score",
        "usage_intensity",
        "customer_value_tier",
        "segment",
        "segment_description",
        "suggested_strategy",
    ]
    data[output_columns].to_csv(SEGMENTS_PATH, index=False)
    joblib.dump({"model": model, "features": SEGMENT_FEATURES, "label_map": label_map}, SEGMENT_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Saved customer segments to {SEGMENTS_PATH}")
    print(f"Saved segmentation model to {SEGMENT_MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
