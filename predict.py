import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from explain_risk import explain_customer_risk
from feature_engineering import (
    add_engineered_features,
    clean_dataframe_columns,
    prepare_model_features,
)
from recommend_actions import (
    priority_from_score,
    recommended_retention_action,
    retention_priority_score,
    risk_level_from_probability,
)


DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "iranian_churn_model.joblib"
DEFAULT_METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"


def load_json_payload(args):
    if args.json:
        raw_payload = args.json
    elif args.input:
        raw_payload = Path(args.input).read_text(encoding="utf-8")
    else:
        raw_payload = sys.stdin.read()

    if not raw_payload.strip():
        raise ValueError("No JSON payload received.")

    payload = json.loads(raw_payload)
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    return payload


def payload_to_records(payload):
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError("Payload must be a JSON object or a list of JSON objects.")


def payload_to_frame(payload):
    records = payload_to_records(payload)
    data = pd.DataFrame(records)
    data = clean_dataframe_columns(data)
    if "customer_id" not in data.columns:
        customer_ids = [f"CUST_{index:03d}" for index in range(1, len(data) + 1)]
        data.insert(0, "customer_id", customer_ids)
    return data


def to_python_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def prediction_record(row_number, row, probability, prediction, threshold, metadata):
    risk_level = risk_level_from_probability(float(probability))
    business_row = row.copy()
    business_row["churn_probability"] = float(probability)
    business_row["churn_prediction"] = int(prediction)
    business_row["risk_level"] = risk_level

    score = retention_priority_score(business_row, metadata)
    business_row["retention_priority_score"] = score
    business_row["priority"] = priority_from_score(score)
    business_row["recommended_action"] = recommended_retention_action(business_row, metadata)

    customer_id = row.get("customer_id", f"CUST_{row_number:03d}")
    result = {
        "row": row_number,
        "customer_id": str(customer_id),
        "churn_probability": to_python_value(round(float(probability), 6)),
        "churn_prediction": to_python_value(int(prediction)),
        "threshold": threshold,
        "risk_level": risk_level,
        "customer_value_tier": row.get("customer_value_tier", "Medium"),
        "retention_priority_score": score,
        "main_reason": explain_customer_risk(row, metadata.get("risk_driver_thresholds", {})),
        "recommended_action": business_row["recommended_action"],
        "priority": business_row["priority"],
    }

    optional_fields = [
        "email",
        "name",
        "nama",
        "phone",
        "segment",
        "segment_description",
        "suggested_strategy",
    ]
    for optional_field in optional_fields:
        if optional_field in row and pd.notna(row[optional_field]):
            result[optional_field] = to_python_value(row[optional_field])
    return result


def predict(payload, model_path=DEFAULT_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH):
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata.get("threshold", metadata.get("selected_threshold", 0.5)))

    data = payload_to_frame(payload)
    features = prepare_model_features(data, metadata)
    business_data, _ = add_engineered_features(
        data,
        value_thresholds=metadata.get("customer_value_quantile_thresholds"),
    )

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = []
    for row_number, (_, row) in enumerate(business_data.iterrows(), start=1):
        results.append(
            prediction_record(
                row_number=row_number,
                row=row,
                probability=probabilities[row_number - 1],
                prediction=predictions[row_number - 1],
                threshold=threshold,
                metadata=metadata,
            )
        )

    return {
        "model": metadata.get("best_model", "unknown"),
        "predictions": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict customer churn risk from JSON.")
    parser.add_argument("--json", help="JSON object or list of objects.")
    parser.add_argument("--input", help="Path to a JSON input file.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        payload = load_json_payload(args)
        result = predict(payload, args.model, args.metadata)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
