import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "iranian_churn_model.joblib"
DEFAULT_METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"


def clean_column_names(columns):
    cleaned_columns = []
    for column in columns:
        column = str(column).strip().lower()
        column = re.sub(r"[^0-9a-zA-Z]+", "_", column).strip("_")
        cleaned_columns.append(column)
    return cleaned_columns


def add_engineered_features(data):
    data = data.copy()
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
    return data


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


def payload_to_frame(payload):
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Payload must be a JSON object or a list of JSON objects.")

    data = pd.DataFrame(records)
    data.columns = clean_column_names(data.columns)
    return data


def prepare_features(data, metadata):
    raw_feature_columns = metadata["raw_feature_columns"]
    model_feature_columns = metadata["model_feature_columns"]

    missing_columns = [column for column in raw_feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[raw_feature_columns].copy()
    for column in raw_feature_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data = add_engineered_features(data)
    return data[model_feature_columns]


def to_python_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def predict(payload, model_path, metadata_path):
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])

    data = payload_to_frame(payload)
    features = prepare_features(data, metadata)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = []
    for row_number, (probability, prediction) in enumerate(zip(probabilities, predictions), start=1):
        results.append(
            {
                "row": row_number,
                "churn_probability": to_python_value(round(probability, 6)),
                "churn_prediction": to_python_value(prediction),
                "threshold": threshold,
            }
        )

    return {
        "model": metadata.get("best_model", "unknown"),
        "predictions": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Iranian telecom customer churn from JSON.")
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
