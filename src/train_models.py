import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    MODEL_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_training_frame,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "Customer Churn.csv"
FALLBACK_DATA_PATH = BASE_DIR / "Customer Churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = ARTIFACTS_DIR / "iranian_churn_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = PROCESSED_DIR / "model_comparison.csv"


def load_dataset() -> pd.DataFrame:
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    return pd.read_csv(path)


def available_models() -> dict:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except Exception as error:
        print(f"Skipping XGBoost: {error}")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as error:
        print(f"Skipping LightGBM: {error}")

    return models


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def score_model(model, x_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }


def risk_driver_thresholds(data: pd.DataFrame) -> dict:
    return {
        "failed_call_rate_high": float(data["failed_call_rate"].quantile(0.75)),
        "frequency_of_use_low": float(data["frequency_of_use"].quantile(0.25)),
        "seconds_of_use_low": float(data["seconds_of_use"].quantile(0.25)),
        "customer_value_low": float(data["customer_value"].quantile(0.25)),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_dataset()
    data, value_thresholds = prepare_training_frame(raw_data)

    x = data[MODEL_FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    fitted_models = {}
    thresholds = {}

    for model_name, model in available_models().items():
        try:
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            threshold = choose_threshold(y_test, probabilities)
            metrics = score_model(model, x_test, y_test, threshold)
            results.append({"model": model_name, "threshold": threshold, **metrics})
            fitted_models[model_name] = model
            thresholds[model_name] = threshold
            print(f"Trained {model_name}: PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}")
        except Exception as error:
            print(f"Skipping {model_name}: {error}")

    if not results:
        raise RuntimeError("No models could be trained.")

    comparison = pd.DataFrame(results).sort_values(["pr_auc", "f1", "recall"], ascending=False)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    best_model_name = str(comparison.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_threshold = float(thresholds[best_model_name])
    best_metrics = comparison.iloc[0].drop(labels=["model", "threshold"]).to_dict()

    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "dataset": "Customer Churn.csv",
        "target": TARGET_COLUMN,
        "best_model": best_model_name,
        "model_comparison_metrics": comparison.to_dict(orient="records"),
        "test_metrics": {key: float(value) for key, value in best_metrics.items()},
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "categorical_features": ["complains", "age_group", "tariff_plan", "status"],
        "numeric_features": [
            column
            for column in MODEL_FEATURE_COLUMNS
            if column not in ["complains", "age_group", "tariff_plan", "status"]
        ],
        "threshold": best_threshold,
        "selected_threshold": best_threshold,
        "customer_value_quantile_thresholds": value_thresholds,
        "risk_level_thresholds": {"high": 0.65, "medium": 0.35, "low": 0.0},
        "risk_driver_thresholds": risk_driver_thresholds(data),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "dataset_row_count": int(len(data)),
        "churn_rate": float(y.mean()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Best model: {best_model_name}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Saved model comparison to {MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
