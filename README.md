# Customer Churn Automation

This project turns the Iranian telecom churn dataset into a small retention workflow:

- train a churn model
- segment customers
- score retention priority
- recommend a follow-up action
- expose a JSON prediction script that can be called from n8n

The repo keeps the notebook and raw dataset for traceability, but the main flow is handled by the Python scripts under `src/` and the n8n-friendly entry point in `predict.py`.

## Repository Layout

```text
app/
  streamlit_app.py
artifacts/
  iranian_churn_model.joblib
  model_metadata.json
  scaler.joblib
  segmentation_model.joblib
automation/
  predict.py
  README_n8n.md
data/
  raw/Customer Churn.csv
  processed/
notebooks/
  Customer_Churn_Prediction.ipynb
src/
  build_segments.py
  explain_risk.py
  feature_engineering.py
  recommend_actions.py
  score_customers.py
  train_models.py
predict.py
sample_customer.json
requirements.txt
Dockerfile
```

Raw data lives in `data/raw/`, and the working notebook lives in `notebooks/`.

## Model

The current saved model is a LightGBM classifier.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9714 |
| Precision | 0.8716 |
| Recall | 0.9596 |
| F1 Score | 0.9135 |
| ROC-AUC | 0.9948 |
| PR-AUC | 0.9744 |

The selected threshold and feature list live in `artifacts/model_metadata.json`.

## Setup

```bash
pip install -r requirements.txt
```

The optional model comparison step will use XGBoost and LightGBM if they are installed. If either package is missing, the training script skips it and continues with the available models.

## Main Commands

Train and compare models:

```bash
python src/train_models.py
```

Build customer segments:

```bash
python src/build_segments.py
```

Score customers and create the retention action tables:

```bash
python src/score_customers.py
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## JSON Prediction

`predict.py` is the script used by n8n. It accepts a JSON object or a list of JSON objects.

```bash
python predict.py --input sample_customer.json
```

Example input:

```json
{
  "email": "johndoe@gatglobal.co.kr",
  "nama": "John Doe",
  "call_failure": 12,
  "complains": 1,
  "subscription_length": 38,
  "charge_amount": 0,
  "seconds_of_use": 4370,
  "frequency_of_use": 71,
  "frequency_of_sms": 5,
  "distinct_called_numbers": 17,
  "age_group": 3,
  "tariff_plan": 1,
  "status": 1,
  "age": 30,
  "customer_value": 197.64
}
```

Example output:

```json
{
  "model": "lightgbm",
  "predictions": [
    {
      "row": 1,
      "customer_id": "CUST_001",
      "churn_probability": 0.687028,
      "churn_prediction": 1,
      "threshold": 0.3292908351136531,
      "risk_level": "High",
      "customer_value_tier": "Medium",
      "retention_priority_score": 49,
      "main_reason": "Complaint recorded",
      "recommended_action": "Personalized retention offer",
      "priority": "P3",
      "email": "johndoe@gatglobal.co.kr",
      "nama": "John Doe"
    }
  ]
}
```

Extra fields such as `email`, `nama`, `name`, and `phone` are preserved in the output when present.

## n8n Usage

Use the Execute Command node and call the root script:

```bash
python /workspace/customer-retention-intelligence-system/predict.py --json '{{ JSON.stringify($json) }}'
```

The command prints JSON to stdout. The next n8n node can parse the output and map these fields into Telegram, HubSpot, or another CRM step:

- `churn_probability`
- `churn_prediction`
- `risk_level`
- `retention_priority_score`
- `main_reason`
- `recommended_action`
- `priority`

More notes are in `automation/README_n8n.md`.

## Dashboard

The Streamlit app has two pages:

1. `Customer Segmentation`: segment size, average value, average churn risk, complaint rate, and suggested handling strategy.
2. `Churn Risk Scoring`: customer-level churn risk, priority, reason, and recommended action.

Before opening the dashboard, generate the processed files:

```bash
python src/train_models.py
python src/build_segments.py
python src/score_customers.py
```

Then run:

```bash
streamlit run app/streamlit_app.py
```

## Notes

- The dataset does not include campaign history, so the recommended actions are rule-based, not causal treatment recommendations.
- `customer_value` is used as a value proxy. It is not proven lost revenue.
- The churn model was trained on a public telecom dataset, so real deployment should include monitoring and retraining with current customer data.
