# Customer Churn Automation

Customer churn prediction project for Iranian telecom customer data. This repository contains the notebook used to train the model, the trained model artifact, and a Python bridge script that can be called from n8n using the Execute Command node.

## Project Contents

| Path | Description |
| --- | --- |
| `Customer Churn.csv` | Original Iranian customer churn dataset from UCI Machine Learning Repository |
| `Customer_Churn_Prediction.ipynb` | Jupyter notebook for EDA, feature engineering, model training, evaluation, and XAI |
| `predict.py` | Python bridge script for n8n integration |
| `sample_customer.json` | Sample customer payload for local prediction testing |
| `artifacts/iranian_churn_model.joblib` | Trained Random Forest model |
| `artifacts/model_metadata.json` | Model metadata, feature list, threshold, and test metrics |
| `Dockerfile` | Custom n8n image setup for Python execution |

## Model Summary

The final model is a Random Forest classifier trained to predict whether a customer will churn.

Test metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9632 |
| Precision | 0.9048 |
| Recall | 0.8539 |
| F1 Score | 0.8786 |
| ROC-AUC | 0.9841 |
| PR-AUC | 0.9048 |

The production threshold is stored in `artifacts/model_metadata.json`.

## Input Schema

`predict.py` expects these customer fields:

```json
{
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

Extra fields such as `email` and `nama` are allowed. They are ignored by the model.

## Local Usage

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

Run prediction with the sample payload:

```bash
python predict.py --input sample_customer.json
```

Expected output:

```json
{
  "model": "random_forest",
  "predictions": [
    {
      "row": 1,
      "churn_probability": 0.631285,
      "churn_prediction": 1,
      "threshold": 0.48215135924114494
    }
  ]
}
```

You can also pass JSON directly:

```bash
python predict.py --json "{\"call_failure\":12,\"complains\":1,\"subscription_length\":38,\"charge_amount\":0,\"seconds_of_use\":4370,\"frequency_of_use\":71,\"frequency_of_sms\":5,\"distinct_called_numbers\":17,\"age_group\":3,\"tariff_plan\":1,\"status\":1,\"age\":30,\"customer_value\":197.64}"
```

## n8n Integration

Use the n8n Execute Command node to call `predict.py`.

Example command:

```bash
python /path/to/predict.py --json '{{ JSON.stringify($json) }}'
```

If the workflow runs inside Docker, make sure these files are available inside the container:

```text
predict.py
artifacts/iranian_churn_model.joblib
artifacts/model_metadata.json
```

The command output is JSON, so n8n can parse it in the next node.

## Execute Command Node Notes

n8n may hide or disable the Execute Command node for security reasons. This node can run shell commands on the host/container, so it is powerful but risky if exposed to untrusted users.

For self-hosted n8n, check the `NODES_EXCLUDE` environment variable. To enable all nodes, set:

```bash
NODES_EXCLUDE="[]"
```

For Docker Compose:

```yaml
environment:
  - NODES_EXCLUDE=[]
```

If `Execute Command` is still unavailable, restart the n8n container after changing the environment variable.

Official n8n docs:

- https://docs.n8n.io/hosting/securing/blocking-nodes/
- https://docs.n8n.io/2-0-breaking-changes/
- https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.executecommand/

## Docker

Build the custom image:

```bash
docker build -t n8n-python .
```

Run n8n with Execute Command enabled:

```bash
docker run -it --rm \
  --name n8n-python \
  -p 5678:5678 \
  -e NODES_EXCLUDE="[]" \
  n8n-python
```

If Docker reports that `apk` or `apt-get` is not found, the base n8n image has changed. Adjust the Dockerfile to match the package manager available in the selected base image.

## Repository

Target repository:

```text
https://github.com/heyitzrizki/Customer-Churn-Automation
```

Recommended push flow:

```bash
git init
git remote add origin https://github.com/heyitzrizki/Customer-Churn-Automation.git
git add .
git commit -m "Add customer churn automation project"
git branch -M main
git push -u origin main
```
