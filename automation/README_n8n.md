# n8n Notes

Use the root `predict.py` script from an Execute Command node. `automation/predict.py` is just a thin wrapper for people who prefer keeping automation entry points in one folder.

Example command:

```bash
python /workspace/customer-retention-intelligence-system/predict.py --json '{{ JSON.stringify($json) }}'
```

The script prints one JSON object to stdout. The main fields to map in later n8n nodes are:

- `churn_probability`
- `churn_prediction`
- `risk_level`
- `customer_value_tier`
- `retention_priority_score`
- `main_reason`
- `recommended_action`
- `priority`

Optional customer fields such as `customer_id`, `email`, `name`, `nama`, and `phone` are copied back into the output when they exist in the input payload.

Sample local test:

```bash
python predict.py --input sample_customer.json
```
