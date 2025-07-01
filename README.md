# Credit Risk Probability Model

## Overview

Build an end-to-end credit scoring pipeline using alternative eCommerce data, from feature engineering through deployment.

## Repo Structure

```
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```
## Credit Scoring Business Understanding

- The Basel II Capital Accord emphasizes not only the quantitative accuracy of credit‐risk estimates but also the qualitative rigor of model governance. Under Pillar II, banks must clearly document feature selection, data preparation, and scoring logic to satisfy regulatory audits. As a result, an interpretable model—such as logistic regression with weight‐of‐evidence encoding—facilitates transparency, auditability, and robust model‐risk management.

- Because true loan‐performance labels are unavailable, we create a **proxy default variable** by clustering customers on Recency,  Frequency, and Monetary metrics and designating the least engaged cluster as high‐risk. This enables supervised training but introduces label risk: misclassifying credit-worthy customers leads to forgone revenue and customer dissatisfaction, while misclassifying risky customers raises exposure to unexpected defaults.

- In a regulated context, model transparency often outweighs marginal gains in accuracy. A logistic‐regression scorecard built with WoE offers clear, additive feature contributions and straightforward documentation. Although gradient boosting machines can deliver superior discrimination by capturing nonlinear interactions, they necessitate post hoc explanation frameworks and more extensive validation to satisfy regulatory scrutiny. We therefore evaluate both paradigms, weighing performance gains against the imperative for transparency and governance.
## Progress to Date
- Task 1 (Business Understanding): Completed in main branch. See above.

- Task 2 (EDA): Executed on task-2 branch. Summary on notebook notebooks/1.0-eda.ipynb:

                * 95,662 transactions loaded, no missing values.
                * Heavy right-skew in Amount/Value, long tails to $9.88 M.
                * financial_services & airtime dominate product categories; ChannelId_3 & ChannelId_2 dominate channels.
                * Amount vs. Value correlation = 0.99 → drop one.
                * Fraud flag extremely imbalanced (0.2% positive) → requires sampling strategies.
- Task 3 (Feature Engineering): Prototyped in task-3 branch, notebook notebooks/feature-engineering.ipynb. 
            Steps completed:
                * RFM Aggregation: Computed Recency, Frequency, Monetary per customer and standardized them.
                * Temporal Aggregation: Generated mean hour, day, month per customer.
                * Categorical Counts: One-hot–encoded product, channel, pricing strategy per transaction and summed per customer.
                * Merged Features: Combined all three into a single customer-level feature table.
                * Visualization: Plotted distributions of RFM, temporal, and top categorical counts to verify feature behavior.

## Next Steps

- Task 4 (Proxy Target Engineering): Compute per-customer RFM, cluster into three segments, label the least engaged as is_high_risk.

- Task 5 (Model Training & Tracking): Split data, train logistic regression & gradient boosting, tune via grid/random search, track with MLflow, write pytest tests.

- Task 6 (Deployment & CI/CD): Create FastAPI /predict endpoint, containerize with Docker, and extend GitHub Actions to build, lint, test, and deploy.
