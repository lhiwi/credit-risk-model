# Credit Risk Probability Model

## Overview

Build an end-to-end credit scoring pipeline using alternative eCommerce data, from feature engineering through deployment.

## Repo Structure

```text
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
## Credit Scoring Business Understanding

- The Basel II Capital Accord emphasizes not only the quantitative accuracy of credit‐risk estimates but also the qualitative rigor of model governance. Under Pillar II, banks must clearly document feature selection, data preparation, and scoring logic to satisfy regulatory audits. As a result, an interpretable model—such as logistic regression with weight‐of‐evidence encoding—facilitates transparency, auditability, and robust model‐risk management.

- Because true loan‐performance labels are unavailable, we create a **proxy default variable** by clustering customers on Recency,  Frequency, and Monetary metrics and designating the least engaged cluster as high‐risk. This enables supervised training but introduces label risk: misclassifying credit-worthy customers leads to forgone revenue and customer dissatisfaction, while misclassifying risky customers raises exposure to unexpected defaults.

- In a regulated context, model transparency often outweighs marginal gains in accuracy. A logistic‐regression scorecard built with WoE offers clear, additive feature contributions and straightforward documentation. Although gradient boosting machines can deliver superior discrimination by capturing nonlinear interactions, they necessitate post hoc explanation frameworks and more extensive validation to satisfy regulatory scrutiny. We therefore evaluate both paradigms, weighing performance gains against the imperative for transparency and governance.
