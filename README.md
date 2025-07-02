# Credit Risk Probability Model

An end-to-end pipeline for building, deploying, and automating a credit-scoring service using alternative eCommerce data.

## Overview

Bati Bank partners with an eCommerce platform to offer buy-now-pay-later. We derive behavioral RFM (Recency, Frequency, Monetary) features, engineer a proxy “high-risk” label via clustering, train and compare Logistic Regression and Gradient Boosting models, track experiments in MLflow, and expose predictions via a containerized FastAPI service.

## Repo Structure

```
credit-risk-model/
├── .github/
│   └── workflows/ci.yml              # CI/CD pipeline: lint, tests, Docker build, smoke-test
├── data/
│   ├── raw/
│   │   └── data.csv                  # Raw transaction data
│   └── processed/
│       └── features_with_label.csv   # Customer-level features + proxy label
├── notebooks/
│   ├── 1.0-eda.ipynb                 # Exploratory Data Analysis
│   ├── 3.0-proxy-label.ipynb         # Proxy target engineering
│   └── 4.0-model-training.ipynb      # Model training & MLflow tracking
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # EDA & feature pipeline + feature_order.json export
│   ├── label_engineering.py          # RFM clustering → is_high_risk
│   ├── train.py                      # Training script: train & log models to MLflow
│   └── api/
│       ├── __init__.py
│       ├── feature_order.json        # Ordered feature list for API
│       ├── pydantic_models.py        # Request/response schemas
│       └── main.py                   # FastAPI service serving /predict
├── tests/
│   ├── conftest.py                   # Fixtures for API tests (feature_order + MLflow stub)
│   ├── test_dummy.py                 # Sanity pytest test
│   ├── test_label_engineering.py     # Unit tests for RFM proxy labeling
│   ├── test_train.py                 # Unit tests for train.py & MLflow runs
│   └── test_api.py                   # Integration test for /predict
├── Dockerfile                        # Containerize the API
├── docker-compose.yml                # Local orchestration
├── requirements.txt                  # All Python dependencies
└── README.md                         # This file
```

### Credit Scoring Business Understanding

- The Basel II Capital Accord emphasizes not only the quantitative accuracy of credit‐risk estimates but also the qualitative rigor of model governance. Under Pillar II, banks must clearly document feature selection, data preparation, and scoring logic to satisfy regulatory audits. As a result, an interpretable model—such as logistic regression with weight‐of‐evidence encoding—facilitates transparency, auditability, and robust model‐risk management.

- Because true loan‐performance labels are unavailable, we create a proxy default variable by clustering customers on Recency,          Frequency, and Monetary metrics and designating the least engaged cluster as high‐risk. This enables supervised training but introduces label risk: misclassifying credit-worthy customers leads to forgone revenue and customer dissatisfaction, while misclassifying risky customers raises exposure to unexpected defaults.

- In a regulated context, model transparency often outweighs marginal gains in accuracy. A logistic‐regression scorecard built with WoE
offers clear, additive feature contributions and straightforward documentation. Although gradient boosting machines can deliver superior discrimination by capturing nonlinear interactions, they necessitate post hoc explanation frameworks and more extensive validation to satisfy regulatory scrutiny. We therefore evaluate both paradigms, weighing performance gains against the imperative for transparency and governance.

## Installation & Setup

```bash
git clone https://github.com/lhiwi/credit-risk-model.git
cd credit-risk-model
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.\.venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

1. **Prepare data**

   * Place raw data at `data/raw/data.csv`.
2. **Generate features**

   ```bash
   python src/data_processing.py \
     --input data/raw/data.csv \
     --output data/processed/features_with_label.csv
   ```
3. **Train models**

   ```bash
   python src/train.py --data data/processed/features_with_label.csv
   ```
4. **Serve API**

   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

## Using the API

**POST /predict**

**Request**

```json
{
  "features": {
    "Recency": 10.0,
    "Frequency": 5.0,
    "Monetary": 100.0,
    "hour": 14.0,
    "day": 3.0,
    "month": 7.0,
    "...": "other categorical counts"
  }
}
```

**Response**

```json
{
  "logistic_proba": 0.23,
  "gbm_proba": 0.17
}
```


## CI/CD

* **GitHub Actions** on `push`/`pull_request`:

  1. Black format check
  2. Flake8 lint
  3. Pytest unit & integration tests
  4. Build Docker image
  5. Launch API container & smoke-test `/predict`
* **Merge** only after CI passes green.

