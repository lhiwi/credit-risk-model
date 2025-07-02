import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def train_and_log(
    data_path: str,
    target_col: str = "is_high_risk",
    test_size: float = 0.2,
    random_state: int = 42,
):
    # 1. Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col, "CustomerId"])
    y = df[target_col]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 3. Scale features for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_train = len(y_train)
    # ensure at least 2 folds, and no more than n_train
    lr_cv = max(2, min(5, n_train))
    gbm_cv = max(2, min(3, n_train))

    # 4. Logistic Regression grid search
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    grid_lr = GridSearchCV(lr, param_grid, scoring="roc_auc", cv=lr_cv, n_jobs=-1)
    grid_lr.fit(X_train_scaled, y_train)
    best_lr = grid_lr.best_estimator_
    lr_auc = roc_auc_score(y_test, best_lr.predict_proba(X_test_scaled)[:, 1])

    # 5. Gradient Boosting random search
    gbm = HistGradientBoostingClassifier(random_state=random_state)
    param_dist = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [20, 50, 100],
    }
    rand_gbm = RandomizedSearchCV(
        gbm,
        param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=gbm_cv,
        random_state=random_state,
        n_jobs=-1,
    )
    rand_gbm.fit(X_train, y_train)
    best_gbm = rand_gbm.best_estimator_
    gbm_auc = roc_auc_score(y_test, best_gbm.predict_proba(X_test)[:, 1])

    # 6. Log to MLflow
    mlflow.set_experiment("CreditRiskModels")
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.log_params(grid_lr.best_params_)
        mlflow.log_metric("roc_auc", lr_auc)
        mlflow.sklearn.log_model(best_lr, "LogisticRegression")

    with mlflow.start_run(run_name="GradientBoosting"):
        mlflow.log_params(rand_gbm.best_params_)
        mlflow.log_metric("roc_auc", gbm_auc)
        mlflow.sklearn.log_model(best_gbm, "GradientBoosting")

    print(f"Logged LR (AUC={lr_auc:.4f}) and GBM (AUC={gbm_auc:.4f}) to MLflow.")
