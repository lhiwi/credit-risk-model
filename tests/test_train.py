import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from src.train import train_and_log


def test_train_and_log_creates_mlflow_runs(tmp_path):
    # 1. Build a tiny synthetic dataset
    df = pd.DataFrame(
        {
            "CustomerId": ["A", "B", "C", "D"],
            "Recency": [10, 20, 30, 40],
            "Frequency": [5, 4, 3, 2],
            "Monetary": [100, 200, 300, 400],
            "hour": [12, 13, 14, 15],
            "day": [1, 2, 3, 4],
            "month": [6, 7, 8, 9],
            "ProductCategory_A": [1, 0, 1, 0],
            "ChannelId_X": [0, 1, 0, 1],
            "PricingStrategy_1": [1, 1, 0, 0],
            "is_high_risk": [1, 0, 1, 0],
        }
    )
    tmp_file = tmp_path / "data.csv"
    df.to_csv(tmp_file, index=False)

    # 2. Clear existing runs
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name("CreditRiskModels")
    if exp:
        runs = client.search_runs([exp.experiment_id])
        for r in runs:
            client.delete_run(r.info.run_id)

    # 3. Run training
    train_and_log(
        str(tmp_file), target_col="is_high_risk", test_size=0.5, random_state=0
    )

    # 4. Verify at least two runs (LR and GBM) were created
    exp = mlflow.get_experiment_by_name("CreditRiskModels")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) >= 2
