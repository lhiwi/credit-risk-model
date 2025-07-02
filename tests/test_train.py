import pandas as pd
import mlflow
from src.train import train_and_log


def test_train_and_log_creates_mlflow_runs(tmp_path):
    # Create a tiny synthetic dataset
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
            # Simple proxy label
            "is_high_risk": [1, 0, 1, 0],
        }
    )
    tmp_file = tmp_path / "data.csv"
    df.to_csv(tmp_file, index=False)

    # Clear any existing runs in this experiment for isolation
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name("CreditRiskModels")
    if exp:
        for r in client.list_run_infos(exp.experiment_id):
            client.delete_run(r.run_id)

    # Run training
    train_and_log(
        str(tmp_file), target_col="is_high_risk", test_size=0.5, random_state=0
    )

    # Verify two new runs exist
    exp = mlflow.get_experiment_by_name("CreditRiskModels")
    runs = client.list_run_infos(exp.experiment_id)
    # We expect at least 2 runs (one for LR and one for GBM)
    assert len(runs) >= 2
