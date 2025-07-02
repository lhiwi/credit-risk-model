# tests/conftest.py

import os
import json
import pytest
import mlflow.pyfunc

# 1) Ensure src/api/feature_order.json exists
@pytest.fixture(scope="session", autouse=True)
def feature_order_file(tmp_path_factory):
    root = os.getcwd()
    api_dir = os.path.join(root, "src", "api")
    os.makedirs(api_dir, exist_ok=True)
    feature_file = os.path.join(api_dir, "feature_order.json")
    # write a minimal feature order that your API expects
    feature_order = ["Recency","Frequency","Monetary","hour","day","month"]
    with open(feature_file, "w") as f:
        json.dump(feature_order, f)
    yield feature_file

# 2) Monkey-patch mlflow to return a dummy model
class DummyModel:
    def predict(self, df):
        return [0.5] * len(df)

@pytest.fixture(autouse=True)
def patch_mlflow(monkeypatch):
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: DummyModel())
