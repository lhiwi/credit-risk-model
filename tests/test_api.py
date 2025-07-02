from fastapi.testclient import TestClient
import json
from src.api.main import app, FEATURE_ORDER

client = TestClient(app)

def test_predict_endpoint():
    dummy = {feat: 0.0 for feat in FEATURE_ORDER}
    resp = client.post("/predict", json={"features": dummy})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["logistic_proba"] <= 1.0
    assert 0.0 <= body["gbm_proba"] <= 1.0
