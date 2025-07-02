import json
from fastapi.testclient import TestClient

def test_predict_endpoint(feature_order_file):
    # Load the JSON our fixture created
    FEATURE_ORDER = json.load(open(feature_order_file))

    # Now import the app *after* feature_order exists and MLflow is patched
    from src.api.main import app, FEATURE_ORDER as APP_FEATURE_ORDER

    # Confirm app uses the same order
    assert APP_FEATURE_ORDER == FEATURE_ORDER

    client = TestClient(app)
    dummy = {feat: 0.0 for feat in FEATURE_ORDER}
    response = client.post("/predict", json={"features": dummy})

    assert response.status_code == 200
    body = response.json()
    assert body["logistic_proba"] == 0.5
    assert body["gbm_proba"]      == 0.5
