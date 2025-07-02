
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictRequest, PredictResponse
import mlflow.pyfunc

app = FastAPI(title="Credit Risk Prediction API")

#  Load feature order at module load time
try:
    with open("src/api/feature_order.json") as f:
        FEATURE_ORDER = json.load(f)
except Exception as e:
    # If the file is missing, the app import itself will error early
    raise RuntimeError(f"Could not load feature_order.json: {e}")

#  Load models from MLflow Model Registry
#    (point to your correct stage or version)
LR_MODEL  = mlflow.pyfunc.load_model("models:/CreditRiskModels/LogisticRegression/Production")
GBM_MODEL = mlflow.pyfunc.load_model("models:/CreditRiskModels/GradientBoosting/Production")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Build DataFrame in the correct order
        df = pd.DataFrame([request.features])[FEATURE_ORDER]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features: {e}")

    lr_proba  = LR_MODEL.predict(df)[0]
    gbm_proba = GBM_MODEL.predict(df)[0]
    return PredictResponse(logistic_proba=lr_proba, gbm_proba=gbm_proba)
