from pydantic import BaseModel, Field
from typing import Dict

class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Map of feature names to values, matching feature_order.json"
    )

class PredictResponse(BaseModel):
    logistic_proba: float = Field(..., description="Probability from LR model")
    gbm_proba:      float = Field(..., description="Probability from GBM model")
