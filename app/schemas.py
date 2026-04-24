from datetime import datetime

from pydantic import BaseModel


class TissueProbability(BaseModel):
    class_code: str
    label: str
    color: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    predicted_label: str
    cancer_percentage: float
    risk_band: str
    confidence_score: float
    top_probability: float
    model_source: str
    class_probabilities: list[TissueProbability]


class PredictionRecordRead(PredictionResponse):
    id: int
    created_at: datetime
