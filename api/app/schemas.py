from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=2000)

    model_config = {"json_schema_extra": {"examples": [{"sentence": "The Eiffel Tower is located in Paris, France."}]}}


class PredictResponse(BaseModel):
    is_claim: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    sentence: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
