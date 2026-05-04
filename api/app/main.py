from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .classifier import is_loaded, load_model, predict
from .schemas import HealthResponse, PredictRequest, PredictResponse

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Claim Detection API",
    description=(
        "Identifies whether a sentence contains a factual claim that could be fact-checked. "
        "Powered by a fine-tuned bert-base-uncased model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(status="ok", model_loaded=is_loaded())


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
@limiter.limit("60/minute")
async def predict_claim(request: Request, body: PredictRequest):
    sentence = body.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=422, detail="sentence must not be empty")

    is_claim, confidence = predict(sentence)
    return PredictResponse(is_claim=is_claim, confidence=confidence, sentence=sentence)
