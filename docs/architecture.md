# System Architecture

## Overview

The claim detection system has three layers: a training pipeline that produces model weights offline, a FastAPI inference service that loads those weights and serves predictions, and a containerized deployment wrapping the service for production use.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE (offline)                  │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Composite   │    │ bert-base-   │    │  HuggingFace Trainer  │  │
│  │  Dataset     │───▶│ uncased      │───▶│  (full param FT)      │  │
│  │  ~13k rows   │    │  (pretrained)│    │  5 epochs / 80% split │  │
│  └──────────────┘    └──────────────┘    └──────────┬────────────┘  │
│                                                     │               │
│                                              ┌──────▼──────┐        │
│                                              │ model_output│        │
│                                              │ (weights +  │        │
│                                              │  tokenizer) │        │
│                                              └─────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    model weights copied into image
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE SERVICE                           │
│                                                                     │
│   Client                  FastAPI app                 BERT model    │
│                                                                     │
│  POST /predict  ──────▶  Input validation   ──────▶  Tokenizer      │
│  {sentence}              (Pydantic schema)           + Forward pass │
│                                                              │      │
│  {is_claim,    ◀──────  Format response    ◀──────  Softmax  │      │
│   confidence}            + echo sentence             probabilities  │
│                                                                     │
│   GET /health  ──────▶  model_loaded: bool                          │
│                                                                     │
│  Rate limiting: 60 req/min per IP (slowapi)                         │
│  Input size:   max 2,000 chars (Pydantic)                           │
│  Model load:   once at startup via lifespan hook                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                            docker-compose up
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CONTAINER (Docker)                          │
│                                                                     │
│  python:3.11-slim base image                                        │
│  Model weights baked into image at build time                       │
│  Health check polls /health every 30s                               │
│  Port 8000 exposed                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Production considerations

### What would change for a real deployment

**Model serving**
The current design bakes model weights into the Docker image. In production this is impractical for large models — weights should be stored in object storage (e.g. S3) and pulled at container startup, or served via a dedicated model registry (e.g. Hugging Face Inference Endpoints, Triton Inference Server).

**Horizontal scaling**
The model is loaded once per container process. To scale horizontally, a load balancer (e.g. AWS ALB, nginx) would sit in front of multiple container replicas. Because BERT inference is CPU/GPU-bound and stateless, this scales well.

**Latency**
On CPU, a single BERT forward pass takes ~50–200ms per sentence. For lower latency at higher throughput, options include: quantization (INT8 BERT via ONNX Runtime, ~2–4× speedup), GPU instances, or batching requests.

**Observability**
A production deployment would add: structured logging (e.g. structlog → CloudWatch/Datadog), request tracing (e.g. OpenTelemetry), Prometheus metrics (latency histograms, error rates, throughput), and model monitoring for distribution shift (confidence score distributions drift over time as the input domain changes).

**Rate limiting**
The current in-process `slowapi` rate limiter resets on restart and doesn't share state across replicas. Production would use a Redis-backed rate limiter.

**Security**
- HTTPS termination at the load balancer level (TLS cert via ACM or Let's Encrypt)
- API key authentication for non-public deployments
- Input sanitization is already handled by Pydantic's type validation; the model itself treats input as plain text so injection attacks don't apply at the ML layer

**CI/CD**
A GitHub Actions pipeline would: run `pytest`, build and push the Docker image to ECR/GHCR, and trigger a rolling deployment on ECS/EKS.

## Dataset

The composite training dataset combines three sources, all human-curated:

| Source | Records | % Claims |
|---|---|---|
| Claimbuster (Hassan et al., 2017) | 7,976 | 25% |
| PoliClaim Gold (Ni et al., 2024) | 1,953 | 59% |
| AVeriTeC (Schlichtkrull et al., 2023) | 3,068 | 100% |
| **Total** | **12,997** | **48%** |

80/20 train/test split, frozen before any fine-tuning.

## Model choice rationale

Per Bell (2025), fine-tuned BERT-based models outperform LLMs on in-domain claim detection while requiring significantly less compute for inference. `bert-base-uncased` (110M params) achieves 91.7% accuracy and 91.1% F1 on the composite dataset in the paper's full fine-tuning run — the best result across all models tested, including GPT-3.5 Turbo.

The tradeoff: BERT-based models generalize poorly to out-of-domain text (e.g. tweets), tending to over-predict claims. For a system restricted to political speeches and fact-checks, BERT is the right call. For unrestricted domains, an LLM would be more robust.
