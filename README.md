# Claim Detection API

A fine-tuned language model for sentence-level factual claim detection, served via a REST API.

**Claim detection** is the task of identifying whether a natural language sentence contains a factual assertion that could be fact-checked — it is upstream of fact-checking itself.

> Example: *"The Empire State Building is the tallest building in New York City"* → `{"is_claim": true, "confidence": 0.99}`

## Overview

This project fine-tunes `bert-base-uncased` on a composite dataset of ~13,000 labeled sentences and serves the model via a FastAPI endpoint. The approach is informed by [Bell (2025)](https://aclanthology.org/2025.fever-1.6/), which finds that smaller fine-tuned BERT-based models outperform LLMs on in-domain claim detection tasks.

## Project Structure

```
.
├── docs/               # Architecture diagram and design notes
├── data/               # Dataset download and preprocessing scripts
├── training/           # Fine-tuning and evaluation code
├── api/                # FastAPI application
├── tests/              # Unit, integration, and load tests
├── Dockerfile
└── docker-compose.yml
```

## Quickstart

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)

### 1. Download the dataset

```bash
cd data && python download_data.py
```

### 2. Fine-tune the model

```bash
cd training
pip install -r requirements.txt
python train.py
```

> **Note on training time:** Full fine-tuning of `bert-base-uncased` on the complete ~10,400-sample training set for 5 epochs takes approximately 8–15 hours on CPU. By default, `train.py` runs a reduced demo using 2,000 samples and 3 epochs (~7 min on Apple Silicon MPS, ~20–30 min on CPU), which is sufficient to produce a functional model. Pass `--full` to run the complete training run.

> **Note on model weights:** `training/model_output/` is gitignored because the weights are too large to store in version control. You must run `train.py` (step 2) before running the API or integration tests — both depend on the weights being present at that path.

### 3. Run the API

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

### 4. Run with Docker

> **Prerequisite:** complete steps 1 and 2 first — the Docker image bakes the model weights in at build time, so `training/model_output/` must exist before running `docker-compose up --build`.

```bash
docker-compose up --build
```

## API

`POST /predict`

```json
// Request
{ "sentence": "The Eiffel Tower is located in Paris, France." }

// Response
{ "is_claim": true, "confidence": 0.97 }
```

## Model Performance

Evaluated on the held-out test set (2,600 samples). See [`docs/evaluation.md`](docs/evaluation.md) for the full report, confusion matrix, and comparison to paper benchmarks.

| Metric    | This model (demo run) | Bell (2025) full BERT |
|-----------|-----------------------|-----------------------|
| Accuracy  | 0.899                 | 0.917                 |
| Precision | 0.900                 | 0.918                 |
| Recall    | 0.880                 | 0.904                 |
| F1        | 0.890                 | 0.911                 |

The demo run uses 2,000 training samples and 3 epochs. The ~2% F1 gap vs. the paper's full run (10,397 samples, 5 epochs) is expected and closes with more training data.

## Testing

```bash
# Unit and integration tests
pytest tests/

# Load testing (requires running API)
locust -f tests/locustfile.py --host=http://localhost:8000
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the system design diagram and production deployment considerations.
