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

> **Note on training time:** Full fine-tuning of `bert-base-uncased` on the complete ~10,400-sample training set for 5 epochs takes approximately 8–15 hours on CPU. By default, `train.py` runs a reduced demo using 2,000 samples and 3 epochs (~20–30 min on CPU), which is sufficient to produce a functional model. Pass `--full` to run the complete training run.

### 3. Run the API

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

### 4. Run with Docker

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

See [`docs/evaluation.md`](docs/evaluation.md) for full metrics and interpretation.

| Metric    | Score |
|-----------|-------|
| Accuracy  | TBD   |
| Precision | TBD   |
| Recall    | TBD   |
| F1        | TBD   |

## Testing

```bash
# Unit and integration tests
pytest tests/

# Load testing (requires running API)
locust -f tests/locustfile.py --host=http://localhost:8000
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the system design diagram and production deployment considerations.
