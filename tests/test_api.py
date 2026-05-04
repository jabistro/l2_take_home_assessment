"""
Unit and integration tests for the claim detection API.

Tests are split into two groups:
- Unit tests: mock the classifier, test HTTP contract and validation logic
- Integration tests: require a real model in training/model_output/ (skipped if absent)
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "training", "model_output")
MODEL_AVAILABLE = os.path.exists(MODEL_DIR)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """TestClient with the classifier mocked out — no model weights needed."""
    with patch("api.app.classifier.load_model"), patch("api.app.classifier.is_loaded", return_value=True):
        from api.app.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


@pytest.fixture
def real_client():
    """TestClient backed by actual model weights. Skipped when model absent."""
    if not MODEL_AVAILABLE:
        pytest.skip("Model not found at training/model_output/ — run train.py first")
    from api.app.main import app

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_200(mock_client):
    r = mock_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Input validation (unit, no model needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected_status",
    [
        ({"sentence": "The Eiffel Tower is in Paris."}, 200),
        ({"sentence": ""}, 422),  # empty string
        ({}, 422),  # missing field
        ({"sentence": "x" * 2001}, 422),  # exceeds max_length
        ({"sentence": "   "}, 422),  # whitespace-only (stripped to empty)
        ({"sentence": 12345}, 422),  # wrong type
    ],
)
def test_input_validation(mock_client, payload, expected_status):
    with patch("api.app.main.predict", return_value=(True, 0.95)):
        r = mock_client.post("/predict", json=payload)
    assert r.status_code == expected_status


def test_response_schema(mock_client):
    with patch("api.app.main.predict", return_value=(True, 0.95)):
        r = mock_client.post("/predict", json={"sentence": "The Earth is round."})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["is_claim"], bool)
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["sentence"], str)


def test_confidence_bounds(mock_client):
    for conf in [0.0, 0.5, 1.0]:
        with patch("api.app.main.predict", return_value=(True, conf)):
            r = mock_client.post("/predict", json={"sentence": "Some sentence."})
        assert r.status_code == 200
        assert r.json()["confidence"] == conf


def test_sentence_echoed_back(mock_client):
    sentence = "Water boils at 100 degrees Celsius."
    with patch("api.app.main.predict", return_value=(False, 0.8)):
        r = mock_client.post("/predict", json={"sentence": sentence})
    assert r.json()["sentence"] == sentence


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def test_sql_injection_is_handled(mock_client):
    payload = {"sentence": "'; DROP TABLE sentences; --"}
    with patch("api.app.main.predict", return_value=(False, 0.6)):
        r = mock_client.post("/predict", json=payload)
    assert r.status_code == 200


def test_script_injection_is_handled(mock_client):
    payload = {"sentence": "<script>alert('xss')</script>"}
    with patch("api.app.main.predict", return_value=(False, 0.7)):
        r = mock_client.post("/predict", json=payload)
    assert r.status_code == 200


def test_oversized_payload_rejected(mock_client):
    r = mock_client.post("/predict", json={"sentence": "a" * 10_000})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Integration tests (requires trained model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence,expected_is_claim",
    [
        ("The Eiffel Tower is located in Paris, France.", True),
        ("The unemployment rate fell to 3.7 percent last month.", True),
        ("I love this movie so much!", False),
        ("What do you think about this?", False),
    ],
)
def test_real_model_known_examples(real_client, sentence, expected_is_claim):
    r = real_client.post("/predict", json={"sentence": sentence})
    assert r.status_code == 200
    body = r.json()
    assert body["is_claim"] == expected_is_claim, (
        f"Expected is_claim={expected_is_claim} for: '{sentence}', got {body}"
    )
