# tests/test_api.py — FastAPI endpoint tests (uses httpx TestClient)

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Patch heavy model loading before importing api
with patch("api.BertSentimentClassifier") as MockModel, \
     patch("api.BertTokenizer") as MockTok, \
     patch("api.torch.load"), \
     patch("api.redis.Redis") as MockRedis:

    mock_instance = MagicMock()
    MockModel.return_value = mock_instance
    mock_instance.return_value = MagicMock()

    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    MockRedis.return_value = mock_redis

    from api import app

client = TestClient(app)

MOCK_RESULT = {
    "label":      "positive",
    "confidence": 0.9312,
    "scores":     {"negative": 0.0341, "neutral": 0.0347, "positive": 0.9312},
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


@patch("api._infer", return_value=MOCK_RESULT)
def test_predict_single(mock_infer):
    r = client.post("/predict", json={"text": "This product is wonderful!"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"]      == "positive"
    assert data["confidence"] == pytest.approx(0.9312)
    assert set(data["scores"].keys()) == {"negative", "neutral", "positive"}


@patch("api._infer", return_value=MOCK_RESULT)
def test_predict_batch(mock_infer):
    r = client.post("/predict/batch", json={"texts": ["Great!", "Terrible.", "Okay."]})
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_predict_empty_text():
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_missing_field():
    r = client.post("/predict", json={})
    assert r.status_code == 422
