# sentiment_analysis/api.py — FastAPI inference server with VADER (no training needed)

import hashlib
import json
import logging
import time
from typing import List, Optional

import redis
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_ENABLED = True

    REQUEST_COUNT = Counter(
        "sentiment_requests_total",
        "Total prediction requests",
        ["endpoint", "label", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "sentiment_request_latency_seconds",
        "Prediction latency in seconds",
        ["endpoint"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    CACHE_HITS = Counter(
        "sentiment_cache_hits_total",
        "Redis cache hits",
    )
    CACHE_MISSES = Counter(
        "sentiment_cache_misses_total",
        "Redis cache misses",
    )
    LABEL_DISTRIBUTION = Counter(
        "sentiment_label_total",
        "Predictions per sentiment label",
        ["label"],
    )
    CONFIDENCE_HISTOGRAM = Histogram(
        "sentiment_confidence",
        "Model confidence scores",
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    )
    MODEL_LOADED = Gauge("sentiment_model_loaded", "Whether the model is loaded (1=yes)")
    log.info("Prometheus metrics enabled")
except ImportError:
    PROMETHEUS_ENABLED = False
    log.warning("prometheus_client not installed — metrics disabled. Run: pip install prometheus-client")


# ── App setup ──────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="3-class sentiment classification powered by VADER.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────
LABELS = ["negative", "neutral", "positive"]

# ── Load VADER analyzer at startup (instant, no training needed) ──
analyzer = SentimentIntensityAnalyzer()
log.info("VADER analyzer loaded successfully")

if PROMETHEUS_ENABLED:
    MODEL_LOADED.set(1)

# ── Redis cache ────────────────────────────────────────────────
try:
    cache = redis.Redis(
        host="localhost", port=6379,
        decode_responses=True, socket_connect_timeout=2
    )
    cache.ping()
    CACHE_ENABLED = True
    log.info("Redis cache connected")
except Exception:
    cache = None
    CACHE_ENABLED = False
    log.warning("Redis unavailable — caching disabled")

CACHE_TTL = 3600

# ── In-memory stats ────────────────────────────────────────────
_stats = {
    "total_requests": 0,
    "cache_hits": 0,
    "errors": 0,
    "label_counts": {"negative": 0, "neutral": 0, "positive": 0},
    "latencies": [],
}


# ── Schemas ────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="The product is amazing!")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=64)

class SentimentResponse(BaseModel):
    label:      str
    confidence: float
    scores:     dict


# ── Core inference using VADER ─────────────────────────────────
def _infer(text: str, endpoint: str = "predict") -> dict:
    t0 = time.perf_counter()

    # Cache lookup
    cache_hit = False
    if CACHE_ENABLED:
        key = hashlib.md5(text.encode()).hexdigest()
        try:
            if cached := cache.get(key):
                cache_hit = True
                if PROMETHEUS_ENABLED:
                    CACHE_HITS.inc()
                _stats["cache_hits"] += 1
                result = json.loads(cached)
                _record_metrics(result, endpoint, time.perf_counter() - t0, "hit")
                return result
        except Exception:
            pass

    if CACHE_ENABLED and not cache_hit:
        if PROMETHEUS_ENABLED:
            CACHE_MISSES.inc()

    # VADER inference
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    # Map compound score to label
    # compound: -1.0 (most negative) to +1.0 (most positive)
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Build confidence and per-label scores
    pos_score  = round(scores["pos"], 4)
    neg_score  = round(scores["neg"], 4)
    neu_score  = round(scores["neu"], 4)

    # Confidence = the winning label's raw score (or abs compound for tie-breaking)
    if label == "positive":
        confidence = round(max(pos_score, (compound + 1) / 2), 4)
    elif label == "negative":
        confidence = round(max(neg_score, (1 - compound) / 2), 4)
    else:
        confidence = round(neu_score, 4)

    result = {
        "label":      label,
        "confidence": confidence,
        "scores": {
            "negative": neg_score,
            "neutral":  neu_score,
            "positive": pos_score,
        },
    }

    if CACHE_ENABLED:
        try:
            cache.setex(key, CACHE_TTL, json.dumps(result))
        except Exception:
            pass

    latency = time.perf_counter() - t0
    _record_metrics(result, endpoint, latency, "miss")
    return result


def _record_metrics(result: dict, endpoint: str, latency: float, cache_status: str):
    label = result["label"]
    conf  = result["confidence"]

    if PROMETHEUS_ENABLED:
        REQUEST_COUNT.labels(endpoint=endpoint, label=label, status="ok").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        LABEL_DISTRIBUTION.labels(label=label).inc()
        CONFIDENCE_HISTOGRAM.observe(conf)

    _stats["total_requests"] += 1
    _stats["label_counts"][label] = _stats["label_counts"].get(label, 0) + 1
    _stats["latencies"] = (_stats["latencies"] + [latency])[-1000:]


# ── Middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-Response-Time"] = f"{duration:.4f}s"
    return response


# ── Routes ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "engine":     "VADER",
        "cache":      CACHE_ENABLED,
        "prometheus": PROMETHEUS_ENABLED,
    }


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    if not PROMETHEUS_ENABLED:
        raise HTTPException(status_code=501, detail="prometheus_client not installed")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics")
async def metrics():
    lats = _stats["latencies"]
    avg_lat = round(sum(lats) / len(lats), 4) if lats else 0
    p95_lat = round(sorted(lats)[int(len(lats) * 0.95)], 4) if lats else 0
    return {
        "total_requests":  _stats["total_requests"],
        "cache_hits":      _stats["cache_hits"],
        "errors":          _stats["errors"],
        "label_counts":    _stats["label_counts"],
        "latency_avg_sec": avg_lat,
        "latency_p95_sec": p95_lat,
        "cache_enabled":   CACHE_ENABLED,
        "engine":          "VADER",
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict(req: TextRequest):
    """Single-text sentiment prediction."""
    try:
        return _infer(req.text, endpoint="predict")
    except Exception as e:
        _stats["errors"] += 1
        if PROMETHEUS_ENABLED:
            REQUEST_COUNT.labels(endpoint="predict", label="none", status="error").inc()
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.post("/predict/batch", response_model=List[SentimentResponse])
async def batch_predict(req: BatchRequest):
    """Batch sentiment prediction (up to 64 texts)."""
    try:
        return [_infer(t, endpoint="batch") for t in req.texts]
    except Exception as e:
        _stats["errors"] += 1
        if PROMETHEUS_ENABLED:
            REQUEST_COUNT.labels(endpoint="batch", label="none", status="error").inc()
        log.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail="Batch inference failed")


@app.get("/")
async def root():
    return {
        "message": "Sentiment Analysis API (VADER)",
        "docs":    "/docs",
        "health":  "/health",
        "metrics": "/metrics",
    }