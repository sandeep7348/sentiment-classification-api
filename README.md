# Sentiment Analysis System

A production-ready 3-class sentiment classifier (Negative / Neutral / Positive) built with fine-tuned BERT, served via a FastAPI REST API, and containerised with Docker.

**Accuracy: 91.4% | F1 (macro): 0.913**

---

## Project Structure

```
sentiment-analysis/
├── model.py           # BERT classifier + Dataset class
├── train.py           # Training loop (AdamW, warmup, early stopping)
├── preprocess.py      # Text cleaning, lemmatisation, sarcasm features
├── evaluate.py        # Metrics, confusion matrix, error analysis
├── predict.py         # CLI / importable inference helper
├── api.py             # FastAPI server with Redis caching
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── checkpoints/       # Saved model weights (.pt)
├── data/
│   ├── raw/           # Original CSVs (IMDb, Twitter, Amazon)
│   └── processed/     # Cleaned train/val/test splits
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_preprocess.py
└── .github/workflows/ci.yml
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py --data data/processed/train.csv
```

The best checkpoint is saved to `checkpoints/bert_sentiment.pt`.

### 3. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Or with Docker:

```bash
docker-compose up --build
```

### 4. Predict

**CLI:**
```bash
python predict.py "The battery life on this phone is incredible!"
```

**API:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Worst purchase I have ever made."}'
```

**Response:**
```json
{
  "label": "negative",
  "confidence": 0.9481,
  "scores": {
    "negative": 0.9481,
    "neutral":  0.0312,
    "positive": 0.0207
  }
}
```

---

## API Endpoints

| Method | Endpoint         | Description                   |
|--------|-----------------|-------------------------------|
| GET    | `/health`        | Health check                  |
| POST   | `/predict`       | Single-text prediction        |
| POST   | `/predict/batch` | Batch prediction (up to 64)   |

Interactive docs available at `http://localhost:8000/docs`.

---

## Model Details

| Component       | Choice                         |
|----------------|-------------------------------|
| Base model      | `bert-base-uncased`           |
| Optimizer       | AdamW (lr=2e-5, wd=0.01)      |
| Scheduler       | Linear warmup (10%) + decay   |
| Loss            | CrossEntropy with class weights|
| Dropout         | 0.3                           |
| Max sequence    | 128 tokens                    |
| Epochs          | 5 (early stopping, patience=3)|

---

## Running Tests

```bash
pytest tests/ -v --cov=.
```

---

## Dataset Sources

- [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Sentiment140 (Twitter)](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [Amazon Product Reviews](https://nijianmo.github.io/amazon/)
- [SemEval-2017 Task 4](https://alt.qcri.org/semeval2017/task4/)

---

## License
