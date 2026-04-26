# sentiment_analysis/predict.py — quick CLI / importable inference helper

import argparse
import sys
import torch
from transformers import BertTokenizer
from model import BertSentimentClassifier

LABELS = ["negative", "neutral", "positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model     = None
_tokenizer = None


def _load(checkpoint: str = "checkpoints/bert_sentiment.pt"):
    global _model, _tokenizer
    if _model is None:
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _model     = BertSentimentClassifier()
        _model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        _model.to(DEVICE).eval()
    return _model, _tokenizer


def predict(text: str, checkpoint: str = "checkpoints/bert_sentiment.pt") -> dict:
    """
    Predict sentiment for a single string.

    Returns:
        {"label": "positive", "confidence": 0.93, "scores": {...}}
    """
    model, tokenizer = _load(checkpoint)
    enc = tokenizer(text, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"].to(DEVICE),
                       enc["attention_mask"].to(DEVICE))
    probs    = torch.softmax(logits, dim=1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits))
    return {
        "label":      LABELS[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "scores":     {l: round(p, 4) for l, p in zip(LABELS, probs)},
    }


def predict_batch(texts: list, checkpoint: str = "checkpoints/bert_sentiment.pt") -> list:
    return [predict(t, checkpoint) for t in texts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for input text.")
    parser.add_argument("text", nargs="?", help="Text to classify (or pipe via stdin)")
    parser.add_argument("--checkpoint", default="checkpoints/bert_sentiment.pt")
    args = parser.parse_args()

    text = args.text or sys.stdin.read().strip()
    if not text:
        parser.print_help()
        sys.exit(1)

    result = predict(text, args.checkpoint)
    print(f"\nLabel     : {result['label'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Scores    :")
    for label, score in result["scores"].items():
        bar = "█" * int(score * 30)
        print(f"  {label:9s} {score:.4f}  {bar}")
