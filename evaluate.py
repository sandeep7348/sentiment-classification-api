# sentiment_analysis/evaluate.py — evaluation, confusion matrix, error analysis

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)
from transformers import BertTokenizer

from model import BertSentimentClassifier, build_dataloader
from preprocess import batch_clean

LABELS  = ["negative", "neutral", "positive"]
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint: str) -> BertSentimentClassifier:
    m = BertSentimentClassifier()
    m.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return m.to(DEVICE).eval()


@torch.no_grad()
def get_predictions(model, loader):
    all_preds, all_probs, all_targets = [], [], []
    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        logits = model(ids, mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds   += preds.tolist()
        all_probs   += probs.tolist()
        all_targets += batch["label"].numpy().tolist()
    return np.array(all_preds), np.array(all_probs), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — BERT Sentiment Classifier")
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def error_analysis(texts, y_true, y_pred, out_dir: Path, n: int = 20):
    """Save the top-N misclassified examples to CSV."""
    errors = [
        {"text": texts[i], "true": LABELS[y_true[i]], "pred": LABELS[y_pred[i]]}
        for i in range(len(texts))
        if y_true[i] != y_pred[i]
    ][:n]
    df = pd.DataFrame(errors)
    path = out_dir / "error_analysis.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(errors)} errors)")


def main(data_path: str, checkpoint: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["text"]  = batch_clean(df["text"].tolist())
    df["label"] = df["label"].map({l: i for i, l in enumerate(LABELS)})
    df = df.dropna(subset=["label"])
    texts  = df["text"].tolist()
    labels = df["label"].astype(int).tolist()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    loader    = build_dataloader(texts, labels, tokenizer, batch_size=64, shuffle=False)
    model     = load_model(checkpoint)

    preds, probs, targets = get_predictions(model, loader)

    acc    = accuracy_score(targets, preds)
    report = classification_report(targets, preds, target_names=LABELS, digits=4)
    auc    = roc_auc_score(targets, probs, multi_class="ovr", average="macro")

    print(f"\nAccuracy : {acc:.4f}")
    print(f"AUC (OvR): {auc:.4f}")
    print(f"\n{report}")

    # Save metrics JSON
    metrics = {"accuracy": round(acc, 4), "auc_macro_ovr": round(auc, 4)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_confusion_matrix(targets, preds, out)
    error_analysis(texts, targets, preds, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed/test.csv")
    parser.add_argument("--checkpoint", default="checkpoints/bert_sentiment.pt")
    parser.add_argument("--out",        default="results/")
    args = parser.parse_args()
    main(args.data, args.checkpoint, args.out)
