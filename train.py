# sentiment_analysis/train.py — training & validation loop

import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from model import BertSentimentClassifier, build_dataloader
from preprocess import batch_clean

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 7
BATCH_SIZE = 32
LR         = 2e-5
MAX_LEN    = 128
WARMUP_PCT = 0.1
PATIENCE   = 3          # early-stopping patience
CHECKPOINT = Path("checkpoints/bert_sentiment.pt")
LABELS     = ["negative", "neutral", "positive"]


# ── Data loading ───────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    assert {"text", "label"}.issubset(df.columns), "CSV must have 'text' and 'label' columns"
    df["text"]  = batch_clean(df["text"].tolist())
    df["label"] = df["label"].map({l: i for i, l in enumerate(LABELS)})
    df = df.dropna(subset=["label"])
    return df["text"].tolist(), df["label"].astype(int).tolist()


# ── Class weights for imbalanced data ─────────────────────────
def get_class_weights(labels):
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(weights, dtype=torch.float)


# ── Single training epoch ──────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []

    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss  += loss.item()
        all_preds   += logits.argmax(1).cpu().tolist()
        all_targets += lbls.cpu().tolist()

    f1 = f1_score(all_targets, all_preds, average="macro")
    return total_loss / len(loader), f1


# ── Validation ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["label"].to(DEVICE)

        logits      = model(ids, mask)
        total_loss += criterion(logits, lbls).item()
        all_preds   += logits.argmax(1).cpu().tolist()
        all_targets += lbls.cpu().tolist()

    f1     = f1_score(all_targets, all_preds, average="macro")
    report = classification_report(all_targets, all_preds,
                                   target_names=LABELS, digits=4)
    return total_loss / len(loader), f1, report


# ── Main ───────────────────────────────────────────────────────
def main(data_path: str):
    log.info(f"Device: {DEVICE}")
    CHECKPOINT.parent.mkdir(exist_ok=True)

    texts, labels = load_data(data_path)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    log.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    tokenizer    = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader = build_dataloader(X_train, y_train, tokenizer, BATCH_SIZE, shuffle=True)
    val_loader   = build_dataloader(X_val,   y_val,   tokenizer, BATCH_SIZE, shuffle=False)
    test_loader  = build_dataloader(X_test,  y_test,  tokenizer, BATCH_SIZE, shuffle=False)

    model     = BertSentimentClassifier().to(DEVICE)
    weights   = get_class_weights(y_train).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_PCT),
        num_training_steps=total_steps,
    )

    best_val_f1, patience_ctr = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        vl_loss, vl_f1, _ = evaluate(model, val_loader, criterion)
        log.info(f"Epoch {epoch:02d} | "
                 f"train loss={tr_loss:.4f} f1={tr_f1:.4f} | "
                 f"val loss={vl_loss:.4f} f1={vl_f1:.4f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            patience_ctr = 0
            torch.save(model.state_dict(), CHECKPOINT)
            log.info(f"  ✓ Saved checkpoint (val F1={best_val_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info("Early stopping triggered.")
                break

    # Final test evaluation
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    _, test_f1, report = evaluate(model, test_loader, criterion)
    log.info(f"\nTest F1 (macro): {test_f1:.4f}\n{report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.csv",
                        help="Path to preprocessed CSV with 'text' and 'label' columns")
    args = parser.parse_args()
    main(args.data)
