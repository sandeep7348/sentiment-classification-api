# tests/test_model.py

import pytest
import torch
from transformers import BertTokenizer
from model import BertSentimentClassifier, SentimentDataset, build_dataloader


@pytest.fixture(scope="module")
def tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def model():
    return BertSentimentClassifier(num_classes=3)


def test_model_output_shape(model, tokenizer):
    texts  = ["This is great!", "Terrible experience."]
    labels = [2, 0]
    loader = build_dataloader(texts, labels, tokenizer, batch_size=2, shuffle=False)
    batch  = next(iter(loader))
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"])
    assert logits.shape == (2, 3), f"Expected (2, 3), got {logits.shape}"


def test_dataset_length(tokenizer):
    texts  = ["hello world", "bad product", "okay I guess"]
    labels = [2, 0, 1]
    ds     = SentimentDataset(texts, labels, tokenizer)
    assert len(ds) == 3


def test_dataset_item_keys(tokenizer):
    ds   = SentimentDataset(["sample text"], [1], tokenizer)
    item = ds[0]
    assert "input_ids"      in item
    assert "attention_mask" in item
    assert "label"          in item


def test_predict_proba_sums_to_one(model, tokenizer):
    ds     = SentimentDataset(["I love this!"], [2], tokenizer)
    item   = ds[0]
    ids    = item["input_ids"].unsqueeze(0)
    mask   = item["attention_mask"].unsqueeze(0)
    probs  = model.predict_proba(ids, mask)
    total  = probs.sum().item()
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}"
