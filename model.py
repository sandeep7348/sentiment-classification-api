# sentiment_analysis/model.py — BERT-based sentiment classifier

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification."""

    LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer: BertTokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BertSentimentClassifier(nn.Module):
    """Fine-tuned BERT for 3-class sentiment (negative / neutral / positive)."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(768, num_classes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        out  = self.bert(input_ids, attention_mask=attention_mask)
        pool = out.pooler_output           # [CLS] token embedding — shape (B, 768)
        drop = self.dropout(pool)
        return self.fc(drop)               # logits — shape (B, num_classes)

    def predict_proba(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (inference helper)."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=1)


def build_dataloader(texts: List[str], labels: List[int],
                     tokenizer: BertTokenizer, batch_size: int = 32,
                     shuffle: bool = True, max_len: int = 128) -> DataLoader:
    dataset = SentimentDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=4, pin_memory=True)
