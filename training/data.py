"""Triplet dataset and batch collation for numeracy contrastive training."""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


class TripletDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 128):
        with open(path) as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text: str):
        return self.tokenizer(text, truncation=True, max_length=self.max_length)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "anchor": self._tokenize(item["comment"]),
            # rewritten fields when the LLM pass has run, plain substituted otherwise
            "positive": self._tokenize(item.get("positive_rewritten") or item["positive"]),
            "negative": self._tokenize(item.get("negative_rewritten") or item["negative"]),
            "positive_number": float(item["positive_number"]),
            "negative_number": float(item["negative_number"]),
            "anchor_number": float(item["number"]),
        }


def make_triplet_collator(tokenizer):
    base_collator = DataCollatorWithPadding(tokenizer)

    def collate(batch):
        return {
            "anchor": base_collator([b["anchor"] for b in batch]),
            "positive": base_collator([b["positive"] for b in batch]),
            "negative": base_collator([b["negative"] for b in batch]),
            "positive_number": torch.tensor(
                [b["positive_number"] for b in batch], dtype=torch.float
            ),
            "negative_number": torch.tensor(
                [b["negative_number"] for b in batch], dtype=torch.float
            ),
            "anchor_number": torch.tensor(
                [b["anchor_number"] for b in batch], dtype=torch.float
            ),
        }

    return collate
