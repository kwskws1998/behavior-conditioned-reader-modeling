"""Data collator for text plus numeric auxiliary features."""

from __future__ import annotations

import torch


class DataCollatorForTextAuxiliaryClassification:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        aux_features = [feature.pop("aux_features") for feature in features]
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["aux_features"] = torch.tensor(aux_features, dtype=torch.float32)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch
