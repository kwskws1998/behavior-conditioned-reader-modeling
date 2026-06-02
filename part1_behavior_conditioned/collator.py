"""Batch collation for token-level TRT regression with reader profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorForBehaviorConditionedTokenRegression:
    tokenizer: Any
    padding: bool | str = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    label_pad_token_id: float = -100.0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        profiles = [feature.pop("reader_profile") for feature in features]
        token_features = [feature.pop("token_features", None) for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            label = list(label)
            padded = label + [self.label_pad_token_id] * (sequence_length - len(label))
            padded_labels.append(padded)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.float32)
        batch["reader_profile"] = torch.tensor(profiles, dtype=torch.float32)
        if any(feature is not None for feature in token_features):
            feature_dim = len(next(feature for feature in token_features if feature is not None)[0])
            padded_features = []
            zero_feature = [0.0] * feature_dim
            for feature in token_features:
                feature = list(feature) if feature is not None else []
                padded = feature + [zero_feature] * (sequence_length - len(feature))
                padded_features.append(padded)
            batch["token_features"] = torch.tensor(padded_features, dtype=torch.float32)
        return batch
