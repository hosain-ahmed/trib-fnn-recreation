"""Standalone Day 3 baseline classifiers.

Each classifier follows the same contract: token IDs and an optional attention
mask go in, and unnormalised class logits come out.  CrossEntropyLoss applies
the numerically stable softmax internally during training.
"""

import torch
import torch.nn as nn


def _masked_max(sequence: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Max-pool a [batch, sequence, features] tensor while ignoring padding."""
    if attention_mask is None:
        return sequence.max(dim=1).values

    mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1)
    sequence = sequence.masked_fill(~mask, torch.finfo(sequence.dtype).min)
    return sequence.max(dim=1).values


class CNNClassifier(nn.Module):
    """Embedding -> Conv1D -> BatchNorm -> ReLU -> max-pool -> classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        channels: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(embedding_dim, channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(input_ids).transpose(1, 2)
        x = self.activation(self.batch_norm(self.conv(x)))
        if attention_mask is not None:
            x = x.masked_fill(~attention_mask.to(dtype=torch.bool).unsqueeze(1), torch.finfo(x.dtype).min)
        x = x.max(dim=2).values
        return self.classifier(self.dropout(x))


class BiLSTMClassifier(nn.Module):
    """Embedding -> bidirectional LSTM -> masked max-pool -> classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x, _ = self.bilstm(self.embedding(input_ids))
        x = _masked_max(x, attention_mask)
        return self.classifier(self.dropout(x))


class GRUClassifier(nn.Module):
    """Embedding -> GRU -> masked max-pool -> classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x, _ = self.gru(self.embedding(input_ids))
        x = _masked_max(x, attention_mask)
        return self.classifier(self.dropout(x))


BASELINE_MODELS = {
    "cnn": CNNClassifier,
    "bilstm": BiLSTMClassifier,
    "gru": GRUClassifier,
}
