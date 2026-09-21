import torch
import torch.nn as nn


class TriBFNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        cnn_channels=128,
        lstm_hidden=128,
        gru_hidden=128,
        num_classes=5,
        dropout=0.3
    ):
        super().__init__()

        # ==========================================
        # 1. Embedding Layer
        # ==========================================

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # ==========================================
        # 2. CNN Layer
        # ==========================================

        self.cnn = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )

        self.batch_norm = nn.BatchNorm1d(cnn_channels)

        self.relu = nn.ReLU()

        # ==========================================
        # 3. Bi-LSTM
        # ==========================================

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # ==========================================
        # 4. GRU
        # ==========================================

        self.gru = nn.GRU(
            input_size=lstm_hidden * 2,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True
        )

        # ==========================================
        # 5. Fully Connected Layers
        # ==========================================

        self.fc1 = nn.Linear(
            gru_hidden,
            128
        )

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(
            128,
            num_classes
        )

    def forward(self, input_ids):

        # ==========================================
        # Embedding
        # ==========================================

        x = self.embedding(input_ids)

        # Shape:
        # [batch, sequence, embedding_dim]

        # ==========================================
        # CNN
        # ==========================================

        x = x.permute(0, 2, 1)

        x = self.cnn(x)

        x = self.batch_norm(x)

        x = self.relu(x)

        # Back to:
        # [batch, sequence, channels]

        x = x.permute(0, 2, 1)

        # ==========================================
        # Bi-LSTM
        # ==========================================

        x, _ = self.bilstm(x)

        # ==========================================
        # GRU
        # ==========================================

        x, _ = self.gru(x)

        # ==========================================
        # Global Max Pooling
        # ==========================================

        x = torch.max(x, dim=1).values

        # ==========================================
        # Fully Connected
        # ==========================================

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x
