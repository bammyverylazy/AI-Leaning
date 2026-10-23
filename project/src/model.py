"""โมเดลลำดับ: Linear -> BiGRU/BiLSTM -> mean+max pooling -> classifier"""
import torch
import torch.nn as nn


class SeqClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=192, layers=2, dropout=0.3, kind="gru"):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            hidden, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0, bidirectional=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, n_classes))

    def forward(self, x):  # x [B, T, F]
        h, _ = self.rnn(self.proj(x))  # [B, T, 2H]
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        return self.head(pooled)
