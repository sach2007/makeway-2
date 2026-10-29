"""PyTorch-based GRU / Sequence Classifier for sequential market feature patterns.

Leverages NVIDIA GPU if available via torch.cuda.is_available().
Trains on rolling sequence windows (e.g., 60 bars) with zero lookahead.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SmallGRU(nn.Module):
    """2-layer GRU sequence classifier with dropout and residual head."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # Use last step representation
        last_step = out[:, -1, :]
        logits = self.fc(last_step).squeeze(-1)
        return logits


class SequenceModelWrapper:
    """End-to-end wrapper for rolling sequence feature data, training, and inference."""

    def __init__(self, seq_len: int = 60, hidden_size: int = 64, epochs: int = 15, lr: float = 1e-3, batch_size: int = 64):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[SmallGRU] = None
        self.is_fitted = False
        self.input_size = 0

    def create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Slice 2D matrix (samples, features) into rolling 3D windows (samples - seq_len + 1, seq_len, features)."""
        n_samples, n_features = X.shape
        if n_samples < self.seq_len:
            raise ValueError(f"Insufficient samples ({n_samples}) for sequence length ({self.seq_len})")

        num_seqs = n_samples - self.seq_len + 1
        X_seq = np.zeros((num_seqs, self.seq_len, n_features), dtype=np.float32)

        for i in range(num_seqs):
            X_seq[i] = X[i : i + self.seq_len]

        if y is not None:
            y_seq = y[self.seq_len - 1 :].astype(np.float32)
            return X_seq, y_seq
        return X_seq, None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SequenceModelWrapper:
        """Fit scaler on training features, create past-only sequence windows, and train GRU on GPU/CPU."""
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        self.input_size = X_seq.shape[2]

        # Init model on target device
        self.model = SmallGRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.2
        ).to(self.device)

        dataset = TensorDataset(
            torch.from_numpy(X_seq).float(),
            torch.from_numpy(y_seq).float()
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of upward movement for each bar in X (padded with 0.5 for initial warm-up bars)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Sequence model is not fitted yet.")

        n_samples = len(X)
        if n_samples < self.seq_len:
            return np.full(n_samples, 0.5, dtype=float)

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.create_sequences(X_scaled)

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_seq).float().to(self.device)
            logits = self.model(tensor_x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # Prepend 0.5 for warm-up bars where full sequence history wasn't available yet
        pad_len = self.seq_len - 1
        full_probs = np.concatenate([np.full(pad_len, 0.5, dtype=float), probs])
        return full_probs
