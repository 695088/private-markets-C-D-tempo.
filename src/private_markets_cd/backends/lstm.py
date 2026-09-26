"""Optional PyTorch LSTM backend (legacy PE_LSTM_V5), loaded lazily."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from private_markets_cd.backends.base import ForecastResult
from private_markets_cd.data.prep import prepare_supervised


@dataclass
class LSTMBackend:
    """Sequence LSTM on supervised windows. Requires ``torch`` at fit time."""

    name: str = "lstm"
    n_step: int = 64
    n_ahead: int = 8
    hidden_size: int = 64
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 0.005
    train_frac: float = 0.7
    num_layers: int = 1
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)
    _x_test: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _y_test: pd.Series | None = field(default=None, init=False, repr=False)
    _last_window: np.ndarray | None = field(default=None, init=False, repr=False)

    def _require_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "LSTM backend requires PyTorch. Install with: pip install torch"
            ) from exc
        self._torch = (torch, nn)
        return self._torch

    def fit(self, series: pd.Series | pd.DataFrame) -> LSTMBackend:
        torch, nn = self._require_torch()
        frame = series.to_frame() if isinstance(series, pd.Series) else series.iloc[:, [0]].copy()
        # Use shorter windows than the academic default so demos finish quickly.
        n_step = min(self.n_step, max(8, len(frame) // 4))
        n_ahead = min(self.n_ahead, max(1, len(frame) // 20))
        self.n_step, self.n_ahead = n_step, n_ahead

        supervised = prepare_supervised(frame, n_step=n_step, n_ahead=n_ahead)
        if supervised.empty:
            raise ValueError("Not enough observations to build supervised LSTM windows.")

        cut = math.floor(self.train_frac * len(supervised))
        x_train = supervised.iloc[:cut, :n_step]
        y_train = supervised.iloc[:cut]["target"]
        x_test = supervised.iloc[cut:, :n_step]
        y_test = supervised.iloc[cut:]["target"]
        self._x_test, self._y_test = x_test, y_test
        self._last_window = supervised.iloc[cut - 1, :n_step].to_numpy(dtype=float) if cut else (
            supervised.iloc[-1, :n_step].to_numpy(dtype=float)
        )

        class _LSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
                self.linear = nn.Linear(hidden_dim, output_dim)

            def forward(self, data_in):
                # data_in: (batch, seq)
                row, col = data_in.shape
                data_in = data_in.view(col, row, 1)
                lstm_out, _ = self.lstm(data_in)
                y_pred = self.linear(lstm_out[-1].view(row, -1))
                return y_pred.view(-1)

        model = _LSTM(1, self.hidden_size, num_layers=self.num_layers)
        loss_fn = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        norm = nn.BatchNorm1d(n_step)

        model.train()
        for _epoch in range(self.num_epochs):
            for i in range(0, len(x_train), self.batch_size):
                xb = torch.tensor(x_train.iloc[i : i + self.batch_size].values, dtype=torch.float32)
                yb = torch.tensor(y_train.iloc[i : i + self.batch_size].values, dtype=torch.float32)
                if len(xb) < 2:
                    continue
                xb = norm(xb)
                optimiser.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimiser.step()

        self._model = (model, norm, loss_fn)
        return self

    def forecast(self, horizon: int | None = None, n_paths: int = 1) -> ForecastResult:
        torch, _nn = self._require_torch()
        if self._model is None or self._last_window is None:
            raise RuntimeError("Call fit() before forecast().")
        model, norm, loss_fn = self._model
        model.eval()

        metrics: dict[str, float] = {}
        meta: dict[str, Any] = {"n_step": self.n_step, "n_ahead": self.n_ahead}

        # Evaluate on holdout windows when available
        if self._x_test is not None and len(self._x_test) > 0:
            preds = []
            with torch.no_grad():
                for i in range(0, len(self._x_test), self.batch_size):
                    xb = torch.tensor(
                        self._x_test.iloc[i : i + self.batch_size].values, dtype=torch.float32
                    )
                    if len(xb) < 2:
                        # BatchNorm needs >1; fall back without norm for tiny tail
                        if len(xb) == 0:
                            continue
                        pred = model(xb)
                    else:
                        pred = model(norm(xb))
                    preds.append(pred.numpy())
            if preds:
                y_hat = np.concatenate(preds)
                y_true = self._y_test.to_numpy(dtype=float)[: len(y_hat)]
                metrics["mse_vs_holdout"] = float(np.mean((y_hat - y_true) ** 2))
                mean = y_hat
            else:
                mean = np.array([float(self._last_window[-1])])
        else:
            mean = np.array([float(self._last_window[-1])])

        if horizon is not None and horizon != len(mean):
            # Pad / truncate to requested horizon using last prediction / level
            if horizon <= len(mean):
                mean = mean[:horizon]
            else:
                pad = np.full(horizon - len(mean), mean[-1] if len(mean) else 0.0)
                mean = np.concatenate([mean, pad])

        paths = np.tile(mean, (max(n_paths, 1), 1))
        return ForecastResult(mean=mean, paths=paths, metrics=metrics, meta=meta)
