"""Ornstein–Uhlenbeck / mean-reverting diffusion backend (legacy PE_MRSR)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from private_markets_cd.backends.base import ForecastResult


@dataclass
class MeanRevertingBackend:
    """Least-squares OU fit + Euler–Maruyama Monte Carlo paths."""

    name: str = "mrsr"
    n_sim: int = 1000
    train_frac: float = 0.7
    rng: np.random.Generator | None = None
    _params: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_train: float | None = field(default=None, init=False, repr=False)
    _test: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, series: pd.Series | pd.DataFrame) -> MeanRevertingBackend:
        rng = self.rng or np.random.default_rng()
        self.rng = rng
        frame = series.to_frame() if isinstance(series, pd.Series) else series.iloc[:, [0]].copy()
        frame = frame.dropna()
        cut = math.floor(self.train_frac * len(frame))
        train = frame.iloc[:cut]
        test = frame.iloc[cut:]
        self._test = test.iloc[:, 0].to_numpy(dtype=float)

        x = train.iloc[:, 0].to_numpy(dtype=float)
        y = train.iloc[:, 0].shift(-1).to_numpy(dtype=float)
        mask = ~np.isnan(y)
        x, y = x[mask], y[mask]

        a_num = float(np.sum(x * y) - np.sum(x) * np.sum(y) / len(y))
        a_denom = float(np.sum(x**2) - np.sum(x) ** 2 / len(x))
        a_hat = a_num / a_denom if a_denom != 0 else 0.0
        b_hat = float((np.sum(y) - np.sum(a_hat * x)) / len(x))
        l_hat = 1.0 - a_hat
        m_hat = b_hat / l_hat if l_hat != 0 else float(np.mean(x))
        e = y - (x * (1 - l_hat) + l_hat * m_hat)
        sigma_hat = float(np.std(e))

        self._params = {"m": m_hat, "l": l_hat, "sigma": sigma_hat, "a": a_hat, "b": b_hat}
        self._last_train = float(train.iloc[-1, 0])
        return self

    def forecast(self, horizon: int | None = None, n_paths: int | None = None) -> ForecastResult:
        if not self._params or self._last_train is None:
            raise RuntimeError("Call fit() before forecast().")
        rng = self.rng or np.random.default_rng()
        n_paths = n_paths if n_paths is not None else self.n_sim
        if horizon is None:
            horizon = len(self._test) if self._test is not None else 1

        m_hat = self._params["m"]
        l_hat = self._params["l"]
        sigma = self._params["sigma"]
        paths = np.zeros((n_paths, horizon), dtype=float)
        for k in range(n_paths):
            paths[k, 0] = self._last_train
            for i in range(1, horizon):
                paths[k, i] = (
                    paths[k, i - 1]
                    + l_hat * (m_hat - paths[k, i - 1])
                    + sigma * float(rng.normal())
                )

        mean = paths.mean(axis=0)
        metrics: dict[str, float] = {}
        meta: dict[str, Any] = {"params": dict(self._params)}
        if self._test is not None and len(self._test) == horizon:
            mse = float(np.mean((mean - self._test) ** 2))
            metrics["mse_vs_holdout"] = mse
            meta["holdout"] = self._test
        return ForecastResult(mean=mean, paths=paths, metrics=metrics, meta=meta)
