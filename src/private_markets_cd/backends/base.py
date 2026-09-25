"""Backend protocol shared by contribution and distribution models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Mean path plus optional Monte Carlo paths for a cash-flow series."""

    mean: np.ndarray
    paths: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(len(self.mean))


@runtime_checkable
class CashFlowBackend(Protocol):
    """Fit on a densified (or raw) univariate series; forecast a horizon."""

    name: str

    def fit(self, series: pd.Series | pd.DataFrame) -> CashFlowBackend:
        ...

    def forecast(self, horizon: int, n_paths: int = 1) -> ForecastResult:
        ...
