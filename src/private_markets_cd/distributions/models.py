"""Distribution (Distributed) forecasts for LPs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from private_markets_cd.backends.base import CashFlowBackend, ForecastResult
from private_markets_cd.backends.registry import get_backend
from private_markets_cd.data.load import load_series
from private_markets_cd.data.prep import densify_log_diffs
from private_markets_cd.types import CashFlowKind, FundSize, FundType


@dataclass
class DistributionModel:
    """LP-facing wrapper: load distribution series → prep → backend forecast."""

    fund_type: FundType = FundType.BUYOUT
    size: FundSize = FundSize.ALL
    backend_name: str = "mrsr"
    steps_between: int = 90
    data_dir: Path | str | None = None
    seed: int | None = 42

    def load_levels(self) -> pd.Series:
        return load_series(
            self.fund_type,
            CashFlowKind.DISTRIBUTION,
            self.size,
            data_dir=self.data_dir,
        )

    def prepare(self, levels: pd.Series | None = None) -> pd.DataFrame:
        import numpy as np

        levels = levels if levels is not None else self.load_levels()
        rng = np.random.default_rng(self.seed)
        return densify_log_diffs(levels, steps_between=self.steps_between, rng=rng)

    def fit_forecast(
        self,
        horizon: int | None = None,
        n_paths: int = 500,
        backend: CashFlowBackend | None = None,
    ) -> ForecastResult:
        prepared = self.prepare()
        model = backend or get_backend(self.backend_name)
        model.fit(prepared)
        return model.forecast(horizon=horizon, n_paths=n_paths)


def forecast_distributions(
    fund_type: FundType | str = FundType.BUYOUT,
    size: FundSize | str = FundSize.ALL,
    backend: str = "mrsr",
    **kwargs,
) -> ForecastResult:
    ft = FundType(fund_type) if not isinstance(fund_type, FundType) else fund_type
    sz = FundSize(size) if not isinstance(size, FundSize) else size
    return DistributionModel(fund_type=ft, size=sz, backend_name=backend).fit_forecast(**kwargs)
