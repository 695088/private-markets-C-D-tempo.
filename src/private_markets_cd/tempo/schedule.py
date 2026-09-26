"""Combine contribution and distribution forecasts into an LP tempo schedule."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from private_markets_cd.backends.base import ForecastResult
from private_markets_cd.contributions.models import ContributionModel
from private_markets_cd.distributions.models import DistributionModel
from private_markets_cd.types import FundSize, FundType


@dataclass
class TempoSchedule:
    """Aligned C/D paths plus derived LP tempo metrics."""

    fund_type: FundType
    size: FundSize
    backend: str
    contributions: np.ndarray
    distributions: np.ndarray
    contribution_result: ForecastResult
    distribution_result: ForecastResult

    def to_frame(self) -> pd.DataFrame:
        n = min(len(self.contributions), len(self.distributions))
        c = self.contributions[:n]
        d = self.distributions[:n]
        # Series are log-diffs of rates; for tempo we treat forecast paths as
        # relative flow intensity and accumulate for LP pacing views.
        cum_c = np.cumsum(np.abs(c))
        cum_d = np.cumsum(np.abs(d))
        net = d - c
        cum_net = np.cumsum(net)
        # Avoid div-by-zero for early periods
        dpi_proxy = np.divide(cum_d, np.maximum(cum_c, 1e-12))
        return pd.DataFrame(
            {
                "step": np.arange(n),
                "contribution_intensity": c,
                "distribution_intensity": d,
                "net_intensity": net,
                "cum_contribution": cum_c,
                "cum_distribution": cum_d,
                "cum_net": cum_net,
                "dpi_proxy": dpi_proxy,
            }
        )

    def summary(self) -> dict[str, float]:
        frame = self.to_frame()
        return {
            "horizon": float(len(frame)),
            "mean_contribution_intensity": float(frame["contribution_intensity"].mean()),
            "mean_distribution_intensity": float(frame["distribution_intensity"].mean()),
            "final_dpi_proxy": float(frame["dpi_proxy"].iloc[-1]) if len(frame) else float("nan"),
            "contribution_holdout_mse": float(
                self.contribution_result.metrics.get("mse_vs_holdout", float("nan"))
            ),
            "distribution_holdout_mse": float(
                self.distribution_result.metrics.get("mse_vs_holdout", float("nan"))
            ),
        }


def build_tempo_schedule(
    fund_type: FundType | str = FundType.BUYOUT,
    size: FundSize | str = FundSize.ALL,
    backend: str = "mrsr",
    n_paths: int = 500,
    horizon: int | None = None,
    data_dir: Path | str | None = None,
    seed: int | None = 42,
) -> TempoSchedule:
    """Fit C and D models and align forecasts into one LP tempo schedule."""
    ft = FundType(fund_type) if not isinstance(fund_type, FundType) else fund_type
    sz = FundSize(size) if not isinstance(size, FundSize) else size

    c_model = ContributionModel(
        fund_type=ft, size=sz, backend_name=backend, data_dir=data_dir, seed=seed
    )
    d_model = DistributionModel(
        fund_type=ft, size=sz, backend_name=backend, data_dir=data_dir, seed=None if seed is None else seed + 1
    )

    c_res = c_model.fit_forecast(horizon=horizon, n_paths=n_paths)
    d_res = d_model.fit_forecast(horizon=horizon, n_paths=n_paths)

    n = min(len(c_res.mean), len(d_res.mean))
    return TempoSchedule(
        fund_type=ft,
        size=sz,
        backend=backend,
        contributions=c_res.mean[:n],
        distributions=d_res.mean[:n],
        contribution_result=c_res,
        distribution_result=d_res,
    )
