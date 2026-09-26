"""Historical LP pacing from cumulative Called Up / Distributed levels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from private_markets_cd.data.load import load_series
from private_markets_cd.types import CashFlowKind, FundSize, FundType


@dataclass(frozen=True)
class PaceCurve:
    """Quarterly cumulative contribution / distribution pace for an LP view."""

    fund_type: FundType
    size: FundSize
    frame: pd.DataFrame

    def summary(self) -> dict[str, float | int | str]:
        f = self.frame
        last = f.iloc[-1] if len(f) else None
        return {
            "fund_type": self.fund_type.value,
            "size": self.size.value,
            "n_quarters": int(len(f)),
            "pct_called": float(last["pct_called"]) if last is not None else float("nan"),
            "pct_distributed": float(last["pct_distributed"]) if last is not None else float("nan"),
            "dpi": float(last["dpi"]) if last is not None else float("nan"),
            "unfunded_pct": float(last["unfunded_pct"]) if last is not None else float("nan"),
            "peak_quarterly_call_pct": float(f["call_pct"].max()) if len(f) else float("nan"),
            "peak_quarterly_dist_pct": float(f["dist_pct"].max()) if len(f) else float("nan"),
        }


def build_pace_curve(
    fund_type: FundType | str = FundType.BUYOUT,
    size: FundSize | str = FundSize.ALL,
    data_dir: Path | str | None = None,
    drop_leading_zeros: bool = True,
) -> PaceCurve:
    """Build LP tempo from benchmark cumulative % Called Up / Distributed.

    Preqin-style sheets store cumulative percentages of committed capital.
    Period increments are first differences of those cumulative series.
    """
    ft = FundType(fund_type) if not isinstance(fund_type, FundType) else fund_type
    sz = FundSize(size) if not isinstance(size, FundSize) else size

    called = load_series(ft, CashFlowKind.CONTRIBUTION, sz, data_dir=data_dir).astype(float)
    dist = load_series(ft, CashFlowKind.DISTRIBUTION, sz, data_dir=data_dir).astype(float)

    frame = pd.DataFrame({"pct_called": called, "pct_distributed": dist})
    frame = frame.dropna(how="all")
    if drop_leading_zeros:
        nonzero = (frame.fillna(0) != 0).any(axis=1)
        if nonzero.any():
            frame = frame.loc[nonzero.idxmax() :]

    frame = frame.reset_index(drop=True)
    frame.insert(0, "quarter", np.arange(1, len(frame) + 1))
    frame["call_pct"] = frame["pct_called"].diff().fillna(frame["pct_called"])
    frame["dist_pct"] = frame["pct_distributed"].diff().fillna(frame["pct_distributed"])
    frame["net_pct"] = frame["dist_pct"] - frame["call_pct"]
    frame["cum_net_pct"] = frame["pct_distributed"] - frame["pct_called"]
    frame["unfunded_pct"] = (100.0 - frame["pct_called"]).clip(lower=0.0)
    frame["dpi"] = np.where(
        frame["pct_called"] > 1e-9,
        frame["pct_distributed"] / frame["pct_called"],
        np.nan,
    )
    # Simple TVPI proxy when only C/D percentages are known (no NAV): DPI only.
    frame["tvpi_proxy"] = frame["dpi"]
    return PaceCurve(fund_type=ft, size=sz, frame=frame)


def scale_pace_to_commitment(pace: PaceCurve, commitment: float) -> pd.DataFrame:
    """Convert percentage pace into currency cash flows for a commitment size."""
    if commitment <= 0:
        raise ValueError("commitment must be positive")
    f = pace.frame.copy()
    scale = commitment / 100.0
    f["contribution"] = f["call_pct"] * scale
    f["distribution"] = f["dist_pct"] * scale
    f["net_cash"] = f["distribution"] - f["contribution"]
    f["cum_contribution"] = f["pct_called"] * scale
    f["cum_distribution"] = f["pct_distributed"] * scale
    f["unfunded"] = f["unfunded_pct"] * scale
    f["commitment"] = commitment
    return f
