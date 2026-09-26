"""LP-facing commitment API: historical pace + optional forecast backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from private_markets_cd.tempo.pace import PaceCurve, build_pace_curve, scale_pace_to_commitment
from private_markets_cd.tempo.schedule import TempoSchedule, build_tempo_schedule
from private_markets_cd.types import FundSize, FundType


@dataclass
class LPCommitment:
    """Model an LP commitment against Buyout / VC / RE C-D benchmarks.

    Primary surface for LPs:
    - ``historical_pace`` / ``cashflows`` — observed Called Up vs Distributed tempo
    - ``forecast_tempo`` — pluggable intensity forecast (MRSR / LSTM)
    - ``overview`` — one JSON-friendly snapshot for dashboards / notebooks
    """

    commitment: float = 10_000_000.0
    fund_type: FundType | str = FundType.BUYOUT
    size: FundSize | str = FundSize.ALL
    data_dir: Path | str | None = None

    def __post_init__(self) -> None:
        self.fund_type = (
            FundType(self.fund_type) if not isinstance(self.fund_type, FundType) else self.fund_type
        )
        self.size = FundSize(self.size) if not isinstance(self.size, FundSize) else self.size
        if self.commitment <= 0:
            raise ValueError("commitment must be positive")

    def historical_pace(self) -> PaceCurve:
        return build_pace_curve(self.fund_type, self.size, data_dir=self.data_dir)

    def cashflows(self) -> pd.DataFrame:
        """Currency cash-flow schedule scaled to this commitment."""
        return scale_pace_to_commitment(self.historical_pace(), self.commitment)

    def forecast_tempo(
        self,
        backend: str = "mrsr",
        n_paths: int = 300,
        seed: int | None = 42,
    ) -> TempoSchedule:
        """Optional model-backed C/D intensity forecast (not level percentages)."""
        return build_tempo_schedule(
            fund_type=self.fund_type,
            size=self.size,
            backend=backend,
            n_paths=n_paths,
            data_dir=self.data_dir,
            seed=seed,
        )

    def overview(self, include_forecast: bool = False, backend: str = "mrsr") -> dict[str, Any]:
        pace = self.historical_pace()
        cash = self.cashflows()
        out: dict[str, Any] = {
            "commitment": self.commitment,
            "fund_type": self.fund_type.value if isinstance(self.fund_type, FundType) else self.fund_type,
            "size": self.size.value if isinstance(self.size, FundSize) else self.size,
            "pace_summary": pace.summary(),
            "lifetime_contributions": float(cash["contribution"].sum()),
            "lifetime_distributions": float(cash["distribution"].sum()),
            "lifetime_net_cash": float(cash["net_cash"].sum()),
            "ending_unfunded": float(cash["unfunded"].iloc[-1]) if len(cash) else float("nan"),
            "ending_dpi": float(cash["dpi"].iloc[-1]) if len(cash) else float("nan"),
        }
        if include_forecast:
            sched = self.forecast_tempo(backend=backend)
            out["forecast_backend"] = backend
            out["forecast_summary"] = sched.summary()
        return out


def compare_fund_types(
    commitment: float = 10_000_000.0,
    size: FundSize | str = FundSize.ALL,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Side-by-side LP pace summary for Buyout / VC / Real Estate."""
    rows = []
    for ft in FundType:
        try:
            lp = LPCommitment(commitment=commitment, fund_type=ft, size=size, data_dir=data_dir)
            overview = lp.overview()
            rows.append(
                {
                    "fund_type": overview["fund_type"],
                    "size": overview["size"],
                    "n_quarters": overview["pace_summary"]["n_quarters"],
                    "pct_called": overview["pace_summary"]["pct_called"],
                    "pct_distributed": overview["pace_summary"]["pct_distributed"],
                    "dpi": overview["ending_dpi"],
                    "lifetime_net_cash": overview["lifetime_net_cash"],
                    "ending_unfunded": overview["ending_unfunded"],
                }
            )
        except Exception as exc:  # noqa: BLE001 — comparison should continue
            rows.append({"fund_type": ft.value, "size": str(size), "error": str(exc)})
    return pd.DataFrame(rows)
