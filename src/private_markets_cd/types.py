"""Shared domain types for LP contribution / distribution modeling."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class CashFlowKind(str, Enum):
    """LP cash-flow side of the commitment."""

    CONTRIBUTION = "Called Up"  # capital calls
    DISTRIBUTION = "Distributed"

    @property
    def short(self) -> str:
        return "C" if self is CashFlowKind.CONTRIBUTION else "D"


class FundType(str, Enum):
    BUYOUT = "buyout"
    VC = "vc"
    REAL_ESTATE = "real_estate"

    @property
    def data_filename(self) -> str:
        return {
            FundType.BUYOUT: "Buyout_Funds_Stats_2014.xlsx",
            FundType.VC: "VC_Funds_Stats_2014.xlsx",
            FundType.REAL_ESTATE: "RE_Stats_benchmarks_2014.xlsx",
        }[self]


class FundSize(str, Enum):
    """Benchmark size bucket (sheet name in Preqin-style workbooks)."""

    ALL = "All"
    LARGE = "Large"
    MID = "Mid"
    SMALL = "Small"

    @property
    def index(self) -> int:
        return {
            FundSize.ALL: 0,
            FundSize.LARGE: 1,
            FundSize.MID: 2,
            FundSize.SMALL: 3,
        }[self]


def default_data_dir() -> Path:
    """Repo `data/` directory (works from installed editable or source tree)."""
    # src/private_markets_cd/types.py -> repo root / data
    here = Path(__file__).resolve()
    repo_data = here.parents[2] / "data"
    if repo_data.is_dir():
        return repo_data
    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        return cwd_data
    return repo_data
