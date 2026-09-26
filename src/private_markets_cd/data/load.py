"""Load Preqin-style Excel fund cash-flow benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from private_markets_cd.types import (
    CashFlowKind,
    FundSize,
    FundType,
    default_data_dir,
)

# Workbooks differ: Buyout has All/Large/Mid/Small; VC has All; RE uses region labels.
_SHEET_ALIASES: dict[FundType, dict[FundSize, tuple[str, ...]]] = {
    FundType.BUYOUT: {
        FundSize.ALL: ("All",),
        FundSize.LARGE: ("Large",),
        FundSize.MID: ("Mid",),
        FundSize.SMALL: ("Small",),
    },
    FundType.VC: {
        FundSize.ALL: ("All",),
        FundSize.LARGE: ("Large", "All"),
        FundSize.MID: ("Mid", "All"),
        FundSize.SMALL: ("Small", "All"),
    },
    FundType.REAL_ESTATE: {
        FundSize.ALL: ("All-2000", "All", "North America"),
        FundSize.LARGE: ("Large", "All-2000", "North America"),
        FundSize.MID: ("Mid", "All-2000", "North America"),
        FundSize.SMALL: ("Small", "All-2000", "North America"),
    },
}


@dataclass(frozen=True)
class FundCashFlowBook:
    """Raw quarterly levels for one fund type, keyed by size and cash-flow kind."""

    fund_type: FundType
    path: Path
    levels: Mapping[FundSize, Mapping[CashFlowKind, pd.Series]]
    sheet_used: Mapping[FundSize, str]

    def series(self, kind: CashFlowKind, size: FundSize = FundSize.ALL) -> pd.Series:
        try:
            return self.levels[size][kind]
        except KeyError as exc:
            raise KeyError(
                f"No series for fund_type={self.fund_type} size={size} kind={kind}"
            ) from exc


def _read_sheet_metric(path: Path, sheet: str, metric: str) -> pd.Series:
    frame = pd.read_excel(
        path,
        sheet_name=sheet,
        header=2,
        usecols="A:G",
        index_col=0,
    )
    if metric not in frame.columns:
        raise KeyError(
            f"Column {metric!r} missing in {path.name} sheet {sheet!r}; have {list(frame.columns)}"
        )
    series = frame[metric].astype(float)
    series.name = metric
    return series


def _resolve_sheet(path: Path, fund_type: FundType, size: FundSize, available: list[str]) -> str | None:
    candidates = _SHEET_ALIASES.get(fund_type, {}).get(size, (size.value,))
    for name in candidates:
        if name in available:
            return name
    return None


def load_fund_book(
    fund_type: FundType | str = FundType.BUYOUT,
    data_dir: Path | str | None = None,
) -> FundCashFlowBook:
    """Load size sheets and both C/D metrics for a fund type workbook."""
    fund_type = FundType(fund_type) if not isinstance(fund_type, FundType) else fund_type
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    path = root / fund_type.data_filename
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark workbook not found: {path}")

    xl = pd.ExcelFile(path)
    available = list(xl.sheet_names)

    levels: dict[FundSize, dict[CashFlowKind, pd.Series]] = {}
    sheet_used: dict[FundSize, str] = {}
    seen_sheets: set[str] = set()

    for size in FundSize:
        sheet = _resolve_sheet(path, fund_type, size, available)
        if sheet is None:
            continue
        # Avoid duplicating the same physical sheet under every size alias.
        if sheet in seen_sheets and size is not FundSize.ALL:
            # Still allow ALL; skip other sizes that fell back to the same sheet.
            if fund_type is not FundType.BUYOUT:
                continue
        by_kind: dict[CashFlowKind, pd.Series] = {}
        for kind in CashFlowKind:
            by_kind[kind] = _read_sheet_metric(path, sheet, kind.value)
        levels[size] = by_kind
        sheet_used[size] = sheet
        seen_sheets.add(sheet)

    if not levels:
        raise ValueError(f"No readable size sheets in {path}; sheets={available}")

    return FundCashFlowBook(
        fund_type=fund_type, path=path, levels=levels, sheet_used=sheet_used
    )


def load_series(
    fund_type: FundType | str = FundType.BUYOUT,
    kind: CashFlowKind | str = CashFlowKind.CONTRIBUTION,
    size: FundSize | str = FundSize.ALL,
    data_dir: Path | str | None = None,
) -> pd.Series:
    """Convenience: load a single quarterly level series."""
    kind = CashFlowKind(kind) if not isinstance(kind, CashFlowKind) else kind
    size = FundSize(size) if not isinstance(size, FundSize) else size
    book = load_fund_book(fund_type, data_dir=data_dir)
    if size not in book.levels:
        # Fall back to ALL when a size sheet is absent (e.g. VC).
        size = FundSize.ALL
    return book.series(kind, size)
