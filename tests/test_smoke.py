"""Smoke tests for data load and MRSR tempo (no torch required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"


def test_load_buyout_book():
    from private_markets_cd.data.load import load_fund_book
    from private_markets_cd.types import CashFlowKind, FundSize, FundType

    book = load_fund_book(FundType.BUYOUT, data_dir=DATA)
    assert FundSize.ALL in book.levels
    c = book.series(CashFlowKind.CONTRIBUTION, FundSize.ALL)
    d = book.series(CashFlowKind.DISTRIBUTION, FundSize.ALL)
    assert len(c) > 5 and len(d) > 5


def test_tempo_mrsr():
    from private_markets_cd.tempo.schedule import build_tempo_schedule

    schedule = build_tempo_schedule(
        fund_type="buyout",
        size="All",
        backend="mrsr",
        n_paths=50,
        data_dir=DATA,
        seed=1,
    )
    frame = schedule.to_frame()
    assert len(frame) > 10
    assert "dpi_proxy" in frame.columns
    summary = schedule.summary()
    assert summary["horizon"] == len(frame)


def test_backend_registry():
    from private_markets_cd.backends import get_backend, list_backends

    assert "mrsr" in list_backends()
    backend = get_backend("mrsr", n_sim=10)
    assert backend.name == "mrsr"
