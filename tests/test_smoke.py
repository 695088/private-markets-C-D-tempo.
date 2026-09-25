"""Smoke / unit tests for LP C/D tempo framework."""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_load_vc_and_re():
    from private_markets_cd.data.load import load_fund_book
    from private_markets_cd.types import FundType

    vc = load_fund_book(FundType.VC, data_dir=DATA)
    re_ = load_fund_book(FundType.REAL_ESTATE, data_dir=DATA)
    assert len(vc.levels) >= 1
    assert len(re_.levels) >= 1


def test_pace_curve_buyout():
    from private_markets_cd.tempo.pace import build_pace_curve

    pace = build_pace_curve("buyout", "All", data_dir=DATA)
    assert pace.summary()["n_quarters"] > 5
    assert "dpi" in pace.frame.columns
    assert pace.frame["pct_called"].iloc[-1] > pace.frame["pct_called"].iloc[0]


def test_lp_commitment_cashflows():
    from private_markets_cd import LPCommitment

    lp = LPCommitment(commitment=5_000_000, fund_type="buyout", size="All", data_dir=DATA)
    cash = lp.cashflows()
    overview = lp.overview()
    assert abs(cash["contribution"].sum() - overview["lifetime_contributions"]) < 1e-6
    assert overview["ending_dpi"] > 0
    assert overview["lifetime_contributions"] > 0


def test_compare_fund_types():
    from private_markets_cd import compare_fund_types

    frame = compare_fund_types(commitment=1_000_000, data_dir=DATA)
    assert set(frame["fund_type"]) >= {"buyout", "vc", "real_estate"}
    assert frame["dpi"].notna().any()


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
