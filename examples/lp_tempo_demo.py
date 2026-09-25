"""Example: LP commitment cashflows + historical pace + optional forecast."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from private_markets_cd import LPCommitment, compare_fund_types


def main() -> None:
    lp = LPCommitment(
        commitment=10_000_000,
        fund_type="buyout",
        size="All",
        data_dir=ROOT / "data",
    )
    print("Overview:", lp.overview())
    cash = lp.cashflows()
    print(cash.head(10).to_string(index=False))
    out = ROOT / "examples" / "buyout_commitment_cashflows.csv"
    cash.to_csv(out, index=False)
    print(f"Wrote {out}")

    print("\nFund-type comparison:")
    print(compare_fund_types(commitment=10_000_000, data_dir=ROOT / "data").to_string(index=False))

    schedule = lp.forecast_tempo(backend="mrsr", n_paths=200, seed=7)
    print("\nForecast tempo summary:", schedule.summary())
    tempo_out = ROOT / "examples" / "buyout_tempo_schedule.csv"
    schedule.to_frame().to_csv(tempo_out, index=False)
    print(f"Wrote {tempo_out}")


if __name__ == "__main__":
    main()
