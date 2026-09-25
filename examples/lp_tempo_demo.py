"""Example: LP tempo schedule for a buyout commitment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from private_markets_cd.tempo.schedule import build_tempo_schedule


def main() -> None:
    schedule = build_tempo_schedule(
        fund_type="buyout",
        size="All",
        backend="mrsr",
        n_paths=200,
        data_dir=ROOT / "data",
        seed=7,
    )
    frame = schedule.to_frame()
    print("Tempo summary:", schedule.summary())
    print(frame.head(10).to_string(index=False))
    out = ROOT / "examples" / "buyout_tempo_schedule.csv"
    frame.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
