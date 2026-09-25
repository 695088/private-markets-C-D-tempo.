"""CLI for LP contribution / distribution / tempo workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fund-type",
        choices=["buyout", "vc", "real_estate"],
        default="buyout",
        help="Benchmark fund type workbook",
    )
    parser.add_argument(
        "--size",
        choices=["All", "Large", "Mid", "Small"],
        default="All",
        help="Size sheet within the workbook",
    )
    parser.add_argument(
        "--backend",
        default="mrsr",
        help="Forecast backend (mrsr | lstm)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing Preqin-style .xlsx files (default: ./data)",
    )
    parser.add_argument("--n-paths", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)


def cmd_summarize(args: argparse.Namespace) -> int:
    from private_markets_cd.data.load import load_fund_book
    from private_markets_cd.types import FundType

    book = load_fund_book(args.fund_type, data_dir=args.data_dir)
    rows = []
    for size, kinds in book.levels.items():
        for kind, series in kinds.items():
            rows.append(
                {
                    "fund_type": book.fund_type.value,
                    "size": size.value,
                    "kind": kind.short,
                    "metric": kind.value,
                    "n_quarters": int(series.dropna().shape[0]),
                    "last_level": float(series.dropna().iloc[-1]),
                }
            )
    print(json.dumps({"path": str(book.path), "series": rows}, indent=2))
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    from private_markets_cd.contributions.models import ContributionModel
    from private_markets_cd.distributions.models import DistributionModel
    from private_markets_cd.types import FundSize, FundType

    ft = FundType(args.fund_type)
    sz = FundSize(args.size)
    if args.side == "C":
        model = ContributionModel(
            fund_type=ft,
            size=sz,
            backend_name=args.backend,
            data_dir=args.data_dir,
            seed=args.seed,
        )
    else:
        model = DistributionModel(
            fund_type=ft,
            size=sz,
            backend_name=args.backend,
            data_dir=args.data_dir,
            seed=args.seed,
        )
    result = model.fit_forecast(n_paths=args.n_paths)
    payload = {
        "side": args.side,
        "fund_type": ft.value,
        "size": sz.value,
        "backend": args.backend,
        "horizon": result.horizon,
        "mean_head": result.mean[:10].tolist(),
        "metrics": result.metrics,
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_tempo(args: argparse.Namespace) -> int:
    from private_markets_cd.tempo.schedule import build_tempo_schedule

    schedule = build_tempo_schedule(
        fund_type=args.fund_type,
        size=args.size,
        backend=args.backend,
        n_paths=args.n_paths,
        data_dir=args.data_dir,
        seed=args.seed,
    )
    frame = schedule.to_frame()
    out = {
        "fund_type": schedule.fund_type.value,
        "size": schedule.size.value,
        "backend": schedule.backend,
        "summary": schedule.summary(),
        "schedule_head": frame.head(8).to_dict(orient="list"),
    }
    if args.csv:
        path = Path(args.csv)
        frame.to_csv(path, index=False)
        out["csv"] = str(path.resolve())
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="private-markets-cd",
        description="LP contribution / distribution / tempo framework for private markets",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summarize", help="List available C/D series in a fund workbook")
    _add_common(p_sum)
    p_sum.set_defaults(func=cmd_summarize)

    p_fc = sub.add_parser("forecast", help="Forecast contributions (C) or distributions (D)")
    _add_common(p_fc)
    p_fc.add_argument("--side", choices=["C", "D"], required=True)
    p_fc.set_defaults(func=cmd_forecast)

    p_t = sub.add_parser("tempo", help="Build aligned C vs D tempo schedule")
    _add_common(p_t)
    p_t.add_argument("--csv", type=Path, default=None, help="Optional path to write schedule CSV")
    p_t.set_defaults(func=cmd_tempo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
