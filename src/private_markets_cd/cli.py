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


def cmd_pace(args: argparse.Namespace) -> int:
    from private_markets_cd.tempo.pace import build_pace_curve

    pace = build_pace_curve(args.fund_type, args.size, data_dir=args.data_dir)
    out = {
        "summary": pace.summary(),
        "head": pace.frame.head(8).to_dict(orient="list"),
        "tail": pace.frame.tail(4).to_dict(orient="list"),
    }
    if args.csv:
        path = Path(args.csv)
        pace.frame.to_csv(path, index=False)
        out["csv"] = str(path.resolve())
    print(json.dumps(out, indent=2))
    return 0


def cmd_commitment(args: argparse.Namespace) -> int:
    from private_markets_cd.lp import LPCommitment

    lp = LPCommitment(
        commitment=args.commitment,
        fund_type=args.fund_type,
        size=args.size,
        data_dir=args.data_dir,
    )
    out = lp.overview(include_forecast=args.forecast, backend=args.backend)
    if args.csv:
        path = Path(args.csv)
        lp.cashflows().to_csv(path, index=False)
        out["csv"] = str(path.resolve())
    print(json.dumps(out, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from private_markets_cd.lp import compare_fund_types

    frame = compare_fund_types(
        commitment=args.commitment,
        size=args.size,
        data_dir=args.data_dir,
    )
    print(frame.to_string(index=False))
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"Wrote {Path(args.csv).resolve()}")
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

    p_t = sub.add_parser(
        "tempo",
        help="Model-backed C/D intensity forecast schedule (MRSR/LSTM backends)",
    )
    _add_common(p_t)
    p_t.add_argument("--csv", type=Path, default=None, help="Optional path to write schedule CSV")
    p_t.set_defaults(func=cmd_tempo)

    p_p = sub.add_parser(
        "pace",
        help="Historical LP tempo from cumulative %% Called Up / Distributed",
    )
    _add_common(p_p)
    p_p.add_argument("--csv", type=Path, default=None)
    p_p.set_defaults(func=cmd_pace)

    p_c = sub.add_parser(
        "commitment",
        help="Scale historical C/D pace to an LP commitment amount",
    )
    _add_common(p_c)
    p_c.add_argument("--commitment", type=float, default=10_000_000.0)
    p_c.add_argument(
        "--forecast",
        action="store_true",
        help="Also attach model-backed intensity forecast summary",
    )
    p_c.add_argument("--csv", type=Path, default=None)
    p_c.set_defaults(func=cmd_commitment)

    p_cmp = sub.add_parser("compare", help="Compare Buyout / VC / RE pace for one commitment")
    _add_common(p_cmp)
    p_cmp.add_argument("--commitment", type=float, default=10_000_000.0)
    p_cmp.add_argument("--csv", type=Path, default=None)
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
