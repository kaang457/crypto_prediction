"""Command line: `python -m btcforecast {download,backtest}`."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import backtest, data, features, metrics, report
from .models import RETURN_MODELS, VOLATILITY_MODELS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="btcforecast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="refresh data/market.csv from Yahoo Finance")
    dl.add_argument("--out", type=Path, default=data.DEFAULT_PATH)

    bt = sub.add_parser("backtest", help="walk-forward evaluation of every model")
    bt.add_argument("--data", type=Path, default=data.DEFAULT_PATH)
    bt.add_argument("--out", type=Path, default=Path("reports"))
    bt.add_argument("--train-days", type=int, default=1095, help="rolling training window (days)")
    bt.add_argument("--refit-every", type=int, default=30, help="days between refits")

    args = parser.parse_args(argv)
    if args.command == "download":
        market = data.download()
        data.save(market, args.out)
        print(f"wrote {len(market):,} days ({market.index[0].date()} → {market.index[-1].date()}) to {args.out}")
        return

    started = time.perf_counter()
    market = data.load(args.data)

    X, y = features.dataset(market, features.next_return(market))
    plan = backtest.WalkForward(train_days=args.train_days, refit_every=args.refit_every)
    returns = backtest.predictions(RETURN_MODELS, X, y, plan)
    return_summary = metrics.summarize(returns, market["close"])

    X, y = features.dataset(market, features.future_volatility(market))
    plan = backtest.WalkForward(train_days=args.train_days, refit_every=args.refit_every, gap=features.VOL_HORIZON - 1)
    vol = backtest.predictions(VOLATILITY_MODELS, X, y, plan)
    vol_summary = metrics.summarize_volatility(vol, baseline="Trailing 30 days")

    report.write(returns, return_summary, vol, vol_summary, market["close"], args.out)
    print("Direction: next-day return\n" + report.markdown_table(return_summary))
    print("\nRisk: next-7-day volatility\n" + report.markdown_table(vol_summary, report.VOL_COLUMNS))
    print(f"\n{time.perf_counter() - started:.0f} s. Report and figures in {args.out}/")


if __name__ == "__main__":
    main()
