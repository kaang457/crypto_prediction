"""Daily market data: Bitcoin plus two macro series (gold and Brent crude).

Bitcoin trades every day; gold and Brent futures trade on weekdays. The
dataset follows Bitcoin's calendar and carries the last known macro close
forward over weekends and holidays. It never fills backward, so a row only
ever contains information that was public on that date.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

SYMBOLS = {"btc": "BTC-USD", "gold": "GC=F", "brent": "BZ=F"}
FIELDS = ["open", "high", "low", "close", "volume"]
DEFAULT_PATH = Path("data/market.csv")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_daily(symbol: str, start: datetime, end: datetime, session: requests.Session | None = None) -> pd.DataFrame:
    """Daily OHLCV for one Yahoo Finance symbol, indexed by UTC date."""
    http = session or requests.Session()
    resp = http.get(
        CHART_URL.format(symbol=symbol),
        params={"period1": int(start.timestamp()), "period2": int(end.timestamp()), "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    resp.raise_for_status()
    chart = resp.json()["chart"]
    if chart.get("error"):
        raise RuntimeError(f"{symbol}: {chart['error']['description']}")
    result = chart["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {field: quote[field] for field in FIELDS},
        index=pd.to_datetime(result["timestamp"], unit="s").normalize(),
    )
    frame.index.name = "date"
    # Yahoo occasionally returns a duplicate row for the current, unfinished day.
    return frame[~frame.index.duplicated(keep="last")].dropna(subset=["close"])


def combine(btc: pd.DataFrame, macro: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join macro closes onto Bitcoin's calendar without looking ahead."""
    out = btc[FIELDS].copy()
    for name, frame in macro.items():
        aligned = frame["close"].reindex(out.index.union(frame.index)).ffill().reindex(out.index)
        out[f"{name}_close"] = aligned
    return out.dropna()


def download(start: datetime = datetime(2014, 9, 17, tzinfo=UTC), end: datetime | None = None) -> pd.DataFrame:
    end = end or datetime.now(UTC)
    with requests.Session() as session:
        frames = {name: fetch_daily(symbol, start, end, session) for name, symbol in SYMBOLS.items()}
    btc = frames.pop("btc")
    # Drop today's row: the candle is still open and its close would change.
    btc = btc[btc.index < pd.Timestamp(end.date())]
    return combine(btc, frames)


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def save(frame: pd.DataFrame, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, float_format="%.6f")
