"""Features built only from information available at the close of day t,
and the two targets the models are asked to forecast from them.

Prices are not stationary: a model trained on 2017 prices has never seen
2024's, which is what broke the tree models in the course version of this
project. Returns, ratios and oscillators keep every feature on a comparable
scale across a decade in which the price rose 200-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOL_HORIZON = 7


def log_returns(market: pd.DataFrame) -> pd.Series:
    return np.log(market["close"]).diff()


def next_return(market: pd.DataFrame) -> pd.Series:
    """Direction task: log(close[t+1] / close[t]), known at the close of t+1."""
    return log_returns(market).shift(-1).rename("next_return")


def future_volatility(market: pd.DataFrame, horizon: int = VOL_HORIZON) -> pd.Series:
    """Risk task: log of the standard deviation of daily returns over days
    t+1 .. t+horizon. It is only known at the close of t+horizon, so training
    rows must stop `horizon - 1` days before a forecast is made."""
    return np.log(log_returns(market).rolling(horizon).std().shift(-horizon)).rename("future_volatility")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / loss)


def build(market: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix; rows with an incomplete look-back window are dropped."""
    close = market["close"]
    ret = log_returns(market)
    f = pd.DataFrame(index=market.index)

    for lag in range(5):
        f[f"ret_lag{lag}"] = ret.shift(lag)
    f["ret_7d"] = np.log(close / close.shift(7))
    f["ret_30d"] = np.log(close / close.shift(30))
    f["vol_7d"] = ret.rolling(7).std()
    f["vol_30d"] = ret.rolling(30).std()
    # Log realised-volatility terms at daily, weekly and monthly scales (the HAR model's inputs).
    f["log_abs_ret"] = np.log(ret.abs().clip(lower=1e-4))
    f["log_vol_7d"] = np.log(f["vol_7d"])
    f["log_vol_30d"] = np.log(f["vol_30d"])

    f["close_vs_sma20"] = close / close.rolling(20).mean() - 1
    f["close_vs_sma50"] = close / close.rolling(50).mean() - 1
    mid, std = close.rolling(20).mean(), close.rolling(20).std()
    f["bollinger_pos"] = (close - mid) / (2 * std)
    f["rsi_14"] = rsi(close) / 100
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd"] = macd / close
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    f["range"] = (market["high"] - market["low"]) / close

    log_volume = np.log(market["volume"].replace(0, np.nan))
    f["volume_z30"] = (log_volume - log_volume.rolling(30).mean()) / log_volume.rolling(30).std()

    for name in ("gold", "brent"):
        macro = np.log(market[f"{name}_close"])
        f[f"{name}_ret_1d"] = macro.diff()
        f[f"{name}_ret_5d"] = macro.diff(5)

    f["day_of_week"] = market.index.day_of_week

    return f.replace([np.inf, -np.inf], np.nan).dropna()


def dataset(market: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target on the days where both are defined."""
    X = build(market)
    y = target.reindex(X.index).dropna()
    return X.loc[y.index], y
