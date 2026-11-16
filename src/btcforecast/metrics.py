"""Scores that separate real skill from tracking yesterday's price."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEE = 0.001  # 10 bp per position change, roughly a retail exchange taker fee


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, lags: int) -> float:
    """Two-sided p-value that forecasts a and b have equal mean absolute error.

    Uses a Newey-West variance with `lags` autocovariances, which multi-day
    targets need: neighbouring forecast errors overlap and are correlated."""
    d = np.abs(np.asarray(errors_a)) - np.abs(np.asarray(errors_b))
    n = len(d)
    centered = d - d.mean()
    var = centered @ centered / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (centered[k:] @ centered[:-k]) / n
    if var <= 0:
        return float("nan")
    stat = d.mean() / math.sqrt(var / n)
    return math.erfc(abs(stat) / math.sqrt(2))


def directional_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """Share of days on which the predicted sign matched the realised one.
    Undefined for a forecast that never takes a side (zero return)."""
    called = predicted != 0
    if not called.any():
        return float("nan")
    return float((np.sign(predicted[called]) == np.sign(actual[called])).mean())


def strategy_returns(actual: pd.Series, predicted: pd.Series, fee: float = FEE) -> pd.Series:
    """Hold Bitcoin on days the forecast is positive, cash otherwise, net of fees."""
    position = (predicted > 0).astype(float)
    trades = position.diff().abs().fillna(position.iloc[0])
    return position * (np.exp(actual) - 1) - trades * fee


def summarize_volatility(preds: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Errors on log volatility. Skill is the error reduction against `baseline`."""
    actual = preds["actual"]
    base_mae = float(np.mean(np.abs(actual - preds[baseline])))
    rows = []
    for name in preds.columns.drop("actual"):
        err = actual - preds[name]
        mae = float(np.mean(np.abs(err)))
        rows.append(
            {
                "model": name,
                "mae_log": mae,
                "r2_log": r2(actual.to_numpy(), preds[name].to_numpy()),
                # Typical size of the miss as a ratio: exp(MAE) - 1 on the volatility itself.
                "typical_miss_pct": (np.exp(mae) - 1) * 100,
                "skill_pct": (1 - mae / base_mae) * 100,
                "p_value": float("nan") if name == baseline else diebold_mariano(err, actual - preds[baseline], lags=10),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def summarize(preds: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    actual = preds["actual"]
    today = close.loc[preds.index]
    true_price = today * np.exp(actual)
    persistence_mae = float(np.mean(np.abs(true_price - today)))
    years = len(actual) / 365
    rows = []
    for name in preds.columns.drop("actual"):
        pred = preds[name]
        price = today * np.exp(pred)
        mae = float(np.mean(np.abs(true_price - price)))
        strat = strategy_returns(actual, pred)
        p_value = float("nan") if name == "Persistence" else diebold_mariano(true_price - price, true_price - today, lags=5)
        rows.append(
            {
                "model": name,
                "mae_usd": mae,
                "mape_pct": float(np.mean(np.abs(true_price - price) / true_price) * 100),
                "r2_price": r2(true_price.to_numpy(), price.to_numpy()),
                "r2_return": r2(actual.to_numpy(), pred.to_numpy()),
                "skill_vs_persistence_pct": (1 - mae / persistence_mae) * 100,
                "p_value": p_value,
                "direction_acc_pct": directional_accuracy(actual, pred) * 100,
                "strategy_cagr_pct": ((1 + strat).prod() ** (1 / years) - 1) * 100,
                "strategy_max_drawdown_pct": max_drawdown(strat) * 100,
            }
        )
    hold = np.exp(actual) - 1
    rows.append(
        {
            "model": "Buy and hold",
            "strategy_cagr_pct": ((1 + hold).prod() ** (1 / years) - 1) * 100,
            "strategy_max_drawdown_pct": max_drawdown(hold) * 100,
        }
    )
    return pd.DataFrame(rows).set_index("model")


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())
