import math

import numpy as np
import pandas as pd
import pytest

from btcforecast import data, metrics


def test_persistence_scores_high_r2_on_a_random_walk():
    """Why price-level R² is not a skill measure: a random walk is 'predicted'
    almost perfectly by yesterday's value."""
    rng = np.random.default_rng(0)
    price = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.03, 2000)))
    assert metrics.r2(price[1:], price[:-1]) > 0.99


def test_direction_is_undefined_for_a_forecast_that_never_calls_it():
    actual = pd.Series([0.01, -0.02])
    assert math.isnan(metrics.directional_accuracy(actual, pd.Series([0.0, 0.0])))
    assert metrics.directional_accuracy(actual, pd.Series([0.5, 0.5])) == 0.5


def test_strategy_pays_fee_on_every_position_change():
    actual = pd.Series(np.log([1.10, 1.10, 1.10]))
    out = metrics.strategy_returns(actual, pd.Series([1.0, -1.0, 1.0]), fee=0.01)
    assert out.tolist() == pytest.approx([0.10 - 0.01, -0.01, 0.10 - 0.01])


def test_max_drawdown():
    assert metrics.max_drawdown(pd.Series([0.5, -0.5, 0.2])) == pytest.approx(-0.5)


def test_macro_series_carry_forward_never_backward():
    days = pd.date_range("2024-01-05", periods=4, freq="D", name="date")  # Fri, Sat, Sun, Mon
    btc = pd.DataFrame({f: 1.0 for f in data.FIELDS}, index=days)
    gold = pd.DataFrame({"close": [2000.0, 2010.0]}, index=pd.DatetimeIndex(["2024-01-05", "2024-01-08"]))
    early = pd.DataFrame({"close": [80.0]}, index=pd.DatetimeIndex(["2024-01-06"]))
    out = data.combine(btc, {"gold": gold, "brent": early})
    # Friday has no Brent print yet; it is dropped rather than back-filled from Saturday.
    assert list(out.index) == list(days[1:])
    assert out["gold_close"].tolist() == [2000.0, 2000.0, 2010.0]


def test_diebold_mariano_separates_real_from_noise_differences():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 3000)
    assert metrics.diebold_mariano(base * 0.8, base, lags=5) < 1e-6
    assert metrics.diebold_mariano(base, base + rng.normal(0, 0.01, 3000), lags=5) > 0.01
