import numpy as np
import pandas as pd
import pytest

from btcforecast import features


def synthetic_market(days: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=days, freq="D", name="date")
    close = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.03, days)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.005, days)),
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": rng.uniform(1e9, 5e9, days),
            "gold_close": 1_800 * np.exp(np.cumsum(rng.normal(0, 0.01, days))),
            "brent_close": 70 * np.exp(np.cumsum(rng.normal(0, 0.02, days))),
        },
        index=index,
    )


def test_features_use_no_future_data():
    """Features for day t must not change when any later day is altered."""
    market = synthetic_market()
    altered = market.copy()
    altered.iloc[300:] *= 1.5  # rewrite every column from day 300 on
    before = market.index[300]
    a = features.build(market)
    b = features.build(altered)
    a, b = a[a.index < before], b[b.index < before]
    assert len(a) > 200
    pd.testing.assert_frame_equal(a, b)
    # Removing the future entirely gives the same rows too.
    truncated = features.build(market.loc[:before])
    pd.testing.assert_frame_equal(truncated[truncated.index < market.index[299]], a[a.index < market.index[299]])


def test_next_return_is_tomorrows_log_return():
    market = synthetic_market()
    target = features.next_return(market)
    day, nxt = market.index[50], market.index[51]
    assert target[day] == pytest.approx(np.log(market.loc[nxt, "close"] / market.loc[day, "close"]))


def test_future_volatility_covers_the_next_horizon_days_only():
    market = synthetic_market()
    target = features.future_volatility(market, horizon=7)
    returns = np.log(market["close"]).diff()
    day = 100
    expected = np.log(returns.iloc[day + 1 : day + 8].std())
    assert target.iloc[day] == pytest.approx(expected)
    assert target.iloc[-7:].isna().all()


def test_dataset_drops_days_without_a_target():
    market = synthetic_market()
    X, y = features.dataset(market, features.next_return(market))
    assert X.index.equals(y.index)
    assert X.index[-1] == market.index[-2]
