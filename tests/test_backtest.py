import numpy as np
import pandas as pd

from btcforecast import backtest, features
from btcforecast.models import RETURN_MODELS, VOLATILITY_MODELS, Drift, Persistence, Spec
from tests.test_features import synthetic_market


def test_folds_train_strictly_before_test():
    plan = backtest.WalkForward(train_days=100, refit_every=30)
    folds = list(plan.folds(400))
    assert folds[0] == (slice(0, 100), slice(100, 130))
    for train, test in folds:
        assert train.stop == test.start
        assert train.stop - train.start == 100
    assert folds[-1][1].stop == 400
    covered = np.concatenate([np.arange(t.start, t.stop) for _, t in folds])
    assert (covered == np.arange(100, 400)).all()


def returns_dataset(days: int = 500):
    market = synthetic_market(days)
    return features.dataset(market, features.next_return(market))


def test_gap_purges_rows_whose_target_is_not_yet_known():
    plan = backtest.WalkForward(train_days=100, refit_every=30, gap=6)
    for train, test in plan.folds(400):
        assert train.stop + 6 == test.start
        assert train.stop - train.start == 100


def test_predictions_cover_every_out_of_sample_day():
    X, y = returns_dataset()
    plan = backtest.WalkForward(train_days=200, refit_every=50)
    preds = backtest.predictions(RETURN_MODELS, X, y, plan)
    assert list(preds.index) == list(X.index[200:])
    assert not preds.isna().any().any()
    assert (preds["Persistence"] == 0).all()


def test_model_never_sees_the_day_it_predicts():
    """A model that memorises its training targets must not reproduce the test target."""

    class Lookup(Persistence):
        def fit(self, X, y):
            self.seen_ = set(y.index)
            return self

        def predict(self, X):
            assert not self.seen_.intersection(X.index)
            return np.zeros(len(X))

    X, y = returns_dataset()
    backtest.run(Spec("lookup", Lookup), X, y, backtest.WalkForward(train_days=150, refit_every=40))


def test_volatility_training_targets_are_known_before_each_forecast():
    """With a 7-day target, a training row's label is only known 7 days later."""
    horizon = features.VOL_HORIZON

    class Audit(Persistence):
        def fit(self, X, y):
            self.last_known_ = y.index.max() + pd.Timedelta(days=horizon)
            return self

        def predict(self, X):
            assert self.last_known_ <= X.index.min()
            return np.zeros(len(X))

    market = synthetic_market(500)
    X, y = features.dataset(market, features.future_volatility(market))
    backtest.run(Spec("audit", Audit), X, y, backtest.WalkForward(train_days=150, refit_every=40, gap=horizon - 1))


def test_volatility_models_run():
    market = synthetic_market(500)
    X, y = features.dataset(market, features.future_volatility(market))
    preds = backtest.predictions(VOLATILITY_MODELS, X, y, backtest.WalkForward(train_days=200, refit_every=60, gap=6))
    assert not preds.isna().any().any()
    assert (preds["Trailing 30 days"] == X.loc[preds.index, "log_vol_30d"]).all()


def test_drift_predicts_training_mean():
    y = pd.Series([0.01, 0.03])
    assert Drift().fit(None, y).predict(pd.DataFrame(index=[0, 1, 2])).tolist() == [0.02] * 3
