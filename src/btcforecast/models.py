"""Candidate models and the baselines each one has to beat.

Direction models predict the next day's log return. Baselines matter more
than usual here: daily crypto returns are close to a random walk, so
"tomorrow's price equals today's" is a strong forecast that scores R² > 0.99
on prices.

Volatility models predict the log standard deviation of the next week's
daily returns. The baseline is the classic estimate: the past 30 days'.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class Persistence(BaseEstimator, RegressorMixin):
    """Predicts a zero return: tomorrow's close equals today's (random walk)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Persistence:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class Drift(BaseEstimator, RegressorMixin):
    """Predicts the mean return of the training window: always 'up' in a bull market."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Drift:
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class Column(BaseEstimator, RegressorMixin):
    """Predicts one feature as is: a trailing estimate used as a baseline."""

    def __init__(self, column: str = "log_vol_30d"):
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Column:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy()


HAR_TERMS = ["log_abs_ret", "log_vol_7d", "log_vol_30d"]


def har() -> BaseEstimator:
    """Heterogeneous autoregressive model (Corsi, 2009): next-period volatility as
    a linear blend of daily, weekly and monthly realised volatility."""
    return make_pipeline(FunctionTransformer(lambda X: X[HAR_TERMS]), RidgeCV(alphas=np.logspace(-3, 3, 13)))


@dataclass(frozen=True)
class Spec:
    name: str
    make: Callable[[], BaseEstimator]
    baseline: bool = False


RETURN_MODELS: list[Spec] = [
    Spec("Persistence", Persistence, baseline=True),
    Spec("Drift", Drift, baseline=True),
    Spec("Ridge", lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))),
    Spec(
        "Random forest",
        lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=42),
    ),
    Spec(
        "Gradient boosting",
        lambda: HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=42
        ),
    ),
]

VOLATILITY_MODELS: list[Spec] = [
    Spec("Trailing 30 days", Column, baseline=True),
    Spec("Trailing 7 days", lambda: Column("log_vol_7d"), baseline=True),
    Spec("HAR", har),
    Spec(
        "Gradient boosting",
        lambda: HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=42
        ),
    ),
]
