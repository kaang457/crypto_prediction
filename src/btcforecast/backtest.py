"""Walk-forward evaluation.

Every prediction is made by a model trained only on days before it. The
model is refit every `refit_every` days on a rolling window of the most
recent `train_days` days, which is how it would run in production: a
random 80/20 split or shuffled K-fold would let it learn from the future.

When a target spans several days, the newest training rows have targets that
are not yet known on the forecast day. `gap` drops them (purging), otherwise
the model would be trained on the very days it is about to forecast.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from .models import Spec


@dataclass(frozen=True)
class WalkForward:
    train_days: int = 1095
    refit_every: int = 30
    gap: int = 0

    def folds(self, n: int) -> Iterator[tuple[slice, slice]]:
        """(train, test) index slices; train ends `gap` rows before test begins."""
        for start in range(self.train_days + self.gap, n, self.refit_every):
            end = start - self.gap
            yield slice(end - self.train_days, end), slice(start, min(start + self.refit_every, n))


def run(spec: Spec, X: pd.DataFrame, y: pd.Series, plan: WalkForward) -> pd.Series:
    """Out-of-sample predictions, indexed by the day each forecast is made."""
    preds = []
    for train, test in plan.folds(len(X)):
        model = clone(spec.make()).fit(X.iloc[train], y.iloc[train])
        preds.append(pd.Series(model.predict(X.iloc[test]), index=X.index[test]))
    return pd.concat(preds).rename(spec.name)


def predictions(specs: list[Spec], X: pd.DataFrame, y: pd.Series, plan: WalkForward) -> pd.DataFrame:
    out = pd.DataFrame({spec.name: run(spec, X, y, plan) for spec in specs})
    out.insert(0, "actual", y.loc[out.index])
    return out


def to_price(close: pd.Series, log_returns: pd.Series) -> pd.Series:
    """Next-day price implied by a predicted log return made at today's close."""
    return close.loc[log_returns.index] * np.exp(log_returns)
