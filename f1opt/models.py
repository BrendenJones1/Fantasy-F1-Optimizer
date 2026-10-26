"""Contenders for the point-prediction benchmark that need only scikit-learn.

The heavier models (LightGBM, PyTorch MLP) live in ``f1opt.models_heavy`` so the
serving path does not import torch. All models expose fit(X, y) / predict(X) on a
DataFrame of ``FEATURE_COLUMNS`` so the evaluator can treat them uniformly."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class NaiveRollingMean(BaseEstimator, RegressorMixin):
    """Predict the driver's mean points over their last 3 races (falls back to the training mean)."""

    def __init__(self, column: str = "drv_pts_ma3"):
        self.column = column

    def fit(self, X: pd.DataFrame, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame):
        return X[self.column].fillna(self.fallback_).to_numpy()


def make_ridge(alpha: float = 10.0):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=alpha))


def get_model_factories(include_heavy: bool = True):
    """Name -> factory for every contender; heavy models are imported only on request."""
    factories = {"Naive (rolling-3 mean)": NaiveRollingMean, "Ridge": make_ridge}
    if include_heavy:
        from f1opt.models_heavy import make_lightgbm, TorchMLP
        factories.update({"LightGBM": make_lightgbm, "PyTorch MLP": TorchMLP})
    return factories
