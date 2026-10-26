"""Heavier contenders (LightGBM, PyTorch MLP); kept apart so the API image need not install torch."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_lightgbm(seed: int = 0):
    # Small, heavily regularised: ~2-3k rows and noisy targets.
    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=30,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                         random_state=seed, verbose=-1, n_jobs=1)


class _Net(nn.Module):
    """Same architecture as the original ``CostEffectivenessModel`` (64-32-16, dropout 0.2)."""

    def __init__(self, n_in, hidden=(64, 32, 16), dropout=0.2):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchMLP(BaseEstimator, RegressorMixin):
    """The original PyTorch MLP, retrained per fold on leakage-safe features.

    Fixed epochs and hyperparameters (no early stopping on the evaluation fold);
    predictions are averaged over ``n_seeds`` initialisations to reduce variance.
    """

    def __init__(self, epochs=60, lr=1e-3, weight_decay=1e-4, batch_size=64, n_seeds=3):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_seeds = n_seeds

    def fit(self, X: pd.DataFrame, y):
        self.prep_ = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        Xt = torch.tensor(self.prep_.fit_transform(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.y_mean_, self.y_std_ = yt.mean(), yt.std()
        yt = (yt - self.y_mean_) / self.y_std_
        self.nets_ = []
        for seed in range(self.n_seeds):
            torch.manual_seed(seed)
            net = _Net(Xt.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            loss_fn = nn.MSELoss()
            n = len(Xt)
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(n)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    if len(idx) < 2:
                        continue
                    opt.zero_grad()
                    loss_fn(net(Xt[idx]).squeeze(-1), yt[idx]).backward()
                    opt.step()
            net.eval()
            self.nets_.append(net)
        return self

    def predict(self, X: pd.DataFrame):
        Xt = torch.tensor(self.prep_.transform(X), dtype=torch.float32)
        with torch.no_grad():
            preds = torch.stack([net(Xt).squeeze(-1) for net in self.nets_]).mean(0)
        return (preds * self.y_std_ + self.y_mean_).numpy()
