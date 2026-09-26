"""Transform raw quarterly levels into model-ready series."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def interpolate_gaussian(
    data: pd.DataFrame | pd.Series,
    num_between: int,
    var: float = 8.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Linear interpolate with additive Gaussian noise between observations.

    Port of legacy ``PE_Data.interpolate_gaussian`` with an optional RNG for
    reproducibility.
    """
    rng = rng or np.random.default_rng()
    if isinstance(data, pd.Series):
        values = data.astype(float).to_numpy()
    else:
        values = data.iloc[:, 0].astype(float).to_numpy()

    new_dat: list[float] = []
    for i, level in enumerate(values):
        new_dat.append(float(level))
        if i == len(values) - 1:
            break
        step = (values[i + 1] - level) / (num_between + 1)
        scale = 1.0 / (var * (num_between + 1))
        for j in range(num_between):
            noise = float(rng.normal(loc=0.0, scale=scale))
            new_dat.append(float(level + j * step + noise))

    return pd.DataFrame(new_dat)


def log_diff_series(levels: pd.Series) -> pd.Series:
    """Log-difference of positive quarterly levels (drop zeros / leading NA)."""
    s = levels.astype(float).copy()
    s = s.loc[~(s == 0)]
    prev = s.shift(1)
    delta = np.log(s / prev)
    delta = delta.dropna()
    delta.name = f"dlog_{levels.name or 'cf'}"
    return delta


def train_test_split_series(
    series: pd.Series | pd.DataFrame,
    train_frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = series.to_frame() if isinstance(series, pd.Series) else series
    cut = math.floor(train_frac * len(frame))
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def densify_log_diffs(
    levels: pd.Series,
    steps_between: int = 90,
    train_frac: float = 0.7,
    train_noise_var: float = 6.0,
    test_noise_var: float = 8.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Log-diff then stochastic-interpolate train/test with different noise scales."""
    rng = rng or np.random.default_rng()
    delta = log_diff_series(levels)
    train, test = train_test_split_series(delta, train_frac=train_frac)
    train_i = interpolate_gaussian(train, steps_between, var=train_noise_var, rng=rng)
    test_i = interpolate_gaussian(test, steps_between, var=test_noise_var, rng=rng)
    out = pd.concat([train_i, test_i], ignore_index=True)
    out.columns = ["value"]
    return out


def prepare_supervised(
    densified: pd.DataFrame | pd.Series,
    n_step: int = 180,
    n_ahead: int = 90,
) -> pd.DataFrame:
    """Sliding-window features + response for sequence models (legacy ``prep_dat``)."""
    if isinstance(densified, pd.Series):
        base = densified.to_frame()
    else:
        base = densified.iloc[:, [0]].copy()
    base.columns = ["t+0"]

    a = base.copy()
    for i in range(1, n_step):
        a[f"t+{i}"] = a["t+0"].shift(-i)
    a["target"] = a["t+0"].shift(-(n_step + n_ahead))
    a.dropna(inplace=True)
    return a
