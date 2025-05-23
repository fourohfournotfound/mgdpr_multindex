"""Utility functions for rolling feature selection.

This module implements a time-series aware feature-selection routine
that re-evaluates the important columns inside a walk-forward loop.
It wraps several popular selection algorithms such as Boruta-SHAP
and elastic-net with exponential decay.  Each selector is run only
on the training slice so validation data never leaks into the fit.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


# Optional imports that are only required when the corresponding
# method is used.  They are placed inside the function to avoid
# mandatory dependencies at import time.


def _hsic_lasso_select(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    """Placeholder for HSIC-Lasso feature selection.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Target vector.
    k : int
        Maximum number of columns to keep.

    Returns
    -------
    List[str]
        Names of selected columns.
    """
    # In a full implementation we would compute the HSIC score for each
    # feature and solve the convex optimisation problem described in the
    # original paper.  For brevity we simply rank features by absolute
    # correlation with the target and take the top-k.
    scores = X.corrwith(y).abs().sort_values(ascending=False)
    return scores.head(k).index.tolist()


def _enet_forgetting(
    X: pd.DataFrame, y: pd.Series, alpha: float = 1e-3, decay: float = 0.97
) -> List[str]:
    """Elastic-Net with exponential sample weights.

    Older observations receive exponentially smaller weights so that the
    selected coefficients adapt to regime changes.  ``alpha`` controls
    the overall regularisation strength while ``decay`` sets the
    forgetting rate per observation.
    """
    from sklearn.linear_model import ElasticNet

    n_samples = X.shape[0]
    # Newest sample has weight 1.0, oldest has weight ``decay ** (n_samples-1)``
    weights = decay ** (n_samples - 1 - np.arange(n_samples))

    model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    model.fit(X.values, y.values, sample_weight=weights)

    # Keep non-zero coefficients
    coef_mask = np.abs(model.coef_) > 0
    selected = X.columns[coef_mask]
    return list(selected)


def _online_leverage_select_stream(
    X: pd.DataFrame, y: pd.Series, k: int
) -> List[str]:
    """Very small placeholder for a streaming selector.

    The real algorithm maintains leverage scores of the design matrix and
    updates them online.  Here we approximate it by simply returning the
    last ``k`` features by alphabetical order so the example is complete.
    """
    return sorted(X.columns)[-k:]


def walk_forward_selector(
    X: pd.DataFrame,
    y: pd.Series,
    dates: Iterable[pd.Timestamp],
    method: str = "boruta",
    window: int = 750,
    step: int = 20,
    top_k: int = 80,
) -> Iterable[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Yield training and validation indices with selected features.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix indexed identically to ``dates``.
    y : pd.Series
        Target vector aligned with ``X`` and ``dates``.
    dates : Iterable[pd.Timestamp]
        Sequence of dates used for the walk-forward split.
    method : {{"boruta", "hsic", "enet", "online"}}
        Selection algorithm to apply.
    window : int
        Size of the rolling training window.
    step : int
        Number of rows between consecutive splits.
    top_k : int
        Maximum number of features to return.

    Yields
    ------
    Tuple of ``(train_idx, val_idx, selected_features)`` for each window.
    """
    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(
        n_splits=(len(dates) - window) // step,
        test_size=step,
    )

    for train_idx, val_idx in splitter.split(X):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]

        assert X_tr.shape[0] == len(train_idx)
        assert X_tr.shape[1] == X.shape[1]

        if method == "boruta":
            from lightgbm import LGBMRegressor
            from BorutaShap import BorutaShap

            selector = BorutaShap(
                model=LGBMRegressor(n_estimators=350),
                importance_measure="shap",
                classification=False,
                train_or_test="test",
                percentile=70,
                verbose=False,
            )
            selector.fit(X_tr, y_tr)
            keep = selector.features
        elif method == "hsic":
            keep = _hsic_lasso_select(X_tr, y_tr, top_k)
        elif method == "enet":
            keep = _enet_forgetting(X_tr, y_tr)
        elif method == "online":
            keep = _online_leverage_select_stream(X_tr, y_tr, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

        assert X_tr[keep].shape[1] <= top_k

        yield train_idx, val_idx, keep
