import numpy as np

try:
    from sklearn.feature_selection import f_regression
except Exception:  # sklearn might not be available
    f_regression = None


def graces_select(X, y, k=10, noise_std=0.01, random_state=None):
    """Select top ``k`` features using a simplified GRACES routine.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix ``(n_samples, n_features)``.
    y : np.ndarray
        Target vector of length ``n_samples``.
    k : int, optional
        Number of features to keep.
    noise_std : float, optional
        Stddev of Gaussian noise injected into ``X`` before ranking.
    random_state : int or ``numpy.random.Generator``, optional
        Seed for reproducibility.

    Returns
    -------
    list[int]
        Indices of the ``k`` most important features.
    """
    if f_regression is None:
        raise ImportError("scikit-learn is required for graces_select")
    rng = np.random.default_rng(random_state)
    X_noisy = X + rng.normal(0.0, noise_std, size=X.shape)
    F_vals, _ = f_regression(X_noisy, y)
    F_vals = np.nan_to_num(F_vals)
    ranked = np.argsort(F_vals)[::-1]
    k = min(k, X.shape[1])
    return ranked[:k].tolist()


def elasticnet_select(
    X,
    y,
    k=10,
    alpha=1e-3,
    l1_ratio=0.8,
    decay=0.97,
    lr=0.01,
    epochs=500,
):
    """Feature ranking with an Elastic-Net style linear model.

    The optimisation uses a simple gradient descent with soft-thresholding for
    the L1 part. Sample weights decay exponentially so that recent observations
    receive more emphasis, which helps adapt to regime changes in stock data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Target vector.
    k : int, optional
        Number of top features to return.
    alpha : float, optional
        Overall regularisation strength.
    l1_ratio : float, optional
        Mix between L1 and L2 penalty (0 <= l1_ratio <= 1).
    decay : float, optional
        Exponential decay factor for sample weights. ``decay`` < 1 downweights
        older observations.
    lr : float, optional
        Learning rate for gradient updates.
    epochs : int, optional
        Number of optimisation iterations.

    Returns
    -------
    list[int]
        Indices of the selected features sorted by importance.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    # Standardise features and target
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    Xs = (X - X_mean) / X_std
    y_mean = y.mean()
    ys = y - y_mean

    # Pre-compute sample weights (recent samples at the end of X have weight 1)
    weights = decay ** (np.arange(n_samples)[::-1])
    weights = weights / weights.max()
    w = weights.reshape(-1, 1)

    coef = np.zeros(n_features)
    for _ in range(epochs):
        pred = Xs.dot(coef)
        grad = (w.squeeze() * (pred - ys)) @ Xs / n_samples
        coef -= lr * (grad + alpha * (1 - l1_ratio) * coef)
        # Proximal step for L1
        coef = np.sign(coef) * np.maximum(0, np.abs(coef) - lr * alpha * l1_ratio)

    ranking = np.argsort(np.abs(coef))[::-1]
    k = min(k, n_features)
    return ranking[:k].tolist()
