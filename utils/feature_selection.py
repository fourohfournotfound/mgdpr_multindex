import numpy as np

try:
    from sklearn.feature_selection import f_regression
except Exception:  # sklearn might not be available
    f_regression = None

try:
    from BorutaShap import BorutaShap  # type: ignore
    import lightgbm as lgb  # type: ignore
except Exception:  # BorutaShap or lightgbm might not be available
    BorutaShap = None  # type: ignore
    lgb = None  # type: ignore

try:
    import torch
except Exception:  # PyTorch is optional for gating selectors
    torch = None


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


def boruta_shap_select(
    X,
    y,
    k=10,
    n_estimators=350,
    percentile=70,
    classification=False,
    verbose=False,
):
    """Feature ranking using the Boruta-Shap wrapper around LightGBM.

    This method requires the ``BorutaShap`` and ``lightgbm`` packages. If they
    are not available an ``ImportError`` is raised.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix ``(n_samples, n_features)``.
    y : np.ndarray
        Target vector.
    k : int, optional
        Number of top features to return. If ``None``, all confirmed features
        are returned.
    n_estimators : int, optional
        Number of trees for the LightGBM model used inside Boruta-Shap.
    percentile : float, optional
        Percentile threshold for feature confirmation.
    classification : bool, optional
        Whether to perform classification rather than regression.
    verbose : bool, optional
        Verbosity flag forwarded to Boruta-Shap.

    Returns
    -------
    list[int]
        Indices of the selected features sorted by importance.
    """
    if BorutaShap is None or lgb is None:
        raise ImportError(
            "BorutaShap and lightgbm are required for boruta_shap_select"
        )

    model = lgb.LGBMRegressor(n_estimators=n_estimators)
    selector = BorutaShap(
        model=model,
        importance_measure="shap",
        classification=classification,
        percentile=percentile,
        verbose=verbose,
        train_or_test="test",
    )
    selector.fit(X, y, sample=False, allow_str=False)
    ranks = np.asarray(selector.ranking)
    order = np.argsort(ranks)
    if k is not None:
        order = order[:k]
    return order.tolist()


def _gate_select(X, y, k=10, hidden_dim=32, epochs=20, lr=0.01):
    """Internal helper implementing a small gating network with PyTorch."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for tft_select and dygformer_select"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

    gate = torch.nn.Linear(X_t.shape[1], X_t.shape[1], bias=False)
    net = torch.nn.Sequential(
        gate,
        torch.nn.Softmax(dim=1),
        torch.nn.Linear(X_t.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(int(epochs)):
        opt.zero_grad()
        preds = net(X_t)
        loss = torch.nn.functional.mse_loss(preds, y_t)
        loss.backward()
        opt.step()

    importance = gate.weight.abs().sum(0).detach().cpu().numpy()
    ranking = np.argsort(importance)[::-1]
    k = min(k, X_t.shape[1])
    return ranking[:k].tolist()


def tft_select(X, y, k=10, hidden_dim=32, epochs=20, lr=0.01):
    """Select features via a lightweight TFT-style gating mechanism."""
    return _gate_select(X, y, k, hidden_dim, epochs, lr)


def dygformer_select(X, y, k=10, hidden_dim=32, epochs=20, lr=0.01):
    """Select features via a lightweight DyGFormer-style gating mechanism."""
    return _gate_select(X, y, k, hidden_dim, epochs, lr)

