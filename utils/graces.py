import numpy as np
from sklearn.feature_selection import f_regression

def graces_select(X, y, k=10, noise_std=0.01, random_state=None):
    """Select important features using a simplified GRACES routine.

    This function mimics the GRACES feature selection idea by injecting Gaussian
    noise into the features and ranking them with an F-statistic. It is
    intentionally lightweight so it can run offline without external
    dependencies beyond scikit-learn.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features). ``X`` should already be
        scaled appropriately for best results.
    y : np.ndarray
        Target values for the training set. Continuous targets are expected.
    k : int, optional
        Number of features to keep. Defaults to 10.
    noise_std : float, optional
        Standard deviation of the Gaussian noise injected into ``X``. Defaults to
        0.01.
    random_state : int or ``numpy.random.Generator``, optional
        Random seed or generator for reproducible noise.

    Returns
    -------
    List[int]
        Indices of the ``k`` most informative features sorted by importance.

    Notes
    -----
    This is a simplified approximation of the GRACES algorithm described in
    ``offline-docs/turn0search5.pdf``. The true method uses graph-aware
    convolutions and iterative reweighting. Here we compute classical
    regression F-scores after adding small Gaussian perturbations. It generally
    works well for offline ranking tasks and avoids label leakage if executed
    only on the training split.
    """
    rng = np.random.default_rng(random_state)
    X_noisy = X + rng.normal(loc=0.0, scale=noise_std, size=X.shape)

    F_vals, _ = f_regression(X_noisy, y)
    F_vals = np.nan_to_num(F_vals)
    ranked = np.argsort(F_vals)[::-1]
    k = min(k, X.shape[1])
    return ranked[:k].tolist()
