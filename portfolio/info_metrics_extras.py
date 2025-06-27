import importlib.util
import subprocess
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxpy import Variable, Problem, Maximize, sum as cvx_sum, entr  # type: ignore
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# OPTIONAL: auto-install cvxpy if missing (silent) -----------------------------
# -----------------------------------------------------------------------------
if importlib.util.find_spec("cvxpy") is None:  # pragma: no cover
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvxpy", "ecos", "--quiet"])

# -----------------------------------------------------------------------------
# 1. Soft MaxEnt with heteroskedastic supports ---------------------------------
# -----------------------------------------------------------------------------

def soft_maxent(
    x: np.ndarray,
    tau: float = 0.7,
    k_sigma: float = 1.5,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Maximum-entropy inference with soft mean/variance bands.

    |E[x]−µ̂| ≤ k_sigma * tau * σ           (mean band)
    |E[x²]−v̂| ≤ k_sigma * tau * σ²         (second-moment band)

    Returns
    -------
    p_hat : np.ndarray
        Maximum-entropy probabilities.
    entropy : float
        Shannon entropy of *p_hat*.
    fitted_moments : np.ndarray
        [E_p[x], E_p[x²]].
    target_moments : np.ndarray
        [µ̂, µ̂² + σ̂²].
    """
    x = x.astype(float)
    mu, var, sd = x.mean(), x.var(ddof=0), x.std(ddof=0)
    F = np.vstack([x, x ** 2])
    c = np.array([mu, mu ** 2 + var])
    eps = k_sigma * tau * np.array([sd, sd ** 2])

    p = Variable(len(x))
    constr = [
        p >= 0,
        cvx_sum(p) == 1,
        F[0] @ p >= c[0] - eps[0],
        F[0] @ p <= c[0] + eps[0],
        F[1] @ p >= c[1] - eps[1],
        F[1] @ p <= c[1] + eps[1],
    ]
    Problem(Maximize(cvx_sum(entr(p))), constr).solve(solver="SCS", verbose=False)
    return p.value, -float(p.value @ np.log(p.value + 1e-12)), F @ p.value, c


# -----------------------------------------------------------------------------
# 2. Posterior-predictive residual histogram ----------------------------------
# -----------------------------------------------------------------------------

def posterior_check(actual: np.ndarray, predicted: np.ndarray) -> None:
    resid = actual - predicted
    plt.figure(figsize=(8, 4))
    plt.hist(resid, bins=15, alpha=0.7)
    plt.axvline(0, color="k", ls="--")
    plt.title("Posterior-Predictive Residuals")
    plt.xlabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 3. AIC-style information score ----------------------------------------------
# -----------------------------------------------------------------------------

def info_score(p: np.ndarray, F: np.ndarray, c: np.ndarray) -> float:
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    rmse = float(np.sqrt(np.mean((F @ p - c) ** 2)))
    return 2 * len(c) - 2 * ent + 100 * rmse  # heavier penalty on mis-fit


# -----------------------------------------------------------------------------
# 4. Constrained Ridge regression (Σ w = 1) -----------------------------------
# -----------------------------------------------------------------------------

def ridge_simplex(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    n = X.shape[1]

    def objective(w: np.ndarray) -> float:
        return np.mean((X @ w - y) ** 2) + lam * np.sum(w ** 2)

    w0 = np.ones(n) / n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


# -----------------------------------------------------------------------------
# 5. Quick demonstration -------------------------------------------------------
# -----------------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover
    np.random.seed(42)
    x = np.random.normal(0.07, 0.15, 10)
    p_hat, ent, fitted, target = soft_maxent(x, tau=0.6, k_sigma=2.0)

    print("Soft MaxEnt demo (tau=0.6, k_sigma=2.0)")
    print(f"Entropy          : {ent:.3f}")
    print(f"Target µ         : {target[0]:.4f} | fitted µ : {fitted[0]:.4f}")
    print(f"Target variance  : {target[1] - target[0] ** 2:.4f} | fitted var : {fitted[1] - fitted[0] ** 2:.4f}")

    posterior_check(x, np.full_like(x, fitted[0]))
    score = info_score(p_hat, np.vstack([x, x ** 2]), target)
    print(f"AIC-style score  : {score:.3f}")
    print("First 10 p̂:", np.round(p_hat, 4))


if __name__ == "__main__":  
    _demo()
