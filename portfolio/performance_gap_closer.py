"""Performance summary & Golan gap‑closer utilities.

This script provides three independent utilities:
1.  Sharpe / final‑value summary with bar‑chart visualisation.
2.  AIC‑style tuning of the soft‑constraint bandwidth k_sigma.
3.  A full “gap‑closer” rolling pipeline: automatic k_sigma selection,
    stress‑adaptive KL optimisation, posterior residual diagnostics, and
    constrained ridge regression for collinear assets.

The code assumes a pre‑existing workspace containing:
    rets          – DataFrame of annual returns indexed by year.
    assets        – list[str]   asset column names in rets.
    rolling_years – Index of rolling window end‑years.
    phat_df       – DataFrame of MaxEnt priors (one per window).
    δ, τ          – Series of information budgets and trust coefficients.
    port_kl_phat, port_years  – KL portfolio weights & their years.

If these are not in scope, the script attempts to fall back to helpers in
`golan_full_suite.py` so that it can still run end‑to‑end.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import cond
from scipy.optimize import minimize
from scipy.special import rel_entr
from scipy.stats import dirichlet

# -----------------------------------------------------------------------------
# FALL‑BACK LOADING (if global objects are missing) ----------------------------
# -----------------------------------------------------------------------------
MISSING = False
try:
    rets  # type: ignore[name‑defined]
except NameError:  # pragma: no cover
    MISSING = True

if MISSING:
    ROOT = Path(__file__).resolve().parents[1] / "code"
    sys.path.append(str(ROOT))
    from golan_full_suite import load_data, stress_signals  # type: ignore

    df, ASSETS = load_data()
    rets, assets = df, ASSETS
    stress, δ, τ = stress_signals(rets)
    rolling_years = rets.index[10:]
    phat_df = pd.DataFrame(
        np.tile(np.ones(len(assets)) / len(assets), (len(rolling_years), 1)),
        index=rolling_years,
    )
    port_kl_phat = np.tile(np.ones(len(assets)) / len(assets), (len(rolling_years) - 1, 1))
    port_years   = rolling_years[1:]

# -----------------------------------------------------------------------------
# UTILITIES -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def sharpe_ratio(r: Sequence[float], rf: float = 0.0) -> float:
    r = np.asarray(r)
    excess = r - rf
    return excess.mean() / excess.std(ddof=1)


def bar_summary(summary: pd.DataFrame) -> None:
    """Two‑panel bar chart: final value and Sharpe ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Final value
    axes[0].bar(summary.index, summary["Final Value"], color=palette)
    axes[0].set_title("Final Portfolio Value (\$1 initial)")
    axes[0].set_ylabel("Dollars")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}", (p.get_x() + 0.05, p.get_height() * 1.01))

    # Sharpe ratio
    axes[1].bar(summary.index, summary["Sharpe Ratio"], color=palette)
    axes[1].set_title("Sharpe Ratio")
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + 0.05, p.get_height() * 1.01))

    plt.suptitle("Strategy Performance Comparison")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# AIC‑STYLE k_sigma TUNING -----------------------------------------------------
# -----------------------------------------------------------------------------

def maxent_soft(x: np.ndarray, tau: float, k_sigma: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu, var, sd = x.mean(), x.var(ddof=0), x.std(ddof=0)
    F = np.vstack([x, x**2])
    c = np.array([mu, mu**2 + var])
    eps = k_sigma * tau * np.array([sd, sd**2])
    import cvxpy as cp  # local import to avoid heavy dependency if unused

    p = cp.Variable(len(x))
    cons = [p >= 0, cp.sum(p) == 1,
            F[0] @ p >= c[0] - eps[0], F[0] @ p <= c[0] + eps[0],
            F[1] @ p >= c[1] - eps[1], F[1] @ p <= c[1] + eps[1]]
    cp.Problem(cp.Maximize(cp.sum(cp.entr(p))), cons).solve(solver="SCS", verbose=False)
    return p.value, -np.sum(p.value * np.log(p.value + 1e-12)), F @ p.value, c


def aic_score(p: np.ndarray, F: np.ndarray, c: np.ndarray) -> float:
    ent = -np.sum(p * np.log(p + 1e-12))
    rmse = np.sqrt(((F @ p - c) ** 2).mean())
    return 2 * len(c) - 2 * ent + 100 * rmse


def choose_k_sigma(x: np.ndarray, tau: float, grid: Sequence[float] = (0.5, 1, 1.5, 2, 3)) -> float:
    scores = []
    for k in grid:
        p, *_rest = maxent_soft(x, tau, k)
        score = aic_score(p, np.vstack([x, x**2]), np.array([x.mean(), x.var(ddof=0) + x.mean() ** 2]))
        scores.append(score)
    return grid[int(np.argmin(scores))]


# -----------------------------------------------------------------------------
# CONSTRAINED RIDGE -----------------------------------------------------------
# -----------------------------------------------------------------------------

def constrained_ridge(X: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    def obj(w):
        return ((X @ w - y) ** 2).mean() + lam * (w @ w)

    w0 = np.ones(X.shape[1]) / X.shape[1]
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    res = minimize(obj, w0, bounds=[(0, 1)] * X.shape[1], constraints=cons)
    return res.x


# -----------------------------------------------------------------------------
# GAP‑CLOSER PIPELINE ---------------------------------------------------------
# -----------------------------------------------------------------------------

def gap_closer_demo() -> None:  # pragma: no cover
    window = 10
    port_w, beta_series, resid_list = [], [], []
    for i in range(len(rets) - window - 1):
        w_data = rets.iloc[i : i + window]
        yr = rets.index[i + window]
        stress_proxy = w_data.std().mean() / w_data.mean().abs().mean()
        delta_t = 0.05 * np.exp(-1.5 * stress_proxy)
        tau_t = np.exp(-stress_proxy)
        beta_t = 1.0 / (delta_t * tau_t)
        beta_series.append((yr, beta_t))

        sp_window = w_data[assets[0]].values
        best_k = choose_k_sigma(sp_window, tau_t)
        p_hat, _, _, _ = maxent_soft(sp_window, tau_t, best_k)

        w_prior = np.ones(len(assets)) * p_hat.mean()
        w_prior /= w_prior.sum()

        mu = w_data.mean().values
        def obj(w):
            return -w @ mu + beta_t * np.sum(w * np.log(np.clip(w / w_prior, 1e-10, None)))
        res = minimize(obj, np.ones(len(assets)) / len(assets),
                       bounds=[(0, 1)] * len(assets), constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        port_w.append(res.x)
        resid_list.append(sp_window - p_hat[0])

    beta_s = pd.Series(dict(beta_series))
    print("Gap‑closer finished. Portfolio years:", len(port_w))
    plt.figure(figsize=(10, 4))
    plt.plot(beta_s)
    plt.title("β(t) over time (gap‑closer demo)")
    plt.grid(); plt.tight_layout(); plt.show()
    plt.figure(figsize=(8, 4))
    plt.hist(np.hstack(resid_list), bins=20, alpha=0.7)
    plt.title("Posterior‑predictive residuals (all windows)")
    plt.grid(); plt.tight_layout(); plt.show()


# -----------------------------------------------------------------------------
# MAIN EXAMPLE ----------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__": 
    kl_ret   = pd.Series((port_kl_phat * rets.loc[port_years, assets].values).sum(axis=1), index=port_years)
    greedy   = pd.Series(rets.loc[port_years, assets].values.max(axis=1), index=port_years)
    dir_post = kl_ret.copy() * 0.0  
    summary = pd.DataFrame({
        "Final Value": [ (1 + kl_ret).cumprod().iloc[-1], (1 + dir_post).cumprod().iloc[-1], (1 + greedy).cumprod().iloc[-1] ],
        "Sharpe Ratio": [ sharpe_ratio(kl_ret), sharpe_ratio(dir_post), sharpe_ratio(greedy) ]
    }, index=["KL", "Dirichlet", "Greedy"])
    bar_summary(summary)
    gap_closer_demo()
