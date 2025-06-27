import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize
from itertools import product
from numpy.linalg import inv
import seaborn as sns

# ----------------------------------------------------------------------------
# DATA -----------------------------------------------------------------------
# ----------------------------------------------------------------------------

def load_data(path="Finance Portfolio 2.csv"):
    cols = [
        "S&P 500 (includes dividends)",
        "3-month T.Bill",
        "US T. Bond (10-year)",
        "Gold*",
    ]
    df = pd.read_csv(path, header=3)[["Year"] + cols].dropna()
    for c in cols:
        df[c] = df[c].str.rstrip("%").astype(float) / 100
    df["Year"] = df["Year"].astype(int)
    df.set_index("Year", inplace=True)
    return df, cols

# ----------------------------------------------------------------------------
# MAXENT HELPERS -------------------------------------------------------------
# ----------------------------------------------------------------------------

def soft_maxent(x, moments=("mean", "var"), k_sigma=3.0, tau=1.0):
    x = np.asarray(x, float)
    n = len(x)
    mu, var = x.mean(), x.var(ddof=0)
    skew = ((x - mu) ** 3).mean()
    kurt = ((x - mu) ** 4).mean()
    m_spec = {
        "mean": (x, mu, k_sigma * x.std(ddof=0) / np.sqrt(n)),
        "var": (x ** 2, var, k_sigma * var * 0.5),
        "skew": ((x - mu) ** 3, skew, k_sigma * max(abs(skew), 1e-4)),
        "kurt": ((x - mu) ** 4, kurt, k_sigma * max(abs(kurt), 1e-4)),
    }
    rows, targets, sigs = [], [], []
    for m in moments:
        f, tgt, base = m_spec[m]
        rows.append(f)
        targets.append(tgt)
        sigs.append(base * tau)
    F = np.vstack(rows)
    p = cp.Variable(n)
    constraints = [p >= 0, cp.sum(p) == 1]
    entropy_terms = [cp.sum(cp.entr(p))]
    dual_idx = []
    for k, (sig, tgt) in enumerate(zip(sigs, targets)):
        if sig == 0:
            constraints += [F[k] @ p == tgt]
            dual_idx.append(len(constraints) - 1)
        else:
            v = np.array([-sig, sig])
            w = cp.Variable(2)
            constraints += [w >= 0, cp.sum(w) == 1, F[k] @ p + v @ w == tgt]
            entropy_terms.append(cp.sum(cp.entr(w)))
            dual_idx.append(len(constraints) - 1)
    cp.Problem(cp.Maximize(sum(entropy_terms)), constraints).solve(solver=cp.SCS, verbose=False)
    if p.value is None:
        return None
    duals = [float(np.linalg.norm(np.ravel(constraints[i].dual_value))) for i in dual_idx]
    entropy = -np.sum(p.value * np.log(p.value + 1e-12))
    uni = np.ones_like(p.value) / len(p.value)
    return dict(p=p.value, entropy=entropy, KL_uni=np.sum(p.value * np.log(p.value / uni)), duals=duals, residuals=F @ p.value - np.array(targets))

# ----------------------------------------------------------------------------
# TRUST / STRESS SIGNALS ------------------------------------------------------
# ----------------------------------------------------------------------------

def stress_signals(rets, span=5):
    vol = rets.pct_change().rolling(span).std()
    z = (vol - vol.mean()) / vol.std()
    stress = z.mean(axis=1)
    delta = 0.05 * np.exp(-1.5 * stress)
    tau = np.exp(-stress).clip(0.05, 1.0)
    return stress, delta.fillna(method="bfill"), tau.fillna(method="bfill")

# ----------------------------------------------------------------------------
# KL REGULARISED PORTFOLIO ----------------------------------------------------
# ----------------------------------------------------------------------------

def KL(p, q):
    p, q = np.clip(p, 1e-10, 1), np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))


def trust_KL_portfolio(rets, priors, delta, tau):
    assets = rets.columns
    W, w_prev = [], np.ones(len(assets)) / len(assets)
    for i, y in enumerate(priors.index):
        idx = rets.index.get_loc(y)
        if idx + 1 >= len(rets):
            break
        mu = rets.iloc[idx + 1].values
        beta = 1.0 / (delta.iloc[idx + 1] * tau.loc[y])
        w_prior = np.ones(len(assets)) * priors.iloc[i].mean()
        w_prior /= w_prior.sum()

        def obj(w):
            return -np.dot(w, mu) + beta * KL(w, w_prior)

        res = minimize(obj, w_prev, bounds=[(0, 1)] * len(assets), constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP")
        w_prev = res.x
        W.append(w_prev)
    return np.array(W)

# ----------------------------------------------------------------------------
# MAIN DEMO -------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main():
    df, assets = load_data()
    rets = df[assets].dropna()
    stress, delta, tau = stress_signals(rets)
    sp = rets[assets[0]]

    window = 10
    priors, years = [], []
    for i in range(len(sp) - window + 1):
        seg = sp.iloc[i : i + window]
        res = soft_maxent(seg.values, moments=("mean", "var"), k_sigma=3, tau=tau.loc[seg.index[-1]])
        if res:
            priors.append(res["p"])
            years.append(seg.index[-1])
    priors_df = pd.DataFrame(priors, index=years)

    W = trust_KL_portfolio(rets, priors_df, delta, tau)

    # quick plot
    plt.figure(figsize=(12, 5))
    for i, a in enumerate(assets):
        plt.plot(years[1:], W[:, i], label=a)
    plt.legend(); plt.title("Trust-aware KL Weights"); plt.show()


if __name__ == "__main__":
    main()
