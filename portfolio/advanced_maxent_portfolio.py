import numpy as np, pandas as pd, matplotlib.pyplot as plt, cvxpy as cp
from scipy.optimize import minimize

# ---------- load data ----------
df = pd.read_csv("Finance Portfolio 2.csv", header=3)
assets = [
    "S&P 500 (includes dividends)",
    "3-month T.Bill",
    "US T. Bond (10-year)",
    "Gold*",
]

df = df[["Year"] + assets].dropna()
for c in assets:
    df[c] = df[c].str.rstrip("%").astype(float) / 100
df["Year"] = df["Year"].astype(int)
df.set_index("Year", inplace=True)
rets = df[assets]

# ---------- stress, δ and τ ----------
vol = rets.pct_change().rolling(5).std()
stress = ((vol - vol.mean()) / vol.std()).mean(axis=1)
_delta = 0.05 * np.exp(-1.5 * stress)
_tau = np.exp(-stress).clip(0.05, 1.0)
_delta = _delta.fillna(method="bfill")
_tau = _tau.fillna(method="bfill")

# ---------- helpers ----------

def KL(p, q):
    p, q = np.clip(p, 1e-10, 1), np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))


def maxent_soft(x, k_sigma=3.0):
    x = np.asarray(x, float)
    n = len(x)
    mu, var = x.mean(), x.var(ddof=0)
    sig_mu = k_sigma * (x.std(ddof=0) / np.sqrt(n))
    sig_var = k_sigma * var * 0.5
    v_mu = np.array([-sig_mu, sig_mu])
    v_var = np.array([-sig_var, sig_var])
    p = cp.Variable(n)
    w_mu = cp.Variable(2)
    w_var = cp.Variable(2)
    F = np.vstack([x, x ** 2])
    constr = [
        p >= 0,
        cp.sum(p) == 1,
        w_mu >= 0,
        cp.sum(w_mu) == 1,
        w_var >= 0,
        cp.sum(w_var) == 1,
        F[0] @ p + v_mu @ w_mu == mu,
        F[1] @ p + v_var @ w_var == var,
    ]
    prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(p)) + cp.sum(cp.entr(w_mu)) + cp.sum(cp.entr(w_var))), constr)
    prob.solve(solver=cp.SCS, verbose=False)
    if p.value is None:
        return None, None, None, None
    lam_mu = np.linalg.norm(np.ravel(constr[6].dual_value))
    lam_var = np.linalg.norm(np.ravel(constr[7].dual_value))
    entropy = -np.sum(p.value * np.log(p.value + 1e-12))
    return p.value, lam_mu, lam_var, entropy

# ---------- rolling priors ----------
window = 10
sp = rets[assets[0]]
phats, years, lam1s, lam2s, ents = [], [], [], [], []
for i in range(len(sp) - window + 1):
    seg = sp.iloc[i : i + window]
    ph, l1, l2, H = maxent_soft(seg.values)
    if ph is not None:
        phats.append(ph)
        lam1s.append(l1)
        lam2s.append(l2)
        ents.append(H)
        years.append(seg.index[-1])
phat_df = pd.DataFrame(phats, index=years)
lam1 = pd.Series(lam1s, index=years, name="lam_mu")
lam2 = pd.Series(lam2s, index=years, name="lam_var")
entropy_s = pd.Series(ents, index=years, name="entropy")

# ---------- trust-aware KL portfolio ----------
W, w_prev = [], np.ones(len(assets)) / len(assets)
for i, y in enumerate(years):
    idx = rets.index.get_loc(y)
    if idx + 1 >= len(rets):
        break
    mu = rets.iloc[idx + 1].values
    beta = 1.0 / (_delta.iloc[idx + 1] * _tau.loc[y])
    prior_scalar = phat_df.iloc[i].mean()
    w_prior = np.ones(len(assets)) * prior_scalar
    w_prior /= w_prior.sum()

    def obj(w):
        return -np.dot(w, mu) + beta * KL(w, w_prior)

    res = minimize(obj, w_prev, bounds=[(0, 1)] * len(assets), constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP")
    w_prev = res.x
    W.append(w_prev)
W = np.array(W)
port_years = years[1:]

# ---------- diagnostics ----------
plt.figure(figsize=(11, 4))
entropy_s.plot(title="Entropy (MaxEnt Prior)")
plt.tight_layout(); plt.show()

plt.figure(figsize=(11, 4))
lam1.plot(label="lambda_mu")
lam2.plot(label="lambda_var")
plt.legend(); plt.title("Dual Multipliers"); plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 5))
for i, a in enumerate(assets):
    plt.plot(port_years, W[:, i], label=a)
plt.title("KL Portfolio Weights (Soft MaxEnt Prior)")
plt.legend(); plt.tight_layout(); plt.show()

# ---------- residual check first window ----------
seg = sp.iloc[:window]
ph = phats[0]
mu_err = seg.mean() - np.dot(seg.values, ph)
var_err = seg.var(ddof=0) - (np.dot(seg.values ** 2, ph) - np.dot(seg.values, ph) ** 2)
print(f"Residuals first window: delta_mu={mu_err:.6f}, delta_var={var_err:.6f}")

# ---------- Bayesian, naive, MCMC-KL benchmarks ----------
from numpy.linalg import inv

def bayes_mean(prior, k0, sm, n):
    return (k0 * prior + n * sm) / (k0 + n)

def markowitz(mu, Sigma, risk_av=6):
    w = inv(Sigma + np.eye(len(Sigma)) * 1e-6) @ mu / risk_av
    w = np.clip(w, 0, None)
    return w / w.sum()

bayes_W = []
for t, y in enumerate(years):
    idx = rets.index.get_loc(y)
    if idx + 1 >= len(rets):
        break
    sample = rets.iloc[: idx + 1]
    mu_hat = sample.mean().values
    Sigma = sample.cov().values
    mu_post = bayes_mean(np.zeros(len(assets)), 1.0, mu_hat, len(sample))
    bayes_W.append(markowitz(mu_post, Sigma))

aBay = np.array(bayes_W)
ret_bayes = (aBay * rets.loc[years[1:], assets].values).sum(axis=1)
ret_naive = (rets.loc[years[1:], assets].values * (np.ones(len(assets)) / len(assets))).sum(axis=1)

def sample_dirichlet(w0, n=3000, beta=0.1):
    draws = np.random.dirichlet(w0 * 50, n)
    keep = np.array([KL(d, w0) <= beta for d in draws])
    return draws[keep]

kl_W = []
for i, y in enumerate(years):
    idx = rets.index.get_loc(y)
    if idx + 1 >= len(rets):
        break
    beta = 1.0 / (_delta.iloc[idx + 1] * _tau.loc[y])
    prior_scalar = phat_df.iloc[i].mean()
    w0 = np.ones(len(assets)) * prior_scalar
    w0 /= w0.sum()
    draw = sample_dirichlet(w0, 2000, beta)
    if draw.size == 0:
        draw = w0.reshape(1, -1)
    kl_W.append(draw.mean(axis=0))
kl_W = np.array(kl_W)
ret_kl = (kl_W * rets.loc[years[1:], assets].values).sum(axis=1)

cum = lambda r: (1 + r).cumprod()
plt.figure(figsize=(12, 5))
plt.plot(years[1:], cum(ret_bayes), label="Bayesian MV")
plt.plot(years[1:], cum(ret_naive), label="Naive 1/N")
plt.plot(years[1:], cum(ret_kl), label="Dirichlet MCMC-KL")
plt.yscale("log")
plt.legend(); plt.tight_layout(); plt.show()

for name, r in zip(["Bayes", "Naive", "MCMC-KL"], [ret_bayes, ret_naive, ret_kl]):
    sharpe = (r - rets.loc[years[1:], "3-month T.Bill"].values).mean() / r.std(ddof=0)
    print(f"{name:7} Sharpe : {sharpe:.2f}")
