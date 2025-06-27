import numpy as np, pandas as pd, matplotlib.pyplot as plt, cvxpy as cp
from scipy.optimize import minimize

# ---------- 1. load data ----------
path = "Finance Portfolio 2.csv"
df = pd.read_csv(path, header=3, engine="python")
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

# ---------- 2. stress signal, information budget δ, trust τ ----------
vol = rets.pct_change().rolling(5).std()
stress = ((vol - vol.mean()) / vol.std()).mean(axis=1)
_delta = 0.05 * np.exp(-1.5 * stress)
_tau = np.exp(-stress).clip(0.05, 1.0)

_delta = _delta.fillna(method="bfill")
_tau = _tau.fillna(method="bfill")

# ---------- 3. helpers ----------

def KL(p, q):
    p, q = np.clip(p, 1e-10, 1), np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))


def maxent_probs(x):
    x = np.asarray(x, float)
    if np.allclose(x.var(ddof=0), 0):
        return None
    F = np.vstack([x, x ** 2])
    c = np.array([x.mean(), x.var(ddof=0)])
    p = cp.Variable(len(x))
    prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(p))), [p >= 0, cp.sum(p) == 1, F @ p == c])
    prob.solve(solver=cp.SCS, verbose=False)
    return None if p.value is None else p.value


# ---------- 4. full‑sample max‑entropy on S&P ----------
sp = rets[assets[0]].values
mu_hat, var_hat = sp.mean(), sp.var(ddof=0)
F_full = np.vstack([sp, sp ** 2])
c_full = np.array([mu_hat, mu_hat ** 2 + var_hat])
p_full = cp.Variable(len(sp))
prob_full = cp.Problem(cp.Maximize(cp.sum(cp.entr(p_full))), [p_full >= 0, cp.sum(p_full) == 1, F_full @ p_full == c_full])
prob_full.solve(solver=cp.SCS, verbose=False)
p_full_val = p_full.value

print("Full‑sample MaxEnt moments check → mean:", np.dot(sp, p_full_val), "var:", np.dot(sp ** 2, p_full_val) - np.dot(sp, p_full_val) ** 2)

plt.figure(figsize=(11, 4))
plt.stem(rets.index, p_full_val, basefmt=" ")
plt.title("Max‑Entropy PMF for Annual S&P 500 Returns")
plt.tight_layout()
plt.show()

# ---------- 5. rolling max‑entropy priors ----------
window = 10
rolling_phat, rolling_years = [], []
ser = rets[assets[0]]
for i in range(len(ser) - window + 1):
    seg = ser.iloc[i : i + window]
    p_hat = maxent_probs(seg.values)
    if p_hat is not None:
        rolling_phat.append(p_hat)
        rolling_years.append(seg.index[-1])
phat_df = pd.DataFrame(rolling_phat, index=rolling_years)

# ---------- 6. KL‑regularised portfolio (MaxEnt prior) ----------
weights_kl, years_kl = [], []
w_prev = np.ones(len(assets)) / len(assets)
for i, y in enumerate(rolling_years):
    idx = rets.index.get_loc(y)
    if idx + 1 >= len(rets):
        break
    mu = rets.iloc[idx + 1].values
    delta_t = _delta.iloc[idx + 1]
    prior_scalar = phat_df.iloc[i].mean()
    w_prior = np.ones(len(assets)) * prior_scalar
    w_prior /= w_prior.sum()
    beta = 1 / delta_t

    def obj(w):
        return -np.dot(w, mu) + beta * KL(w, w_prior)

    res = minimize(obj, w_prev, bounds=[(0, 1)] * len(assets), constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w_prev = res.x
    weights_kl.append(w_prev)
    years_kl.append(rets.index[idx + 1])
weights_kl = np.array(weights_kl)

plt.figure(figsize=(12, 6))
for i, a in enumerate(assets):
    plt.plot(years_kl, weights_kl[:, i], label=a)
plt.title("KL‑Regularised Portfolio Weights (MaxEnt Prior)")
plt.legend(); plt.tight_layout(); plt.show()

# ---------- 7. trust‑aware KL‑regularised ----------
weights_trust, years_trust = [], []
w_prev = np.ones(len(assets)) / len(assets)
for i, y in enumerate(rolling_years):
    idx = rets.index.get_loc(y)
    if idx + 1 >= len(rets):
        break
    mu = rets.iloc[idx + 1].values
    delta_t = _delta.iloc[idx + 1]
    tau_t = _tau.loc[y]
    beta = (1 / delta_t) * (1 / tau_t)
    prior_scalar = phat_df.iloc[i].mean()
    w_prior = np.ones(len(assets)) * prior_scalar
    w_prior /= w_prior.sum()

    def obj(w):
        return -np.dot(w, mu) + beta * KL(w, w_prior)

    res = minimize(obj, w_prev, bounds=[(0, 1)] * len(assets), constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w_prev = res.x
    weights_trust.append(w_prev)
    years_trust.append(rets.index[idx + 1])
weights_trust = np.array(weights_trust)

plt.figure(figsize=(12, 6))
for i, a in enumerate(assets):
    plt.plot(years_trust, weights_trust[:, i], label=a)
plt.title("Trust‑Aware KL‑Regularised Portfolio Weights\n(MaxEnt Prior + Stress & Trust)")
plt.legend(); plt.tight_layout(); plt.show()
