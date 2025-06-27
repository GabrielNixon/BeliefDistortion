import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize

def kl(p, q):
    p, q = np.clip(p, 1e-10, 1), np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def obj(w, mu, w_prev, beta):
    return -np.dot(w, mu) + beta * kl(w, w_prev)

def step_opt(mu, w_prev, delta_t):
    beta = 1 / delta_t
    res = minimize(
        obj,
        w_prev,
        args=(mu, w_prev, beta),
        bounds=[(0, 1)] * len(mu),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP"
    )
    return res.x

def run(path="Finance Portfolio 2.csv"):
    df = pd.read_csv(path, header=3, engine="python")
    assets = [
        "S&P 500 (includes dividends)",
        "3-month T.Bill",
        "US T. Bond (10-year)",
        "Gold*"
    ]
    df = df[["Year"] + assets].dropna()
    for c in assets:
        df[c] = df[c].str.rstrip("%").astype(float) / 100
    df["Year"] = df["Year"].astype(int)
    df.set_index("Year", inplace=True)

    vol = df.pct_change().rolling(5).std()
    stress = ((vol - vol.mean()) / vol.std()).mean(axis=1)
    delta = (0.05 * np.exp(-1.5 * stress)).fillna(method="bfill")
    trust = np.exp(-stress).clip(0.05, 1.0).fillna(method="bfill")

    rets = df[assets].dropna()
    delta_aligned = delta.loc[rets.index]

    w_prev = np.ones(len(assets)) / len(assets)
    w_kl = []
    for t in range(len(rets)):
        mu = rets.iloc[t].values
        w_prev = step_opt(mu, w_prev, delta_aligned.iloc[t])
        w_kl.append(w_prev)
    w_kl = np.array(w_kl)

    w_g = np.zeros_like(w_kl)
    idx_best = rets.values.argmax(axis=1)
    for i, j in enumerate(idx_best):
        w_g[i, j] = 1.0

    r_kl = (w_kl * rets.values).sum(axis=1)
    r_g = (w_g * rets.values).sum(axis=1)

    cum_kl = (1 + r_kl).cumprod()
    cum_g = (1 + r_g).cumprod()

    rf = rets["3-month T.Bill"].values
    sharpe_kl = (r_kl - rf).mean() / (r_kl - rf).std(ddof=0)
    sharpe_g = (r_g - rf).mean() / (r_g - rf).std(ddof=0)

    roll_sharpe_kl = pd.Series(r_kl - rf, index=rets.index).rolling(5).apply(lambda x: x.mean() / x.std(ddof=0), raw=True)
    roll_sharpe_g = pd.Series(r_g - rf, index=rets.index).rolling(5).apply(lambda x: x.mean() / x.std(ddof=0), raw=True)

    plt.figure(figsize=(12, 4))
    plt.plot(rets.index, stress.loc[rets.index], label="Stress")
    plt.plot(rets.index, delta_aligned, label="δ(t)")
    plt.plot(rets.index, trust.loc[rets.index], label="τ(t)")
    plt.legend(); plt.tight_layout()

    plt.figure(figsize=(12, 5))
    plt.plot(rets.index, cum_kl, label="KL")
    plt.plot(rets.index, cum_g, label="Greedy")
    for yr in [2008, 2020]:
        plt.axvline(yr, ls="--", c="gray")
    plt.yscale("log")
    plt.legend(); plt.tight_layout()

    plt.figure(figsize=(12, 4))
    plt.plot(roll_sharpe_kl, label="Sharpe KL")
    plt.plot(roll_sharpe_g, label="Sharpe Greedy")
    plt.axhline(0, c="black")
    plt.legend(); plt.tight_layout()

    plt.figure(figsize=(12, 6))
    for i, a in enumerate(assets):
        plt.plot(rets.index, w_kl[:, i], label=f"KL {a}")
        plt.plot(rets.index, w_g[:, i], ls="--", alpha=0.4, label=f"G {a}")
    plt.legend(ncol=2); plt.tight_layout()
    plt.show()

    print(f"Sharpe KL: {sharpe_kl:.2f}")
    print(f"Sharpe Greedy: {sharpe_g:.2f}")
    return r_kl, r_g, rets.index

if __name__ == "__main__":
    run()
