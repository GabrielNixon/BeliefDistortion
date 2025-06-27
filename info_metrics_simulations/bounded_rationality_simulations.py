import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class TwoArmedBandit:
    def __init__(self, p1=0.7, p2=0.5, switch=100):
        self.p1, self.p2, self.switch, self.t = p1, p2, switch, 0
    def pull(self, arm):
        p = [self.p1, self.p2] if self.t < self.switch else [self.p2, self.p1]
        self.t += 1
        return 1 if np.random.rand() < p[arm] else 0

class BayesianAgent:
    def __init__(self):
        self.a, self.b = np.ones(2), np.ones(2)
    def select(self):
        return np.argmax(np.random.beta(self.a, self.b))
    def update(self, arm, r):
        self.a[arm] += r
        self.b[arm] += 1 - r

class TrustDecayAgentDynamic(BayesianAgent):
    def __init__(self, k=0.7, recover=0.002, min_t=0.05):
        super().__init__()
        self.tau, self.k, self.rec, self.min_t = 1.0, k, recover, min_t
        self.trace = []
    def update(self, arm, r):
        m = self.a[arm] / (self.a[arm] + self.b[arm])
        s = abs(r - m)
        self.tau = max(self.min_t, self.tau * np.exp(-self.k * s))
        self.tau += self.rec * (1 - self.tau)
        self.a[arm] = (1 - self.tau) * self.a[arm] + self.tau * (self.a[arm] + r)
        self.b[arm] = (1 - self.tau) * self.b[arm] + self.tau * (self.b[arm] + (1 - r))
        self.trace.append(self.tau)

class EntropyPenalizedAgent(BayesianAgent):
    def __init__(self, lam=0.02):
        super().__init__()
        self.lam = lam
    def update(self, arm, r):
        super().update(arm, r)
        self.a = (1 - self.lam) * self.a + self.lam
        self.b = (1 - self.lam) * self.b + self.lam

def simulate_bandit(agent, steps=200):
    env = TwoArmedBandit()
    R, C, B = [], [], []
    for _ in range(steps):
        arm = agent.select()
        r = env.pull(arm)
        agent.update(arm, r)
        R.append(r)
        C.append(arm)
        B.append(agent.a / (agent.a + agent.b))
    trust = getattr(agent, 'trace', None)
    return np.array(R), np.array(C), np.array(B), np.array(trust) if trust is not None else None

def kl(p, q):
    p, q = np.clip(p, 1e-10, 1), np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def objective(w, mu, w_prev, beta):
    return -np.dot(w, mu) + beta * kl(w, w_prev)

def update_portfolio(mu, w_prev, delta_t):
    beta = 1 / delta_t
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bounds = [(0, 1)] * len(mu)
    res = minimize(objective, w_prev, args=(mu, w_prev, beta), bounds=bounds, constraints=cons, method="SLSQP")
    return res.x

def simulate_portfolio(path="Finance Portfolio 2.csv"):
    df = pd.read_csv(path, header=3, engine="python")
    assets = ["S&P 500 (includes dividends)", "3-month T.Bill", "US T. Bond (10-year)", "Gold*"]
    df = df[["Year"] + assets].dropna()
    for c in assets:
        df[c] = df[c].str.rstrip("%").astype(float) / 100
    df["Year"] = df["Year"].astype(int)
    df.set_index("Year", inplace=True)
    vol = df.pct_change().rolling(5).std()
    z = (vol - vol.mean()) / vol.std()
    z["Stress"] = z.mean(axis=1)
    delta = 0.05 * np.exp(-1.5 * z["Stress"]).fillna(method="bfill")
    rets = df[assets].dropna()
    delta_aligned = delta.loc[rets.index]
    w_prev = np.ones(len(assets)) / len(assets)
    w_kl = []
    for t in range(len(rets)):
        mu = rets.iloc[t].values
        w_prev = update_portfolio(mu, w_prev, delta_aligned.iloc[t])
        w_kl.append(w_prev)
    w_kl = np.array(w_kl)
    w_g = np.zeros_like(w_kl)
    best_idx = rets.values.argmax(axis=1)
    for i, idx in enumerate(best_idx):
        w_g[i, idx] = 1.0
    r_kl = (w_kl * rets.values).sum(axis=1)
    r_g = (w_g * rets.values).sum(axis=1)
    return rets.index, r_kl, r_g, w_kl, w_g

def main():
    np.random.seed(0)
    agents = {
        "Bayes": BayesianAgent(),
        "Trust": TrustDecayAgentDynamic(),
        "Entropy": EntropyPenalizedAgent()
    }
    results = {name: simulate_bandit(ag) for name, ag in agents.items()}
    plt.figure(figsize=(12, 5))
    for name, (r, _, _, _) in results.items():
        plt.plot(np.cumsum(r), label=name, linewidth=2)
    plt.axvline(100, ls="--", c="gray")
    plt.xlabel("steps")
    plt.ylabel("cum reward")
    plt.title("Cumulative reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    years, r_kl, r_g, _, _ = simulate_portfolio()
    cum_kl = (1 + r_kl).cumprod()
    cum_g = (1 + r_g).cumprod()
    plt.figure(figsize=(12, 5))
    plt.plot(years, cum_kl, label="KL-Constrained", linewidth=2)
    plt.plot(years, cum_g, label="Greedy", linewidth=2)
    for yr in [2008, 2020]:
        plt.axvline(yr, color="gray", linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.title("Cumulative Portfolio Value")
    plt.xlabel("Year")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
