import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
#  CONFIG ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
CSV_PATH = "Finance Portfolio 2.csv"   # adjust if your data file is elsewhere
WINDOW   = 10                           # rolling window length for priors
RISK_SIG = 3                            # k_sigma for soft‑constraint MaxEnt

ASSETS = [
    "S&P 500 (includes dividends)",
    "3-month T.Bill",
    "US T. Bond (10-year)",
    "Gold*",
]

# -----------------------------------------------------------------------------
#  HELPERS --------------------------------------------------------------------
# -----------------------------------------------------------------------------

def load_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=3)[["Year"] + ASSETS].dropna()
    for c in ASSETS:
        df[c] = df[c].str.rstrip("%" ).astype(float) / 100
    df["Year"] = df["Year"].astype(int)
    df.set_index("Year", inplace=True)
    return df[ASSETS]


def KL(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.clip(p, 1e-10, 1), np.clip(q, 1e-10, 1)
    return float(np.sum(p * np.log(p / q)))


# -----------------------------------------------------------------------------
#  MAIN WORKFLOW ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    rets = load_returns(CSV_PATH)
    stress = ((rets.pct_change().rolling(5).std() - rets.pct_change().rolling(5).std().mean()) / rets.pct_change().rolling(5).std().std()).mean(axis=1)
    delta  = 0.05 * np.exp(-1.5 * stress)
    tau    = np.exp(-stress).clip(0.05, 1.0)

    # --- rolling simple priors (uniform for brevity) -------------------------
    years   = rets.index[WINDOW:]
    priors  = pd.DataFrame(np.tile(np.ones(len(ASSETS))/len(ASSETS), (len(years),1)), index=years, columns=ASSETS)

    # --- build KL portfolio ---------------------------------------------------
    W, betas = [], []
    w_prev = np.ones(len(ASSETS)) / len(ASSETS)
    for yr in years:
        idx = rets.index.get_loc(yr) - 1  # look‑back window end
        mu  = rets.iloc[idx + 1].values
        beta= 1.0 / (delta.iloc[idx + 1] * tau.iloc[idx])
        betas.append(beta)
        w_prior = priors.loc[yr].values

        def obj(w):
            return -np.dot(w, mu) + beta * KL(w, w_prior)

        res = minimize(obj, w_prev, bounds=[(0, 1)] * len(ASSETS), constraints={
            "type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP")
        w_prev = res.x
        W.append(w_prev)

    W = np.array(W)
    kl_ret   = (W * rets.loc[years, ASSETS].values).sum(axis=1)

    # --- greedy benchmark -----------------------------------------------------
    greedy_w = np.zeros_like(W)
    best_idx = rets.loc[years, ASSETS].values.argmax(axis=1)
    greedy_w[np.arange(len(best_idx)), best_idx] = 1.0
    greedy_ret = (greedy_w * rets.loc[years, ASSETS].values).sum(axis=1)

    # --- cumulative regret plot ----------------------------------------------
    cum_kl = (1 + kl_ret).cumprod()
    cum_gr = (1 + greedy_ret).cumprod()
    regret = cum_gr - cum_kl

    plt.figure(figsize=(10, 5))
    plt.plot(years, regret, label="Cum Regret (Greedy − KL)")
    plt.axhline(0, color="gray", ls="--")
    plt.title("Cumulative Regret of KL vs Greedy")
    plt.grid(); plt.legend(); plt.tight_layout(); plt.show()

    # --- beta vs delta diagnostic --------------------------------------------
    beta_s  = pd.Series(betas, index=years, name="beta")
    plt.figure(figsize=(10, 4))
    plt.plot(beta_s, label="β(t)")
    plt.plot(delta.loc[years], label="δ(t)", ls="--")
    plt.title("Lagrange β vs Stress δ(t)")
    plt.grid(); plt.legend(); plt.tight_layout(); plt.show()
    print(f"Correlation β vs δ(t): {beta_s.corr(delta.loc[years]):.2f}")


if __name__ == "__main__":
    main()
