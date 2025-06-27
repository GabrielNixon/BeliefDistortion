import pandas as pd, numpy as np, matplotlib.pyplot as plt
from portfolio_kl import run

def drawdown(series):
    growth = (1 + series).cumprod()
    peak = growth.cummax()
    return (growth - peak) / peak

def risk_metrics(r_kl, r_g, idx):
    port_kl = pd.Series(r_kl, index=idx)
    port_g = pd.Series(r_g, index=idx)

    dd_kl = drawdown(port_kl)
    dd_g = drawdown(port_g)

    plt.figure(figsize=(12, 4))
    plt.plot(dd_kl, label="KL")
    plt.plot(dd_g, label="Greedy")
    plt.title("Portfolio Drawdowns")
    plt.legend(); plt.tight_layout()
    plt.show()

    var_kl = np.quantile(port_kl, 0.05)
    var_g = np.quantile(port_g, 0.05)
    print(f"VaR95 KL: {var_kl:.2%}")
    print(f"VaR95 Greedy: {var_g:.2%}")

if __name__ == "__main__":
    r_kl, r_g, idx = run()
    risk_metrics(r_kl, r_g, idx)
