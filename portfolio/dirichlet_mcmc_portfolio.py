import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from scipy.special import rel_entr

# expects rets, assets, phat_df, rolling_years, δ, τ, port_kl_phat, port_years
# if not in scope, import helper pipeline (assumes previous scripts live in code/)
try:
    rets
except NameError:  # fallback load
    from pathlib import Path, PurePath
    import importlib.util, sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT / "code"))
    from golan_full_suite import load_data, stress_signals  # type: ignore
    rets, assets = load_data()[0], load_data()[1]
    stress, δ, τ = stress_signals(rets)
    rolling_years = rets.index[10:]
    phat_df = pd.DataFrame(np.tile(np.ones(len(assets)) / len(assets), (len(rolling_years), 1)), index=rolling_years)
    port_kl_phat, port_years = np.tile(np.ones(len(assets)) / len(assets), (len(rolling_years) - 1, 1)), rolling_years[1:]

n_assets = len(assets)
alpha_base = 50
n_samples = 5000

post_mean, ent_list, kl_uni = [], [], []
valid_years = rolling_years[1: len(port_kl_phat) + 1]

for i, yr in enumerate(valid_years):
    w0 = np.full(n_assets, phat_df.iloc[i].mean())
    w0 /= w0.sum()
    beta = 1.0 / (δ.iloc[i + 1] * τ.iloc[i + 1])
    alpha = alpha_base * w0
    acc = []
    for _ in range(n_samples * 3):
        w = dirichlet.rvs(alpha).squeeze()
        if rel_entr(w, w0).sum() <= beta:
            acc.append(w)
            if len(acc) >= n_samples:
                break
    acc = np.array(acc)
    m = acc.mean(axis=0)
    post_mean.append(m)
    ent_list.append(-np.sum(m * np.log(m + 1e-12)))
    kl_uni.append(rel_entr(m, np.ones_like(m) / n_assets).sum())

mcmc_df = pd.DataFrame({"year": valid_years, "entropy": ent_list, "kl_uniform": kl_uni})
for j, a in enumerate(assets):
    mcmc_df[a] = [w[j] for w in post_mean]

plt.figure(figsize=(10, 5))
plt.plot(mcmc_df.year, mcmc_df.entropy, label="Dirichlet Entropy")
if "diag_df" in globals():
    plt.plot(diag_df.year, diag_df.entropy, label="SoftMV Entropy")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

# returns
rets_m = rets.loc[valid_years, assets].values
post_w = np.array(post_mean)
ret_dir = (post_w * rets_m).sum(axis=1)
ret_dir = pd.Series(ret_dir, index=valid_years, name="Dirichlet")
ret_kl = pd.Series((port_kl_phat * rets.loc[port_years, assets].values).sum(axis=1), index=port_years, name="KL")
ret_kl = ret_kl.loc[valid_years]
idx_best = rets_m.argmax(axis=1)
ret_greedy = (np.eye(n_assets)[idx_best] * rets_m).sum(axis=1)
ret_greedy = pd.Series(ret_greedy, index=valid_years, name="Greedy")

cum = lambda r: (1 + r).cumprod()
plt.figure(figsize=(10, 5))
plt.plot(cum(ret_dir), label="Dirichlet")
plt.plot(cum(ret_kl), label="KL")
plt.plot(cum(ret_greedy), label="Greedy")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
plt.plot(cum(ret_greedy) - cum(ret_kl), label="Regret vs KL")
plt.plot(cum(ret_greedy) - cum(ret_dir), label="Regret vs Dirichlet")
plt.axhline(0, ls="--", c="gray")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()
