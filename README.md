# Stress-Aware Portfolios via Info-Metrics

This repository contains the full LaTeX manuscript and figures for the project:

**Stress–Aware Portfolio Choice via Info–Metrics: A Technical Walk-Through (1936–2023)**  
Author: Gabriel Nixon Raj  
NYU Center for Data Science

---

## Overview

This project investigates how bounded rationality can be modeled using information-theoretic tools. It features two complementary experiments:

1. **Non-Stationary Bandit Simulation**  
   - Agents: Bayesian, Trust-Decay, Entropy-Penalized  
   - Environment: Two-armed bandit with a regime switch  
   - Focus: Belief adaptation under uncertainty

2. **KL-Regularized Portfolio Allocation**  
   - Assets: S&P 500, 3-month T-Bill, 10-year Treasury Bond, Gold  
   - Time Period: 1936–2023  
   - Method: KL-constrained optimization with dynamic stress-aware information budgets

---

## Key Results

| Metric                         | KL-Regularized | Greedy |
|-------------------------------|----------------|--------|
| Full-Sample Sharpe Ratio      | 0.48           | 1.13   |
| Annual VaR at 95% Confidence  | -4.12%         | 2.25%  |

The KL-constrained portfolio delivers more stable performance under stress but sacrifices peak return. The greedy strategy achieves higher Sharpe ratios with greater drawdown and concentration risk.

---

## Concepts

- Bounded Rationality and Info-Metrics  
- Thompson Sampling under Regime Shift  
- KL Divergence and Belief Inertia  
- Dynamic Trust Coefficients  
- Stress-Aware Asset Allocation

---
