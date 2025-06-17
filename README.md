# 🧮 Derivatives Pricing with Stochastic Models

This project was developed as part of the **Stochastic Calculus for Finance** module in the MSc Quantitative Finance at The University of Manchester.

It implements a full pricing pipeline for both plain-vanilla and exotic options using two stochastic processes: **Geometric Brownian Motion (GBM)** and the **Heston Stochastic Volatility Model**.

---

## 💡 Overview

The project simulates 50,000 stock price paths using:

- **GBM** (with constant drift & volatility)
- **Heston** (with stochastic variance)

It then applies **Girsanov’s Theorem** to convert physical probabilities into **risk-neutral probabilities**, allowing us to price:

- A European call option
- An up-and-out barrier call option

---

## 📈 Models Implemented

- **GBM**:  
  \( dS(t) = \alpha S(t) dt + \sigma S(t) dW(t) \)

- **Heston**:  
  \[
  \begin{aligned}
  dS(t) &= \alpha S(t) dt + \sqrt{v(t)} S(t) dW_S(t) \\\\
  dv(t) &= \kappa(\theta - v(t))dt + \xi \sqrt{v(t)} dW_v(t)
  \end{aligned}
  \]

- **Radon-Nikodym Derivatives** for both processes to compute risk-neutral measures

---

## 🔧 How It Works

1. Historical stock data (e.g. XOM) is downloaded using `yfinance`
2. Parameters α and σ are estimated from log returns
3. Monte Carlo simulations are run under both GBM and Heston models
4. Girsanov’s theorem is applied to derive \( Z(T) \)
5. The final prices are binned and risk-neutral probabilities are computed
6. Payoffs are calculated and discounted to estimate option values

---

## ▶️ Run the Code

Install dependencies:
```bash
pip install -r requirements.txt
