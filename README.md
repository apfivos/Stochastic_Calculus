# 🧮 Derivatives Pricing with Stochastic Models

This project was developed to fulfill the coding requirements for the group CourseWorkAssigment in the **Stochastic Calculus for Finance** module in the MSc Quantitative Finance at The University of Manchester.

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

- **GBM (Geometric Brownian Motion):**

  dS(t) = α·S(t)·dt + σ·S(t)·dW(t)

- **Heston Stochastic Volatility Model:**

  dS(t) = α·S(t)·dt + √v(t)·S(t)·dW_S(t)  
  dv(t) = κ(θ − v(t))·dt + ξ·√v(t)·dW_V(t)

- **Radon-Nikodym Derivatives**  
  used to convert physical measures to risk-neutral measures.

---

## 🔧 How It Works

1. Historical stock data (e.g. XOM) is downloaded using `yfinance`
2. Parameters α and σ are estimated from log returns
3. Monte Carlo simulations are run under both GBM and Heston models
4. Girsanov’s theorem is applied to derive Z(T)
5. The final prices are binned and risk-neutral probabilities are computed
6. Payoffs are calculated and discounted to estimate option values

---

## ▶️ Run the Code

Install dependencies:
```bash
pip install -r requirements.txt
```

Then run:
```bash
python src/main.py
```

---



## 📚 Notes

- No data or figures are stored in the repo.
- The script automatically downloads all required data.
- Designed for academic and educational purposes.

---

## ©️ Author

Apollon Foivos Bakis — MSc Quantitative Finance  
University of Manchester  
