# src/models.py
import numpy as np


def simulate_gbm_paths(S0, alpha, sigma, T, days, paths, seed=1):
    dt = T / days
    np.random.seed(seed)
    Z = np.random.normal(0, 1, (paths, days + 1))
    S = np.zeros((paths, days + 1))
    S[:, 0] = S0

    for i in range(1, days + 1):
        S[:, i] = S[:, i - 1] * np.exp((alpha - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, i])

    return S, Z


def simulate_heston_paths(S0, sigma, alpha, T, days, paths, kappa, theta, rho, xi, seed=1):
    dt = T / days
    np.random.seed(seed)

    S = np.full((days + 1, paths), S0)
    V = np.full((days + 1, paths), sigma ** 2)

    Z1 = np.random.normal(size=(days, paths))
    Z2 = np.random.normal(size=(days, paths))

    dWs = np.sqrt(dt) * Z1
    dWv = (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2) * np.sqrt(dt)

    for t in range(1, days + 1):
        V[t] = V[t - 1] + kappa * (theta - V[t - 1]) * dt + xi * np.sqrt(np.maximum(V[t - 1], 0)) * dWv[t - 1]
        V[t] = np.maximum(V[t], 1e-10)  # Avoid negative variances

    for t in range(1, days + 1):
        S[t] = S[t - 1] + alpha * S[t - 1] * dt + S[t - 1] * np.sqrt(V[t - 1]) * dWs[t - 1]

    return S, V, dWs




def compute_Z_T_gbm(alpha, r, sigma, Z, dt, T):
    W_T = np.cumsum(np.sqrt(dt) * Z, axis=1)[:, -1]
    Z_T = np.exp(-((alpha - r) / sigma) * W_T - 0.5 * ((alpha - r) / sigma) ** 2 * T)
    return Z_T


def compute_Z_T_heston(alpha, r, V, dWs, dt):
    a_r = alpha - r
    term1 = (a_r / np.sqrt(V[:-1, :])) * dWs
    term2 = (a_r ** 2 / V[:-1, :]) * dt
    stochastic_int = np.sum(term1, axis=0)
    drift_int = np.sum(term2, axis=0)
    Z_T = np.exp(-stochastic_int - 0.5 * drift_int)
    return Z_T
