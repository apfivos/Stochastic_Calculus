# src/risk_neutral.py
import numpy as np


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
