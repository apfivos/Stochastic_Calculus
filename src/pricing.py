# src/pricing.py
import numpy as np


def price_european_call(bin_means, risk_neutral_probs, K, r, T):
    payoff = np.maximum(bin_means - K, 0)
    discounted_payoff = np.sum(payoff * risk_neutral_probs) * np.exp(-r * T)
    return discounted_payoff


def price_up_and_out_call(S_paths, ST, bin_edges, risk_neutral_probs, K, B, r, T):
    paths = S_paths.shape[0] if S_paths.ndim == 2 else S_paths.shape[1]
    days = S_paths.shape[1] - 1 if S_paths.ndim == 2 else S_paths.shape[0] - 1

    active = np.all(S_paths <= B, axis=1) if S_paths.ndim == 2 else np.all(S_paths <= B, axis=0)
    payoffs = np.maximum(ST - K, 0)
    payoffs[~active] = 0

    bins = len(risk_neutral_probs)
    bin_indices = np.digitize(ST, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    exotic_bin_avg = np.zeros(bins)
    for i in range(bins):
        mask = bin_indices == i
        exotic_bin_avg[i] = np.mean(payoffs[mask]) if mask.any() else 0.0

    exotic_price = np.sum(exotic_bin_avg * risk_neutral_probs) * np.exp(-r * T)
    return exotic_price
