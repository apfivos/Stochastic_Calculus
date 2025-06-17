# src/utils.py
import numpy as np
import pandas as pd


def get_bin_stats(ST, Z_T, bins=70):
    bin_edges = np.linspace(ST.min(), ST.max(), bins + 1)
    bin_indices = np.digitize(ST, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    bin_counts = pd.Series(bin_indices).value_counts(sort=False).reindex(range(bins), fill_value=0)
    physical_probs = bin_counts / len(ST)

    z_means = []
    for i in range(bins):
        mask = bin_indices == i
        z_mean = Z_T[mask].mean() if mask.any() else 0.0
        z_means.append(z_mean)

    z_means = np.array(z_means)
    risk_neutral_probs = physical_probs.values * z_means
    risk_neutral_probs /= risk_neutral_probs.sum()

    bin_means = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "bin_edges": bin_edges,
        "bin_means": bin_means,
        "physical_probs": physical_probs.values,
        "z_means": z_means,
        "risk_neutral_probs": risk_neutral_probs
    }
