# src/main.py
from models import simulate_gbm_paths, simulate_heston_paths
from risk_neutral import compute_Z_T_gbm, compute_Z_T_heston
from utils import get_bin_stats
from pricing import price_european_call, price_up_and_out_call
import numpy as np
import yfinance as yf
import pandas as pd


#parameters
symbol = 'XOM'
start_date = "1993-01-01"
end_date = "2023-12-31"

#data
data = yf.download(symbol, start=start_date, end=end_date)
df = pd.DataFrame(data)
df_returns = pd.DataFrame()
df_returns["Returns"] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

#annualized alpha and sigma
alpha = df_returns["Returns"].mean() * 252
sigma = df_returns["Returns"].std() * np.sqrt(252)
S0 = df['Adj Close'].iloc[-1].item()
alpha = alpha.item()
sigma = sigma.item()

r = 0.03 #assume risk free rate
T = 1
paths = 50000
days = 252
bins = 70
K = 100
B = 120

#simulate GBM paths
S_gbm, Z_gbm_raw = simulate_gbm_paths(S0, alpha, sigma, T, days, paths)
ST_gbm = S_gbm[:, -1]
Z_T_gbm = compute_Z_T_gbm(alpha, r, sigma, Z_gbm_raw, T / days, T)
bins_gbm = get_bin_stats(ST_gbm, Z_T_gbm, bins)

#price options under GBM
gbm_euro = price_european_call(bins_gbm["bin_means"], bins_gbm["risk_neutral_probs"], K, r, T)
gbm_exotic = price_up_and_out_call(S_gbm, ST_gbm, bins_gbm["bin_edges"], bins_gbm["risk_neutral_probs"], K, B, r, T)

print("GBM European Call Price:", gbm_euro)
print("GBM Up-and-Out Call Price:", gbm_exotic)

#simulate Heston paths
kappa, theta, rho, xi = 2.0, 0.1, -0.5, 0.207
S_heston, V_heston, dWs = simulate_heston_paths(S0, sigma, alpha, T, days, paths, kappa, theta, rho, xi)
ST_heston = S_heston[-1]
Z_T_heston = compute_Z_T_heston(alpha, r, V_heston, dWs, T / days)
bins_heston = get_bin_stats(ST_heston, Z_T_heston, bins)

#price options under Heston
heston_euro = price_european_call(bins_heston["bin_means"], bins_heston["risk_neutral_probs"], K, r, T)
heston_exotic = price_up_and_out_call(S_heston, ST_heston, bins_heston["bin_edges"], bins_heston["risk_neutral_probs"], K, B, r, T)

print("Heston European Call Price:", heston_euro)
print("Heston Up-and-Out Call Price:", heston_exotic)
