import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

symbol = 'XOM'
start_date = "1993-01-01"
end_date = "2023-12-31"

data = yf.download(symbol, start=start_date, end=end_date) 
df = pd.DataFrame(data)

# Calculate daily returns, annualized drift and volatility
df_returns = pd.DataFrame()
df_returns["Returns"] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
a = df_returns.mean() * 252
sigma = df_returns.std() * np.sqrt(252)
S0 = df['Adj Close'].iloc[-1].item()
a = a.iloc[0] 
sigma = sigma.iloc[0]

# Parameters
T = 1
days = 252
paths = 50000
dt = T/days
r = 0.03  # assumed risk-free rate
alpha = a


np.random.seed(seed=1)

#######################################
# Task 2: GBM Model
#######################################

Z1 = np.random.normal(0, 1, (paths, days+1))
SGBM = np.zeros((paths, days+1))
SGBM[:, 0] = S0

for i in range(1, days+1):
    SGBM[:, i] = SGBM[:, i-1] * np.exp((alpha - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z1[:, i])

# Extract final prices and bin them
ST1D = SGBM[:, -1]
bins = 70
bin_edges = np.linspace(ST1D.min(), ST1D.max(), bins + 1)
binned_v = pd.cut(ST1D, bins=bin_edges, labels=False, include_lowest=True) 
bin_counts = pd.Series(binned_v).value_counts(sort=False).reindex(range(bins), fill_value=0)
bin_probabilities = bin_counts / len(ST1D)

#######################################
# Task 2: Heston Model
#######################################
np.random.seed(1)

# Heston parameters 
kappa = 2
theta = 0.1
rho = -0.5
xi = 0.207

S = np.full((days+1, paths), S0)
V = np.full((days+1, paths), sigma**2)  

Z2 = np.random.normal(size=(days, paths))
Z3 = np.random.normal(size=(days, paths))

dWs = np.sqrt(dt)*Z2
dWv = (rho * Z2 + np.sqrt(1 - rho**2)*Z3)*np.sqrt(dt)

for t in range(1, days+1):
    V[t] = V[t-1] + kappa*(theta - V[t-1])*dt + xi*np.sqrt(V[t-1])*dWv[t-1]
    # Ensure variance remains non-negative
    V[t] = np.maximum(V[t], 1e-10)

for t in range(1, days+1):
    S[t] = S[t-1] + alpha*S[t-1]*dt + S[t-1]*np.sqrt(V[t-1])*dWs[t-1]

S1D = S[-1]
binsH = 70
bin_edgesH = np.linspace(S1D.min(), S1D.max(), binsH+1)
binned_VH = pd.cut(S1D, bins=bin_edgesH, labels=False, include_lowest=True)
bin_countsH = pd.Series(binned_VH).value_counts(sort=False).reindex(range(binsH), fill_value=0)
bin_probabilitiesH = bin_countsH / len(S1D)

#######################################
# Task 3: Risk-Neutral Measure
#######################################


# W_cumsum_gbm: cumulative Brownian motion for GBM
W_cumsum_gbm = np.hstack([np.zeros((paths, 1)), np.cumsum(np.sqrt(dt) * Z1, axis=1)])  # shape (paths, days+1)

# Z(T) for GBM:

Z_T_gbm = np.exp(-((alpha - r)/sigma)*W_cumsum_gbm[:, -1] - 
                 0.5*((alpha - r)/sigma)**2 * T)

S_T_gbm = SGBM[:, -1]
mean_z_t_gbm = Z_T_gbm.mean()
correlation_gbm = np.corrcoef(Z_T_gbm, S_T_gbm)[0,1]
print("GBM Mean of Z(T):", mean_z_t_gbm)
print("GBM Corr(Z(T), S(T)):", correlation_gbm)


bin_means_gbm = []
for i in range(len(bin_edges)-1):
    mask = (S_T_gbm >= bin_edges[i]) & (S_T_gbm < bin_edges[i+1])
    z_mean_gbm = Z_T_gbm[mask].mean() if mask.sum() > 0 else 0
    bin_means_gbm.append(z_mean_gbm)

bin_df_gbm = pd.DataFrame({
    'Bin Range': [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)],
    'Physical Probability': bin_probabilities,
    'Mean Z(T)': bin_means_gbm
})

bin_df_gbm['Risk-Neutral Probability'] = bin_df_gbm['Physical Probability'] * bin_df_gbm['Mean Z(T)']

bin_df_gbm['Risk-Neutral Probability'] /= bin_df_gbm['Risk-Neutral Probability'].sum()

# For the Heston model:


a_r = (alpha - r)


term_stochastic = (a_r / np.sqrt(V[:-1, :])) * dWs
term_drift = (a_r**2 / V[:-1, :]) * dt             

stochastic_integral = np.sum(term_stochastic, axis=0) 
drift_integral = np.sum(term_drift, axis=0)

Z_T_heston = np.exp(- stochastic_integral - 0.5 * drift_integral)

S_T_heston = S1D
mean_z_t_heston = Z_T_heston.mean()
correlation_heston = np.corrcoef(Z_T_heston, S_T_heston)[0,1]
print("Heston Mean of Z(T):", mean_z_t_heston)
print("Heston Corr(Z(T), S(T)):", correlation_heston)

# Compute bin-wise mean Z(T) for Heston
bin_means_heston = []
for i in range(len(bin_edgesH)-1):
    mask = (S_T_heston >= bin_edgesH[i]) & (S_T_heston < bin_edgesH[i+1])
    z_mean_heston = Z_T_heston[mask].mean() if mask.sum() > 0 else 0
    bin_means_heston.append(z_mean_heston)

bin_df_heston = pd.DataFrame({
    'Bin Range': [f"{bin_edgesH[i]:.2f} - {bin_edgesH[i+1]:.2f}" for i in range(len(bin_edgesH)-1)],
    'Physical Probability': bin_probabilitiesH,
    'Mean Z(T)': bin_means_heston
})

bin_df_heston['Risk-Neutral Probability'] = bin_df_heston['Physical Probability'] * bin_df_heston['Mean Z(T)']

bin_df_heston['Risk-Neutral Probability'] /= bin_df_heston['Risk-Neutral Probability'].sum()

###################################
#task 4 european call option
###################################

K = 100

hest_mean_binvalues = (bin_edgesH[:-1] + bin_edgesH[1:]) / 2
hest_p_neutral = bin_df_heston['Risk-Neutral Probability'].values
hest_payoff = np.maximum(hest_mean_binvalues - K, 0) * hest_p_neutral

h_disc_payoff = sum(hest_payoff) * np.exp(-r*T)

gbm_mean_binvalues = (bin_edges[:-1] + bin_edges[1:]) / 2
gbm_p_neutral = bin_df_gbm['Risk-Neutral Probability'].values
gbm_payoff = np.maximum(gbm_mean_binvalues - K, 0) * gbm_p_neutral

g_disc_payoff = sum(gbm_payoff) * np.exp(-r*T)

print("gbm european call",g_disc_payoff)
print("heston european call",h_disc_payoff)

######################################
#task 5 up and out call option
######################################

B = 120

#gbm
SGBM_active_paths = np.ones(paths, dtype=bool)


for i in range(days + 1):
    SGBM_active_paths = SGBM_active_paths & (SGBM[:, i] <= B)


    
gbm_payoff = np.maximum(ST1D - K, 0)
for i in range (paths):
    if SGBM_active_paths[i] == False:
        gbm_payoff[i] = 0

bin_indices = np.digitize(ST1D, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, bins - 1)  



gbm_exotic_payoff = np.zeros(bins) 
for i in range(bins):
    indices_in_bin = bin_indices == i
    if np.any(indices_in_bin):  
        gbm_exotic_payoff[i] = np.mean(gbm_payoff[indices_in_bin])
    else:
        gbm_exotic_payoff[i] = 0 

gbm_exotix_price = sum(gbm_exotic_payoff * gbm_p_neutral) * np.exp(-r*T)

print("gbm exotic price",gbm_exotix_price)

#######################################
#test code to check the value of the option using the closed form solution  
#######################################

# import scipy.stats as ss

# d1 = lambda t, s: 1 / (sigma * np.sqrt(t)) * (np.log(s) + (r + sigma**2 / 2) * t)
# d2 = lambda t, s: 1 / (sigma * np.sqrt(t)) * (np.log(s) + (r - sigma**2 / 2) * t)
# closed_barrier_u = (
#     S0 * (ss.norm.cdf(d1(T, S0 / K)) - ss.norm.cdf(d1(T, S0 / B)))
#     - np.exp(-r * T) * K * (ss.norm.cdf(d2(T, S0 / K)) - ss.norm.cdf(d2(T, S0 / B)))
#     - B * (S0 / B) ** (-2 * r / sigma**2) * (ss.norm.cdf(d1(T, B**2 / (S0 * K))) - ss.norm.cdf(d1(T, B / S0)))
#     + np.exp(-r * T)
#     * K
#     * (S0 / B) ** (-2 * r / sigma**2 + 1)
#     * (ss.norm.cdf(d2(T, B**2 / (S0 * K))) - ss.norm.cdf(d2(T, B / S0)))
# )
# print("The price of the Up and Out call option by closed formula is: ", closed_barrier_u)

#Heston
heston_active_paths = np.ones(paths, dtype=bool)

for i in range(days + 1):
    heston_active_paths = heston_active_paths & (S[i] <= B)

heston_payoff = np.maximum(S1D - K, 0)

for i in range (paths):
    if heston_active_paths[i] == False:
        heston_payoff[i] = 0

bin_indicesH = np.digitize(S1D, bin_edgesH) - 1
bin_indicesH = np.clip(bin_indicesH, 0, binsH - 1)

heston_exotic_payoff = np.zeros(binsH)
for i in range(binsH):
    indices_in_bin = bin_indicesH == i
    if np.any(indices_in_bin):
        heston_exotic_payoff[i] = np.mean(heston_payoff[indices_in_bin])
    else:
        heston_exotic_payoff[i] = 0

heston_exotic_price = sum(heston_exotic_payoff * hest_p_neutral) * np.exp(-r*T)

print("heston exotic price",heston_exotic_price)
