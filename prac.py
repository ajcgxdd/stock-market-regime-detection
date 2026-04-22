"""
Stock Market Regime Detection & Volatility Forecasting (Upgraded)
-----------------------------------------------------------------
Updates:
1. Feature Stationarity: Uses VIX Z-Score and Bollinger %B.
2. Scientific Selection: Calculates AIC/BIC to find optimal HMM states.
3. Advanced Volatility: Uses GJR-GARCH (skewed) with Student's T distribution.
4. Actionable Backtest: Simulates a trading strategy based on regimes.

Files used:
- /mnt/data/nhd.csv  (NIFTY — DD-MM-YY)
- /mnt/data/ivd.csv  (India VIX — YYYY-MM-DD)
"""

# ---------------------------
# Imports & setup
# ---------------------------
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
import ta

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NIFTY_FILE = 'nhd.csv'
VIX_FILE   = 'ivd.csv'

# ---------------------------
# 1. Load CSVs with robust parsing
# ---------------------------
def load_nifty(path):
    print(f"Loading NIFTY CSV: {path}")
    df = pd.read_csv(
        path,
        skiprows=2,
    )
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%y", errors='coerce')
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index('Date').sort_index()
    return df[['Open','High','Low','Close','Volume']]

def load_vix(path):
    print(f"Loading India VIX CSV: {path}")
    df = pd.read_csv(
        path,
        skiprows=2,
    )
    # Changed to %d-%m-%y to match the new CSV format
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%y", errors='coerce')
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index('Date').sort_index()
    return df[['Open','High','Low','Close','Volume']]

# Load Data
try:
    index_df = load_nifty(NIFTY_FILE)
    vix_df   = load_vix(VIX_FILE)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure input CSVs are in the root directory.")
    sys.exit(1)

# Align dates
common_idx = index_df.index.intersection(vix_df.index)
print(f"Aligned Data: {len(common_idx)} rows (Intersection of Nifty & VIX)")
index_df = index_df.loc[common_idx]
vix_df   = vix_df.loc[common_idx]

# ---------------------------
# 2. Advanced Feature Engineering
# ---------------------------
df = index_df.copy()
df['Adj Close'] = df['Close']
df['return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))

# Volatility features
for w in [10, 20]:
    df[f'vol_{w}'] = df['return'].rolling(w).std() * np.sqrt(252)

# Technical Indicators
try:
    # RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(df['Adj Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Adj Close'])
    df['macd'] = macd.macd()
    
    # Bollinger Bands %B (Better stationarity than Width)
    bb = ta.volatility.BollingerBands(df['Adj Close'])
    df['bb_pband'] = bb.bollinger_pband() 
    df['bb_width'] = bb.bollinger_wband()
    
    # ATR
    df['atr_14']  = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
except Exception as e:
    print("TA error:", e)

# VIX Engineering (Z-Score is more stationary than raw price)
df['vix_close'] = vix_df['Close']
df['vix_zscore'] = (df['vix_close'] - df['vix_close'].rolling(50).mean()) / df['vix_close'].rolling(50).std()

# Drop NaNs created by rolling windows
df = df.dropna()

# ---------------------------
# 3. Scientific HMM State Selection (AIC/BIC)
# ---------------------------
print("\n=== Optimizing HMM States ===")
# We use Returns, Volatility, and VIX Z-Score to form regimes
hmm_features = ['return', 'vol_20', 'vix_zscore'] 
X_hmm_raw = df[hmm_features].values

# Scale features so variances don't overwhelm the model
scaler = StandardScaler()
X_hmm = scaler.fit_transform(X_hmm_raw)

import warnings
warnings.filterwarnings("ignore")

best_score = float('inf')
best_k = 3
best_model = None

# Loop to find optimal K (2 to 5)
for k in range(2, 6):
    try:
        tmp_model = GaussianHMM(n_components=k, covariance_type='full', n_iter=1000, random_state=42)
        tmp_model.fit(X_hmm)
        
        # Calculate AIC: AIC = -2 * logL + 2 * k * n_features
        logL = tmp_model.score(X_hmm)
        n_features = X_hmm.shape[1]
        
        # Number of parameters for 'full' covariance: 
        # Transitions: k*(k-1)
        # Means: k*n_features
        # Covariances: k * n_features * (n_features + 1) / 2
        n_params = k * (k - 1) + k * n_features + k * n_features * (n_features + 1) / 2
        
        aic = -2 * logL + 2 * n_params
        
        print(f"States: {k} | LogL: {logL:.2f} | AIC: {aic:.2f}")
        
        if aic < best_score:
            best_score = aic
            best_k = k
            best_model = tmp_model
    except Exception as e:
        print(f"Error fitting K={k}: {e}")
        continue

print(f"-> Selected Optimal States: {best_k}")

# Predict States using Best Model
hidden_states = best_model.predict(X_hmm)
df['hmm_state'] = hidden_states

# Map States to Logic (Bear/Neutral/Bull) based on Mean Return
state_stats = df.groupby('hmm_state')['return'].agg(['mean', 'std', 'count'])
print("\nState Statistics:\n", state_stats)

# Sort states by mean return: Lowest = Bear, Highest = Bull
sorted_states = state_stats.sort_values('mean').index

# Dynamic labeling based on optimal K
if best_k == 2:
    labels = ['Bear', 'Bull']
elif best_k == 3:
    labels = ['Bear', 'Neutral', 'Bull']
elif best_k == 4:
    labels = ['Bear', 'Neutral', 'Bull', 'Strong Bull']
elif best_k == 5:
    labels = ['Strong Bear', 'Bear', 'Neutral', 'Bull', 'Strong Bull']
else:
    labels = [f"State_{i}" for i in range(best_k)]

state_mapping = {state_id: labels[i] for i, state_id in enumerate(sorted_states)}

print("State Mapping:", state_mapping)
df['regime_desc'] = df['hmm_state'].map(state_mapping)

# Save Regime Data
df.to_csv(os.path.join(OUTPUT_DIR,'regimes_hmm.csv'))

# Plot Regimes
colors = ['red', 'orange', 'green', 'blue', 'purple']
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Adj Close'], color='black', alpha=0.3, label='Price')

for s in sorted_states:
    mask = df['hmm_state'] == s
    lbl = state_mapping[s]
    clr = colors[list(sorted_states).index(s) % len(colors)]
    plt.scatter(df.index[mask], df['Adj Close'][mask], s=4, color=clr, label=lbl)

plt.title(f'NIFTY Regimes (Optimal K={best_k})')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'regime_plot.png'))
plt.close()

# ---------------------------
# 4. Strategy Backtest (Equity Curve)
# ---------------------------
print("\n=== Running Strategy Backtest ===")

# Define dynamic weights based on new labels
weight_map = {
    'Strong Bear': -0.5, # Shorting allowed
    'Bear': 0.0,         # Cash
    'Neutral': 0.5,      # 50% Exposure
    'Bull': 1.0,         # 100% Fully Invested
    'Strong Bull': 1.5   # 150% Leveraged
}

# Apply weights (Default to 0.5 if label not found)
df['strategy_weight'] = df['regime_desc'].map(weight_map).fillna(0.5)

# IMPORTANT: Shift weights by 1 day. 
# We detect regime at Close today -> We trade at Open tomorrow (or capture tomorrow's return).
df['target_position'] = df['strategy_weight'].shift(1).fillna(0)

# Factor in transaction costs for turnover (10 bps per trade)
tc_rate = 0.0010
df['turnover'] = df['target_position'].diff().abs().fillna(0)

df['buy_hold_ret'] = df['return']
# Realistic strategy return (Net of trading costs)
df['strategy_ret'] = (df['return'] * df['target_position']) - (df['turnover'] * tc_rate)

# Cumulative Returns
df['cum_buy_hold'] = (1 + df['buy_hold_ret']).cumprod()
df['cum_strategy'] = (1 + df['strategy_ret']).cumprod()

print(f"-> Buy & Hold Return: {(df['cum_buy_hold'].iloc[-1] - 1) * 100:.2f}%")
print(f"-> HMM Strategy Return (Net): {(df['cum_strategy'].iloc[-1] - 1) * 100:.2f}%")

# Latest Recommendation
current_regime = df['regime_desc'].iloc[-1]
next_day_weight = df['strategy_weight'].iloc[-1] # No shift for current day signal
print(f"\n=> LIVE SIGNAL ----------------------------")
print(f"=> Last close Date: {df.index[-1].strftime('%Y-%m-%d')}")
print(f"=> Current Regime: {current_regime}")
print(f"=> Recommended Market Exposure Tomorrow: {next_day_weight * 100:.0f}%")
print(f"-------------------------------------------\n")

# Plot Backtest
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['cum_buy_hold'], label='Buy & Hold (Nifty)', color='gray')
plt.plot(df.index, df['cum_strategy'], label='HMM Strategy (Net of Costs)', color='blue')
plt.title(f'Strategy Backtest (Final Capital: {df["cum_strategy"].iloc[-1]:.2f}x)')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'strategy_backtest.png'))
plt.close()

# ---------------------------
# 5. Advanced GARCH Forecasting (GJR-GARCH + Student's T)
# ---------------------------
print("\n=== GJR-GARCH Volatility Forecasting ===")

ret_pct = df['return'] * 100 # Convert to percentage for GARCH
FORECAST_DAYS = 20

# GJR-GARCH(1,1) with Student's T distribution
# o=1 enables asymmetry (Leverage effect)
# dist='t' handles fat tails
am = arch_model(ret_pct, vol='Garch', p=1, o=1, q=1, dist='t', rescale=False)
try:
    res = am.fit(disp='off', options={'ftol': 1e-4})
    print(res.summary())
except Exception as e:
    print(f"GARCH Error: {e}\nRetrying with continuous scaling...")
    res = am.fit(disp='off', update_freq=0)
    print(res.summary())

# Forecast
fcast = res.forecast(horizon=FORECAST_DAYS)
last_var_row = fcast.variance.iloc[-1].values
daily_vol_pct = np.sqrt(last_var_row)
annual_vol_pct = daily_vol_pct * np.sqrt(252)

# Date handling for forecast
last_date = df.index[-1]
future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

forecast_df = pd.DataFrame({
    'daily_vol_pct': daily_vol_pct,
    'annualized_vol_pct': annual_vol_pct
}, index=future_dates)

forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'garch_future_vol.csv'))

# Get recent historical volatility for context (last 60 days)
hist_vol = df['vol_20'].iloc[-60:] * 100

plt.figure(figsize=(12, 6))

# Plot Historical Volatility
plt.plot(hist_vol.index, hist_vol, label='Historical Volatility (20-day)', color='blue', linewidth=2)

# Seamless connection between historical and forecast
conn_idx = [hist_vol.index[-1], forecast_df.index[0]]
conn_val = [hist_vol.iloc[-1], forecast_df['annualized_vol_pct'].iloc[0]]
plt.plot(conn_idx, conn_val, color='red', linestyle='--', linewidth=2)

# Plot Forecast Volatility
plt.plot(forecast_df.index, forecast_df['annualized_vol_pct'], label='GARCH Forecast', color='red', linestyle='--', marker='o', linewidth=2)
plt.fill_between(forecast_df.index, forecast_df['annualized_vol_pct'], alpha=0.2, color='red')

plt.title(f'GJR-GARCH Volatility Forecast (Next {FORECAST_DAYS} Days)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Annualized Volatility (%)', fontsize=12)
plt.axvline(x=hist_vol.index[-1], color='black', linestyle='-.', alpha=0.5, label='Current Date')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'garch_forecast.png'))
plt.close()

# ---------------------------
# 6. Future Regime Probability
# ---------------------------
print(f"\n=== Forecasting Regimes for {FORECAST_DAYS} Days ===")

T = best_model.transmat_
current_state = int(df['hmm_state'].iloc[-1])
state_vec = np.zeros(best_k)
state_vec[current_state] = 1.0

probs = []
for _ in range(FORECAST_DAYS):
    state_vec = state_vec @ T
    probs.append(state_vec.copy())

# Reorder state arrays according to state mapping names
prob_cols = [state_mapping.get(i, f'State_{i}') for i in range(best_k)]
prob_df = pd.DataFrame(probs, index=future_dates, columns=prob_cols)
prob_df.to_csv(os.path.join(OUTPUT_DIR, 'hmm_future_probs.csv'))

# Plot Probabilities
plt.figure(figsize=(12, 5))
for col in prob_cols:
    plt.plot(prob_df.index, prob_df[col], label=col)

plt.title('Future Regime Probabilities')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'hmm_forecast.png'))
plt.close()

print("\n------------------------------------------------")
print(f"Processing Complete. All outputs saved to '{OUTPUT_DIR}/'")
print(f"1. Regime Plot -> regime_plot.png")
print(f"2. Backtest -> strategy_backtest.png")
print(f"3. Volatility Forecast -> garch_forecast.png")
print(f"4. Regime Forecast -> hmm_forecast.png")
print("------------------------------------------------")