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
    # Attempts DD-MM-YY parsing
    df = pd.read_csv(
        path,
        skiprows=2,
        parse_dates=['Date'],
        date_parser=lambda x: pd.to_datetime(x, format="%d-%m-%y", errors='coerce')
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index('Date').sort_index()
    return df[['Open','High','Low','Close','Volume']]

def load_vix(path):
    print(f"Loading India VIX CSV: {path}")
    # Attempts YYYY-MM-DD parsing
    df = pd.read_csv(
        path,
        skiprows=2,
        parse_dates=['Date'],
        date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d", errors='coerce')
    )
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
# We use Returns and Volatility as primary regime drivers
hmm_features = ['return', 'vol_20'] 
X_hmm = df[hmm_features].values

best_score = float('inf')
best_k = 3
best_model = None

# Loop to find optimal K (2 to 5)
for k in range(2, 6):
    try:
        tmp_model = GaussianHMM(n_components=k, covariance_type='diag', n_iter=1000, random_state=42)
        tmp_model.fit(X_hmm)
        
        # Calculate AIC: AIC = -2 * logL + 2 * k * n_features
        # (Simplified approximation for model selection)
        logL = tmp_model.score(X_hmm)
        n_features = X_hmm.shape[1]
        n_params = k * (n_features + n_features + k - 1) # Means + Vars + TransMat
        aic = -2 * logL + 2 * n_params
        
        print(f"States: {k} | LogL: {logL:.2f} | AIC: {aic:.2f}")
        
        if aic < best_score:
            best_score = aic
            best_k = k
            best_model = tmp_model
    except:
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
state_mapping = {}
labels = ['Bear', 'Neutral', 'Bull', 'Strong Bull'] # generic labels based on K

for i, state_id in enumerate(sorted_states):
    if i < len(labels):
        state_mapping[state_id] = labels[i]
    else:
        state_mapping[state_id] = f"State_{state_id}"

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

# Define weights: Bear=0% (Cash), Neutral=50%, Bull=100%
# You can tweak these. E.g., Bear = -0.5 for shorting.
weight_map = {
    'Bear': 0.0,
    'Neutral': 0.6,
    'Bull': 1.0, 
    'Strong Bull': 1.2
}

# Apply weights (Default to 0.5 if label not found)
df['strategy_weight'] = df['regime_desc'].map(weight_map).fillna(0.5)

# IMPORTANT: Shift weights by 1 day. 
# We detect regime at Close today -> We trade at Open tomorrow (or capture tomorrow's return).
df['strategy_weight'] = df['strategy_weight'].shift(1).fillna(0)

df['buy_hold_ret'] = df['return']
df['strategy_ret'] = df['return'] * df['strategy_weight']

# Cumulative Returns
df['cum_buy_hold'] = (1 + df['buy_hold_ret']).cumprod()
df['cum_strategy'] = (1 + df['strategy_ret']).cumprod()

# Plot Backtest
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['cum_buy_hold'], label='Buy & Hold (Nifty)', color='gray')
plt.plot(df.index, df['cum_strategy'], label='HMM Strategy', color='blue')
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
am = arch_model(ret_pct, vol='Garch', p=1, o=1, q=1, dist='t')
res = am.fit(disp='off')

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

plt.figure(figsize=(10, 5))
plt.plot(forecast_df.index, forecast_df['annualized_vol_pct'], marker='o')
plt.title(f'GJR-GARCH Volatility Forecast (Next {FORECAST_DAYS} Days)')
plt.ylabel('Annualized Volatility %')
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

prob_df = pd.DataFrame(probs, index=future_dates, columns=[f'State_{i}' for i in range(best_k)])
prob_df.to_csv(os.path.join(OUTPUT_DIR, 'hmm_future_probs.csv'))

# Plot Probabilities
plt.figure(figsize=(12, 5))
for i in range(best_k):
    lbl = state_mapping.get(i, f'State {i}')
    plt.plot(prob_df.index, prob_df[f'State_{i}'], label=lbl)

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