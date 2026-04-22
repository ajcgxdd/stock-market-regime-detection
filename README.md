# Stock Market "Mood" Detection & Turbulence Forecasting

- If you are not a financial expert, do not worry. This project uses machine learning and statistical models to answer two questions about the Indian Stock market (specifically the NIFTY50):
1. **What “mood” or “season” is the market currently in?** (Is it aggressively going up, crashing down, or just doing nothing?)
2. **How bumpy will the ride be over the next month?** (Will prices swing wildy or stay calm?)


- Just like we have seasons, the stock market has regimes. A “Bull” market means things are generally going up and people are optimistic. A “Bear” market means things are crashing and people are scared. This project uses models called a **Hidden Markov Model (HMM)** to automatically figure out the current season without a human telling what to look for. 

## 🚀 What This Script Does

1. **Reads the Data:** It takes historical stock market data (`nhd.csv`) and the volatility/fear index (`ivd.csv`). 
2. **Finds the Hidden Patterns:** It feeds this data into our AI model to split the market's history into 5 distinct “Stages” or “Regimes” (e.g., Strong Bear, Bear, Neutral, Bull, Strong Bull). 
3. **Tests a Trading Robot:** It simulates a “what if” scenario: *What if a robot traded based purely on these 5 moods?* It buys more when the market is “Bullish” and holds cash (or bets againt the market) when it is “Bearish”, then subtracts fake trading fees to see if it makes a profit. 
4. **Prediciton the Future**: It looks 20 days into the future and predicts how turbulent the market will be and what the most likely mood will be.

---

## 📊 Understanding the Outputs

When you run the script, it generates several files in the `output/` folder. Here is how to read them: 

### 1. The Market Mood Chart (`regime_plot.png`)

- This chart plots the entire history of the stock market, and also colors the line based on what AI thinks the market's moods on the specific day. 
*Different colors indicates different regimes (e.g., Red for Strong Bear/Crashing, Blue/Green for Bull/Rising)*. 
- Helps us to visualise how the AI groups similar markets behaviors together. 

### 2. The Trading Simulation (`strategy_bactest.png`)
This shows a comparison between two strategies: 
- **Grey Line (Buy & Hold):** What happens if you just bought the stock on day and held it for years. 
- **Blue Line (HMM Strategy):** What happens if you actively changed your investments based on the AI's “mood” detection. 

### 3. The Turbulence Forecast (`garch_forecast.png`)
Think of this as weather forecast but for markets: 
- The **Blue Line** shows how bumpy the market has been over the last two months. 
- The **Red Dotted Line** (with shaded red area) shows our model's prediction for how volatile the markets are going to be over the next 20 days. 

### 4. The Future Mood Probabilites (`hmm_forecase.png`)
This chart has 5 lines, representing 5 different market moods. It projects 20 days into the future, showing a percentage of how likely we are to be in each mood. 

### 5. The Data Files (csv)
For those who like raw numbers, the script also outputs the data shown in the images: 
	- `regimes_hmm.csv`: Shows the historical data, with Bollinger Band data, Average True range and the which mood it was in that day. 
	- `garch_future_vol.csv`: The volatility numbers for the next 20 days.
	- `hmm_future_probs.csv`: The probability of being in any given state for the next 20 days.  

---

## 🛠️ How to Run the Code

To run the python script yourself: 

1. Ensure you have Python installed. 
2. Clone the repository 
   ```bash
   git clone https://github.com/ajcgxdd/stock-market-regime-detection.git
   cd stock-market-regime-detection
   ```
3. Install the dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
5. Make sure the dataset (`nhd.csv` and `ihd.csv`) is present in the root directory. And extend it to the present date if required. 
6. Execute the program using: 
   ```bash
   python prac.py
   ```
7. After the execution of the above command, check the `output/` folder for the charts and the numbers.

