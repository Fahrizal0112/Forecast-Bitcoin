from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

app = Flask(__name__)
CORS(app)
@app.route('/bitcoin_price', methods=['GET'])
def get_bitcoin_price():
    days = request.args.get('days', default=365, type=int)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    btc_data = yf.Ticker("BTC-USD")
    history = btc_data.history(start=start_date, end=end_date)

    price_data = []
    for date, row in history.iterrows():
        price_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        })

    return jsonify(price_data)

@app.route('/bitcoin_forecast/<int:days>', methods=['GET'])
def get_bitcoin_forecast(days):
    if days <= 0:
        return jsonify({"error": "Days must be a positive integer"}), 400

    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    btc_data = yf.Ticker("BTC-USD")
    history = btc_data.history(start=start_date, end=end_date)

    data = history['Close'].values
    index = pd.date_range(start=start_date, end=end_date, freq='D')
    ts = pd.Series(data, index=index)

    model = ARIMA(ts, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=days)
    forecast_dates = pd.date_range(start=end_date, periods=days+1)[1:]

    forecast_data = []
    for date, price in zip(forecast_dates, forecast):
        forecast_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'predicted_price': float(price)
        })

    return jsonify(forecast_data)

@app.route('/bitcoin_buy_recommendation', methods=['GET'])
def get_bitcoin_buy_recommendation():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    btc_data = yf.Ticker("BTC-USD")
    history = btc_data.history(start=start_date, end=end_date)
    
    history['MA50'] = history['Close'].rolling(window=50).mean()
    history['MA200'] = history['Close'].rolling(window=200).mean()
    
    latest = history.iloc[-1]
    
    if latest['MA50'] > latest['MA200']:
        recommendation = "Buy"
        reason = "The 50-day moving average is above the 200-day moving average, indicating a potential uptrend."
    else:
        recommendation = "Hold"
        reason = "The 50-day moving average is below the 200-day moving average, indicating a potential downtrend."
    
    return jsonify({
        "recommendation": recommendation,
        "reason": reason,
        "current_price": latest['Close'],
        "MA50": latest['MA50'],
        "MA200": latest['MA200'],
        "date": latest.name.strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(debug=True)
