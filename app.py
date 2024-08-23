from flask import Flask, jsonify
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

app = Flask(__name__)

@app.route('/bitcoin_price', methods=['GET'])
def get_bitcoin_price():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1*365)

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

if __name__ == '__main__':
    app.run(debug=True)
