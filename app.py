import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, render_template_string
import io
import pandas as pd
import numpy as np
from model_updater import ModelUpdater
from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from macroeconomic_data import MacroeconomicData
from config import Config
from retrain_scheduler import RetrainScheduler

app = Flask(__name__)

data_fetcher = DataFetcher()
data_preprocessor = DataPreprocessor()
model_updater = ModelUpdater()
macro_data = MacroeconomicData(Config.FRED_API_KEY)

# Load the HTML content for the form
try:
    with open('index.html', 'r') as file:
        index_html = file.read()
    print("index.html loaded successfully.")
except Exception as e:
    index_html = None
    print(f"Error loading index.html: {e}")

@app.route('/')
def index():
    if index_html is None:
        return "index.html not found or error loading file.", 404
    return render_template_string(index_html)

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Please provide a stock symbol'}), 400

    # Fetch historical stock data up to today
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    stock_data = data_fetcher.fetch_historical_stock_data(ticker, Config.START_DATE, end_date)
    if stock_data.empty:
        return jsonify({'error': 'Failed to fetch stock data or stock ticker is invalid'}), 400

    stock_data = stock_data.resample('M').last()  # Resample to monthly data
    stock_data_index = stock_data.index

    # Fetching macroeconomic data
    try:
        unemployment_rate = macro_data.fetch_historical_data('UNRATE', Config.START_DATE, end_date, stock_data_index)
        interest_rate = macro_data.fetch_historical_data('FEDFUNDS', Config.START_DATE, end_date, stock_data_index)
        inflation_rate = macro_data.fetch_historical_data('CPIAUCSL', Config.START_DATE, end_date, stock_data_index)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch macroeconomic data: {e}'}), 500

    # Combine data
    combined_data = stock_data.copy()
    combined_data['UnemploymentRate'] = unemployment_rate
    combined_data['InterestRate'] = interest_rate
    combined_data['InflationRate'] = inflation_rate

    # Handle any missing values by filling them forward
    combined_data.fillna(method='ffill', inplace=True)
    combined_data.dropna(inplace=True)

    # Scale data
    scaled_data = data_preprocessor.scale_data(combined_data)

    # Prepare data for prediction
    time_step = Config.TIME_STEP
    if len(scaled_data) < time_step:
        return jsonify({'error': 'Not enough data to create time steps for prediction'}), 400

    X, y = data_preprocessor.create_time_steps(scaled_data)
    if X.size == 0:
        return jsonify({'error': 'Not enough data to make predictions'}), 400

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Predict
    model = model_updater.load_model()
    predictions = model_updater.predict(model, X)

    # Predict future data
    future_predictions = []
    last_data = scaled_data[-time_step:]
    for _ in range(12):  # Predicting for the next 12 months
        X_future = last_data.reshape(1, time_step, combined_data.shape[1])
        future_pred = model_updater.predict(model, X_future)
        future_pred_full = np.full((1, combined_data.shape[1]), future_pred[0][0])
        future_predictions.append(future_pred[0][0])
        last_data = np.vstack([last_data[1:], future_pred_full])

    # Inverse transform to get actual CAD values
    predictions = data_preprocessor.inverse_transform_data(predictions, combined_data)
    future_predictions = data_preprocessor.inverse_transform_data(np.array(future_predictions).reshape(-1, 1), combined_data)
    actual_data = data_preprocessor.inverse_transform_data(scaled_data[-len(predictions):], combined_data)

    return jsonify({
        'actual_data': actual_data.tolist(),
        'predictions': predictions.tolist(),
        'future_predictions': future_predictions.tolist()
    })

@app.route('/plot', methods=['GET'])
def plot():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Please provide a stock ticker symbol'}), 400

    # Fetch historical stock data up to today
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    stock_data = data_fetcher.fetch_historical_stock_data(ticker, Config.START_DATE, end_date)
    if stock_data.empty:
        return jsonify({'error': 'Failed to fetch stock data or stock ticker is invalid'}), 400

    stock_data = stock_data.resample('M').last()  # Resample to monthly data
    stock_data_index = stock_data.index

    # Fetching macroeconomic data
    try:
        unemployment_rate = macro_data.fetch_historical_data('UNRATE', Config.START_DATE, end_date, stock_data_index)
        interest_rate = macro_data.fetch_historical_data('FEDFUNDS', Config.START_DATE, end_date, stock_data_index)
        inflation_rate = macro_data.fetch_historical_data('CPIAUCSL', Config.START_DATE, end_date, stock_data_index)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch macroeconomic data: {e}'}), 500

    # Combine data
    combined_data = stock_data.copy()
    combined_data['UnemploymentRate'] = unemployment_rate
    combined_data['InterestRate'] = interest_rate
    combined_data['InflationRate'] = inflation_rate

    # Handle any missing values by filling them forward
    combined_data.fillna(method='ffill', inplace=True)
    combined_data.dropna(inplace=True)

    # Scale data
    scaled_data = data_preprocessor.scale_data(combined_data)

    # Prepare data for prediction
    time_step = Config.TIME_STEP
    if len(scaled_data) < time_step:
        return jsonify({'error': 'Not enough data to create time steps for prediction'}), 400

    X, y = data_preprocessor.create_time_steps(scaled_data)
    if X.size == 0:
        return jsonify({'error': 'Not enough data to make predictions'}), 400

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Predict
    model = model_updater.load_model()
    predictions = model_updater.predict(model, X)

    # Predict future data
    future_predictions = []
    last_data = scaled_data[-time_step:]
    for _ in range(12):  # Predicting for the next 12 months
        X_future = last_data.reshape(1, time_step, combined_data.shape[1])
        future_pred = model_updater.predict(model, X_future)
        future_pred_full = np.full((1, combined_data.shape[1]), future_pred[0][0])
        future_predictions.append(future_pred[0][0])
        last_data = np.vstack([last_data[1:], future_pred_full])

    # Inverse transform to get actual CAD values
    predictions = data_preprocessor.inverse_transform_data(predictions, combined_data)
    future_predictions = data_preprocessor.inverse_transform_data(np.array(future_predictions).reshape(-1, 1), combined_data)
    actual_data = data_preprocessor.inverse_transform_data(scaled_data[-len(predictions):], combined_data)

    # Ensure x and y have the same length
    if len(combined_data.index) != len(actual_data):
        combined_data = combined_data.iloc[-len(actual_data):]

    # Generate plot
    plt.figure(figsize=(10, 5))
    plt.plot(combined_data.index, actual_data, label='Actual')
    plt.plot(combined_data.index, predictions, label='Predicted', linestyle='dashed')
    future_dates = pd.date_range(combined_data.index[-1], periods=13, freq='M')[1:]
    plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='dotted')
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price in CAD')
    plt.legend()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    retrain_scheduler = RetrainScheduler()
    retrain_scheduler.schedule_retraining()
    app.run(debug=True)
