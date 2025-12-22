import argparse
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import yfinance as yf

from DeTention import DeTention, load_model
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger("BackToThePredictLogger (Predict)")


def get_recent_data(ticker: str, period: str = "3y") -> tuple[np.ndarray, list]:
    data = yf.Ticker(ticker)
    df = data.history(period=period)
    df.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors='ignore')
    close_prices = df['Close'].values.reshape(-1, 1)
    dates = df.index.tolist()
    return close_prices, dates


# test the model by predicting today's price and comparing with actual
def test_prediction(model: DeTention, scaler: MinMaxScaler, ticker: str = "PINS", L: int = 30, period: str = "3y"):
    logger.info("TESTING MODEL ON RECENT DATA")

    series, dates = get_recent_data(ticker, period=period)

    if len(series) < L + 1:
        raise ValueError(f"Not enough data points ({len(series)}) for lookback {L} + 1 for testing")

    # use the last L + 1 points: L for input, 1 for actual target
    test_window = series[-(L + 1):-1]  # last L days (excluding most recent)
    actual_price = series[-1, 0]  # most recent day (today's actual close)
    actual_date = dates[-1]

    test_scaled = scaler.transform(test_window)
    input_tensor = torch.tensor(test_scaled, dtype=torch.float32).unsqueeze(0)  # (1, L, 1)

    with torch.no_grad():
        scaled_pred = model(input_tensor).item()

    predicted_price = scaler.inverse_transform([[scaled_pred]])[0][0]
    error = abs(predicted_price - actual_price)
    error_pct = (error / actual_price) * 100

    logger.info(f"Date: {actual_date.strftime('%Y-%m-%d')}")
    logger.info(f"Actual closing price: ${actual_price:.2f}")
    logger.info(f"Predicted price: ${predicted_price:.2f}")
    logger.info(f"Absolute error: ${error:.2f}")
    logger.info(f"Percentage error: {error_pct:.2f}%")

    return {
        'date': actual_date,
        'actual': actual_price,
        'predicted': predicted_price,
        'error': error,
        'error_pct': error_pct
    }


def predict_next_day(model: DeTention, scaler: MinMaxScaler, ticker: str = "PINS", L: int = 30, period: str = "3y"):
    logger.info("PREDICTING NEXT TRADING DAY")

    series, dates = get_recent_data(ticker, period=period)

    if len(series) < L:
        raise ValueError(f"Not enough data points ({len(series)}) for lookback {L}")

    # use the most recent L days
    recent_window = series[-L:]
    last_date = dates[-1]
    input_tensor = torch.tensor(scaler.transform(recent_window), dtype=torch.float32).unsqueeze(0)  # (1, L, 1)

    with torch.no_grad():
        scaled_pred = model(input_tensor).item()

    pred = scaler.inverse_transform([[scaled_pred]])[0][0]

    logger.info(f"Last trading day: {last_date.strftime('%Y-%m-%d')}")
    logger.info(f"Last closing price: ${series[-1, 0]:.2f}")
    logger.info(f"Predicted next-day closing price: ${pred:.2f}")

    return pred


# I used Gemini 3 Pro for writing this method, I suck at visualization
def visualize_recent_predictions(model: DeTention, scaler: MinMaxScaler, series: np.ndarray, dates: list, ticker: str = "PINS", L: int = 30, lookback_days: int = 30):
    logger.info("VISUALIZING RECENT PREDICTIONS OVER PAST %d DAYS", lookback_days)

    if len(series) < L + lookback_days:
        logger.warning("Not enough data for %d-day visualization with L=%d; skipping", lookback_days, L)
        return

    recent_series = series[-(L + lookback_days):]
    recent_dates = dates[-(lookback_days):]

    actual_prices = recent_series[-lookback_days:, 0]
    predicted_prices = []

    for i in range(lookback_days):
        window = recent_series[i:i + L]
        scaled_window = scaler.transform(window)
        input_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)  # (1, L, 1)
        with torch.no_grad():
            scaled_pred = model(input_tensor).item()
        pred = scaler.inverse_transform([[scaled_pred]])[0][0]
        predicted_prices.append(pred)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(recent_dates, actual_prices, color='green', label='Actual Price', marker='o', linewidth=2)
    ax.plot(recent_dates, predicted_prices, color='purple', label='Predicted Price', marker='x', linewidth=2, linestyle='--')
    ax.set_title(f'{ticker} Stock Price: Actual vs Predicted (Last {lookback_days} Days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price ($)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = f"models/{ticker}_prediction_plot.png"
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    logger.info("Prediction plot saved to %s", plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stock prices using trained DeTention model")
    parser.add_argument("--model_path", type=str, default="models/DeTention.pth", help="Path to the trained model")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl", help="Path to the saved scaler")
    parser.add_argument("--ticker", type=str, default="PINS", help="Stock ticker symbol")
    parser.add_argument("--L", type=int, default=30, help="Lookback window size (must match training)")
    parser.add_argument("--period", type=str, default="3y", help="Historical period to fetch")
    parser.add_argument("--test", action="store_true", default=True, help="Run test on most recent actual data")
    parser.add_argument("--predict", action="store_true", default=True, help="Predict next trading day")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    if not os.path.exists(args.scaler_path):
        raise FileNotFoundError(f"Scaler not found: {args.scaler_path}")

    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    model.eval()

    logger.info("Loading scaler from %s", args.scaler_path)
    scaler = joblib.load(args.scaler_path)

    if args.test:
        test_results = test_prediction(model, scaler, ticker=args.ticker, L=args.L, period=args.period)
        logger.info("TEST RESULTS FOR %s", args.ticker)
        logger.info("Date: %s", test_results['date'].strftime('%Y-%m-%d'))
        logger.info("Actual Price: $%.3f", test_results['actual'])
        logger.info("Predicted Price: $%.3f", test_results['predicted'])
        logger.info("Error: $%.3f (%.2f%%)", test_results['error'], test_results['error_pct'])

    if args.predict:
        next_price = predict_next_day(model, scaler, ticker=args.ticker, L=args.L, period=args.period)
        logger.info("NEXT-DAY PREDICTION FOR %s", args.ticker)
        logger.info("Predicted Closing Price: $%.3f", next_price)

        # fetch data once and reuse for visualization (from predict_next_day or test)
        series, dates = get_recent_data(args.ticker, period=args.period)
        visualize_recent_predictions(model, scaler, series, dates, ticker=args.ticker, L=args.L)