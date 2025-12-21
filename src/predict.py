# src/predict.py
import argparse
import logging
import yfinance as yf
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from DeTention import DeTention, load_model
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger("BackToThePredictLogger (Predict)")

def get_recent_data(ticker: str, period: str = "3y") -> np.ndarray:
    data = yf.Ticker(ticker)
    df = data.history(period=period)
    df.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors='ignore')
    close_prices = df['Close'].values.reshape(-1, 1)
    return close_prices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next-day stock price using trained DeTention model")
    parser.add_argument("--model_path", type=str, default="models/DeTention.pth", help="Path to the trained model")
    parser.add_argument("--ticker", type=str, default="PINS", help="Stock ticker symbol")
    parser.add_argument("--L", type=int, default=30, help="Lookback window size (must match training)")
    parser.add_argument("--period", type=str, default="3y", help="Historical period to fetch for scaling consistency")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    model.eval()

    logger.info("Fetching recent data for %s", args.ticker)
    series = get_recent_data(args.ticker, period=args.period)

    if len(series) < args.L:
        raise ValueError(f"Not enough data points ({len(series)}) for lookback {args.L}")

    # Fit scaler on the same historical data (consistent with training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(series)  # In practice, use the exact same train portion; here we approximate with full history

    # Take the most recent L days
    recent_window = series[-args.L:].astype(np.float32)
    recent_scaled = scaler.transform(recent_window)
    input_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0)  # (1, L, 1)

    logger.info("Input window shape: %s", input_tensor.shape)

    with torch.no_grad():
        scaled_pred = model(input_tensor).item()

    # Inverse transform to real price
    pred_price = scaler.inverse_transform([[scaled_pred]])[0][0]

    logger.info("Predicted next-day closing price for %s: %.2f", args.ticker, pred_price)
    print(f"\nPredicted next-day closing price for {args.ticker}: ${pred_price:.2f}\n")