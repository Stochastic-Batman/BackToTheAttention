import logging
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger("BackToTheTrainLogger (Train)")


def get_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    data = yf.Ticker(ticker)
    df = data.history(period=period)  # only the latest 'period' stocks
    logger.info("HISTORY:\n%s\n%s", list(df.columns), df.head())  # we want to predict the 'Close' price

    # we don't really care about dividends and stock splits, as they provide no information related to the 'Close' price (most of the time they are 0).
    # I am not a quant, but this is based on my limited knowledge; therefore, we drop those two columns and use the rest for the task.
    df.drop(columns=["Dividends", "Stock Splits"], inplace=True)

    return df


# MinMax scaling is necessary because neural network based architectures, including Transformers,
# train more stably and converge faster when input features are normalized to a consistent range like [0,1]
# rather than using raw stock prices that can vary widely in magnitude.
# large or varying input values can lead to exploding or vanishing gradients during backpropagation,
# particularly in attention mechanisms where dot products between queries and keys become unbalanced and dominate the softmax unfairly.
# scaling also ensures that the model's learned patterns are driven by relative changes in price rather than absolute levels, which is especially helpful for financial time series.

# fit MinMaxScaler (0 to 1) ONLY on training portion to prevent data leakage.
# x_scaled = (x - train_min) / (train_max - train_min)
def minmax_scale(series: np.ndarray, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    n = len(series)
    split_idx = int(n * train_ratio)  # chronological split point
    train_series = series[:split_idx]
    test_series = series[split_idx:]

    # fit scaler only on train data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_series)
    train_scaled = scaler.transform(train_series)
    test_scaled = scaler.transform(test_series)

    logger.info("Train scaled shape: %s, Test scaled shape: %s", train_scaled.shape, test_scaled.shape)
    logger.info("Scaler min: %.4f, scale: %.4f", scaler.min_[0], scaler.scale_[0])

    return train_scaled, test_scaled, scaler


def create_windows(scaled_data: np.ndarray, L: int = 30) -> tuple[np.ndarray, np.ndarray]:
    # sliding window for a sequence x_0, x_1, \cdots, x_{T - 1},
    # creates X_i = [x_{i}, x_{i + 1}, \cdots, x_{i + L - 1}] \in \mathbb{R}^{L \times 1}; y_i = x_{i + L}
    # returns X \in \mathbb{R}^{N \times L \times 1}, y \in \mathbb{R}^{N \times 1} where N = len(scaled_data) - lookback
    if len(scaled_data) <= L:
        raise ValueError("Data length %d must be > lookback %d", len(scaled_data), L)

    X, y = [], []
    for i in range(len(scaled_data) - L):
        X.append(scaled_data[i:i + L])
        y.append(scaled_data[i + L])

    X = np.array(X)  # shape: (N, L, 1)
    y = np.array(y).squeeze(-1)  # shape: (N,) for easier loss computation

    logger.info("Created windows -> X shape: %s, y shape: %s", X.shape, y.shape)

    return X, y


def prepare_datasets(df: pd.DataFrame, L: int = 30, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    series = df['Close'].values.reshape(-1, 1)
    train_scaled, test_scaled, scaler = minmax_scale(series, train_ratio=train_ratio)

    # test windows use ONLY past test data (no train leakage)
    X_train, y_train = create_windows(train_scaled, L=L)
    X_test, y_test = create_windows(test_scaled, L=L)

    logger.info("Final datasets ready -> Train samples: %d, Test samples: %d", len(X_train), len(X_test))

    return X_train, y_train, X_test, y_test, scaler


if __name__ == "__main__":
    df = get_data("PINS")
    X_train, y_train, X_test, y_test, scaler = prepare_datasets(df, L=30, train_ratio=0.8)

    first_window_flat = X_train[0].flatten()
    window_str = " ".join(["{:.4f}".format(val) for val in first_window_flat])
    logger.info("First train window (scaled): %s", window_str)
    logger.info("Corresponding target (scaled): %.4f", y_train[0])
    logger.info("Corresponding target (real price): %.2f", scaler.inverse_transform([[y_train[0]]])[0][0])