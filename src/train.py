import argparse
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from DeTention import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, TensorDataset


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
    # For X_i = [x_{i}, x_{i + 1}, \cdots, x_{i + L - 1}] \in \mathbb{R}^{L \times 1}; y_i = x_{i + L}
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


def prepare_datasets(df: pd.DataFrame, L: int = 30, train_ratio: float = 0.8) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    series = df['Close'].values.reshape(-1, 1)
    train_scaled, test_scaled, scaler = minmax_scale(series, train_ratio=train_ratio)

    # test windows use ONLY past test data (no train leakage)
    X_train, y_train = create_windows(train_scaled, L=L)
    X_test, y_test = create_windows(test_scaled, L=L)

    logger.info("Final datasets ready -> Train samples: %d, Test samples: %d", len(X_train), len(X_test))

    return X_train, y_train, X_test, y_test, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeTention model on stock data")
    parser.add_argument("--ticker", type=str, default="PINS")
    parser.add_argument("--period", type=str, default="3y")
    parser.add_argument("--L", type=int, default=30, help="Lookback window size")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--ff_hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_avg_pool", type=bool, default=True)

    args = parser.parse_args()

    # enforce divisibility constraints
    if args.d_model % args.n_heads != 0:
        logger.info("d_model (%d) not divisible by n_heads (%d) -> resetting d_model to 64 and n_head to 4", args.d_model, args.n_heads)
        args.d_model = 64
        args.n_heads = 4

    if args.ff_hidden_size & 1:
        logger.info("ff_hidden_size (%d) must be even for GLU -> resetting to 256", args.ff_hidden_size)
        args.ff_hidden_size = 256

    df = get_data(args.ticker, period=args.period)
    X_train, y_train, X_test, y_test, scaler = prepare_datasets(df, L=args.L, train_ratio=args.train_ratio)

    # set torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(dim=-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(dim=-1)

    logger.info("Tensors -> X_train: %s, y_train: %s | X_test: %s, y_test: %s", X_train_t.shape, y_train_t.shape, X_test_t.shape, y_test_t.shape)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size, shuffle=False)

    logger.info("DataLoaders -> train batches: %d, test batches: %d", len(train_loader), len(test_loader))

    # Model & training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = DeTention(seq_len=args.L, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, ff_hidden_size=args.ff_hidden_size, dropout=args.dropout, use_avg_pool=args.use_avg_pool).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info("Initialized DeTention -> layers: %d, d_model: %d, heads: %d", args.n_layers, args.d_model, args.n_heads)

    best_test_loss = float('inf')
    patience_counter = 0
    best_path = "models/DeTention_best.pth"
    final_path = "models/DeTention.pth"
    scaler_path = "models/scaler.pkl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            pred = model(batch_x)
            loss = criterion(pred, batch_y.squeeze(dim=-1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # evaluation
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y.squeeze(dim=-1))

                test_loss += loss.item() * batch_x.size(0)

        test_loss /= len(test_loader.dataset)

        logger.info("Epoch %03d | Train Loss: %.4f | Test Loss: %.4f", epoch, train_loss, test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            save_model(model, path=best_path)
            logger.info("New best model saved (test loss: %.4f)", best_test_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered after epoch %d", epoch)
                break

    # load best (temporary) model and save as final, then remove temporary best
    best_model = load_model(path=best_path)
    save_model(best_model, path=final_path)
    os.remove(best_path)

    # Save the scaler for inference
    joblib.dump(scaler, scaler_path)
    logger.info("Training completed. Final model saved as '%s' (temporary best model removed)", final_path)
    logger.info("Scaler saved as '%s'", scaler_path)

    logger.info("Fitting ARIMA baseline for comparison")

    # recompute series, train_series, test_series (since local in minmax_scale)
    series = df['Close'].values.reshape(-1, 1)
    split_idx = int(len(series) * args.train_ratio)
    train_series = series[:split_idx].flatten()
    test_series = series[split_idx:].flatten()

    # fit ARIMA on original train_series (unscaled)
    arima_order = (5, 1, 0)  # p,d,q
    arima_model = ARIMA(train_series, order=arima_order)
    arima_fit = arima_model.fit()

    # ARIMA is autoregressive, so update with actuals
    arima_predictions = []
    history = list(train_series)
    for t in range(len(test_series)):
        arima_temp = ARIMA(history, order=arima_order)
        arima_fit_temp = arima_temp.fit()
        forecast = arima_fit_temp.forecast()[0]
        arima_predictions.append(forecast)
        history.append(test_series[t])

    # DeTention predictions on test set (unscaled)
    model.eval()
    detention_predictions = []
    with torch.no_grad():
        for i in range(0, len(X_test_t), args.batch_size):
            batch_x = X_test_t[i:i + args.batch_size].to(device)
            batch_pred = model(batch_x).cpu().numpy()
            detention_predictions.extend(batch_pred)

    detention_predictions = scaler.inverse_transform(np.array(detention_predictions).reshape(-1, 1)).flatten()
    actual_test = test_series[args.L:]  # metrics (on original scale), align with windowed test set (skips first L in test)

    arima_predictions_aligned = arima_predictions[args.L:]  # Align ARIMA to match
    arima_mse = mean_squared_error(actual_test, arima_predictions_aligned)
    arima_mae = mean_absolute_error(actual_test, arima_predictions_aligned)

    detention_mse = mean_squared_error(actual_test, detention_predictions)
    detention_mae = mean_absolute_error(actual_test, detention_predictions)

    logger.info("ARIMA Test MSE: %.4f | MAE: %.4f", arima_mse, arima_mae)
    logger.info("DeTention Test MSE: %.4f | MAE: %.4f", detention_mse, detention_mae)
    logger.info("DeTention vs ARIMA: MSE Improvement: %.2f%% | MAE Improvement: %.2f%%",100 * (arima_mse - detention_mse) / arima_mse, 100 * (arima_mae - detention_mae) / arima_mae)

    logger.info("Generating comparison plot for test set predictions")

    dates = df.index  # get dates for test set (from original df)
    test_start_idx = split_idx + args.L  # adjust for windowing offset
    test_dates = dates[test_start_idx:test_start_idx + len(actual_test)]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates, actual_test, color='green', label='Actual Price', marker='o', linewidth=2)
    ax.plot(test_dates, detention_predictions, color='purple', label='DeTention Predicted', marker='x', linewidth=2, linestyle='--')
    ax.plot(test_dates, arima_predictions_aligned, color='orange', label='ARIMA Predicted', marker='^', linewidth=2, linestyle=':')
    ax.set_title(f'{args.ticker} Test Set: Actual vs Predictions')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price ($)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = f"models/{args.ticker}_test_comparison_plot.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info("Test set comparison plot saved to %s", plot_path)