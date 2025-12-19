import logging
import pandas as pd
import yfinance as yf


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger("BackToTheTrainLogger (Train)")


def get_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    data = yf.Ticker(ticker)
    df = data.history(period=period)  # only the latest 'period' stocks
    logger.info("HISTORY:\n%s\n%s", list(df.columns), df.head())  # we want to predict the 'Close' price

    # we don't really care about dividends and stock splits, as they provide no information related to the 'Close' price (most of the time they are 0).
    # I am not a quant, but this is based on my limited knowledge; therefore, we drop those two columns and use the rest for the task.
    df.drop(columns=["Dividends", "Stock Splits"], inplace=True)

    return df


if __name__ == "__main__":
    get_data("PINS")