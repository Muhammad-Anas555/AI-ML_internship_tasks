import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads historical stock data from Yahoo Finance API and cleans column indexes."""
    print(f"📡 Fetching historical data for {ticker} from {start_date} to {end_date}...")
    raw_data = yf.download(ticker, start=start_date, end=end_date)

    if raw_data.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. Check spelling or dates."
        )

    # Flatten multi-index columns if returned by yfinance
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    # Filter to only the core required market metrics
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    return raw_data[required_cols].copy()