import yfinance as yf
import pandas as pd
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]


def download_market_data():
    all_prices = {}

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")

        for attempt in range(3):
            try:
                data = yf.download(
                    ticker,
                    period="10y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    multi_level_index=False,
                )

                if data.empty or "Close" not in data.columns:
                    raise RuntimeError(f"No valid price data for {ticker}")

                prices = data["Close"].rename(ticker)
                prices.index = pd.to_datetime(prices.index)

                if prices.isna().all():
                    raise RuntimeError(f"All prices are missing for {ticker}")

                all_prices[ticker] = prices
                print(f"Successfully downloaded {ticker}: {len(prices)} rows")
                break

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {ticker}: {e}")

                if attempt < 2:
                    time.sleep(2)

        else:
            raise RuntimeError(
                f"Failed to download {ticker} after 3 attempts. "
                "Stopping to avoid saving an incomplete dataset."
            )

    prices = pd.concat(all_prices.values(), axis=1)
    prices = prices.reindex(columns=TICKERS).sort_index()

    if prices.empty:
        raise RuntimeError("No price data downloaded.")

    # Do not silently fill missing market prices.
    print("\nMissing values:")
    print(prices.isna().sum())

    output_file = RAW_DATA_DIR / "adjusted_prices.csv"
    prices.to_csv(output_file, index_label="Date")

    print(f"\nSaved to: {output_file}")
    print(f"Dataset shape: {prices.shape}")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")

    print("\nFirst five rows:")
    print(prices.head())


if __name__ == "__main__":
    download_market_data()