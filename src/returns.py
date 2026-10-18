import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRICE_FILE = PROJECT_ROOT / "data" / "raw" / "adjusted_prices.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "daily_returns.csv"


def calculate_returns():
    # Load historical prices
    prices = pd.read_csv(
        PRICE_FILE,
        index_col="Date",
        parse_dates=True
    )

    # Validate the input
    if prices.empty:
        raise ValueError("Price dataset is empty.")

    if prices.isna().any().any():
        raise ValueError("Missing prices detected.")

    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices detected.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted.")

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates detected.")

    # Calculate daily simple returns
    returns = prices.pct_change(fill_method=None).dropna()

    # Validate returns
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates are not sorted.")

    if returns.isna().any().any():
        raise ValueError("Missing returns detected.")

    # Save processed returns
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    returns.to_csv(OUTPUT_FILE, index_label="Date")

    print("Return calculation completed successfully.")
    print(f"\nNumber of observations: {len(returns)}")

    print("\nMean daily returns:")
    print(returns.mean())

    print("\nDaily volatility:")
    print(returns.std())

    print("\nFirst five return observations:")
    print(returns.head())

    print(f"\nSaved to: {OUTPUT_FILE}")

    return returns


if __name__ == "__main__":
    calculate_returns()