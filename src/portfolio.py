import pandas as pd
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RETURNS_FILE = PROJECT_ROOT / "data" / "processed" / "daily_returns.csv"
CONFIG_FILE = PROJECT_ROOT / "config" / "portfolio.yaml"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "portfolio_returns.csv"


def calculate_portfolio_returns():
    # Load configuration
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    portfolio_config = config["portfolio"]
    asset_config = config["assets"]

    initial_value = float(portfolio_config["initial_value"])

    weights = {
        asset["ticker"]: float(asset["weight"])
        for asset in asset_config
    }

    # Validate portfolio weights
    if not weights:
        raise ValueError("Portfolio contains no assets.")

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Negative weights are not supported yet.")

    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("Portfolio weights must sum to 1.")

    # Load daily asset returns
    returns = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True
    )

    if returns.empty:
        raise ValueError("Return dataset is empty.")

    if returns.isna().any().any():
        raise ValueError("Missing returns detected.")

    # Validate that every configured asset exists
    missing_assets = set(weights) - set(returns.columns)

    if missing_assets:
        raise ValueError(
            f"Missing asset returns: {sorted(missing_assets)}"
        )

    # Use configured portfolio weights
    weights_series = pd.Series(weights)

    asset_returns = returns[weights_series.index]

    # Calculate daily portfolio returns
    portfolio_returns = asset_returns.mul(
        weights_series,
        axis=1
    ).sum(axis=1)

    portfolio_returns.name = "portfolio_return"

    # Calculate one-day P&L for a portfolio of constant value
    # This is a fixed-notional approximation for initial risk modelling.
    pnl = initial_value * portfolio_returns
    pnl.name = "portfolio_pnl"

    # Save results
    output = pd.concat(
        [portfolio_returns, pnl],
        axis=1
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUTPUT_FILE, index_label="Date")

    # Report summary statistics
    print("Portfolio construction completed successfully.")

    print("\nPortfolio weights:")
    print(weights_series)

    print(f"\nInitial portfolio value: ${initial_value:,.2f}")

    print(
        f"\nMean daily portfolio return: "
        f"{portfolio_returns.mean():.6%}"
    )

    print(
        f"Daily portfolio volatility: "
        f"{portfolio_returns.std():.6%}"
    )

    print(
        f"Annualized portfolio volatility: "
        f"{portfolio_returns.std() * (252 ** 0.5):.2%}"
    )

    print(
        f"\nWorst daily portfolio return: "
        f"{portfolio_returns.min():.2%}"
    )

    print(
        f"Worst daily P&L: ${pnl.min():,.2f}"
    )

    print("\nFirst five observations:")
    print(output.head())

    print(f"\nSaved to: {OUTPUT_FILE}")

    return output


if __name__ == "__main__":
    calculate_portfolio_returns()