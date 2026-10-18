import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = ROOT / "data/processed/daily_returns.csv"
CONFIG_FILE = ROOT / "config/portfolio.yaml"


def main():
    # Load asset returns
    returns = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    # Load portfolio configuration
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    portfolio_value = float(
        config["portfolio"]["initial_value"]
    )

    weights = pd.Series({
        asset["ticker"]: float(asset["weight"])
        for asset in config["assets"]
    })

    returns = returns.loc[:, weights.index]

    # Calculate portfolio returns directly
    portfolio_returns = returns @ weights

    direct_mean = portfolio_returns.mean()
    direct_vol = portfolio_returns.std(ddof=1)

    # Calculate portfolio mean and volatility from covariance
    mean_vector = returns.mean().to_numpy()
    covariance = returns.cov().to_numpy()
    w = weights.to_numpy()

    covariance_mean = float(w @ mean_vector)
    covariance_vol = float(
        np.sqrt(w @ covariance @ w)
    )

    print("\nPORTFOLIO MODEL VALIDATION")
    print("-" * 40)

    print(f"Direct mean return: {direct_mean:.10f}")
    print(f"Covariance mean:    {covariance_mean:.10f}")

    print(f"\nDirect volatility:  {direct_vol:.10f}")
    print(f"Covariance volatility: {covariance_vol:.10f}")

    print("\nMean difference:",
          abs(direct_mean - covariance_mean))

    print("Volatility difference:",
          abs(direct_vol - covariance_vol))

    # Calculate analytical normal VaR and ES
    confidence = 0.99
    z = norm.ppf(confidence)

    analytical_var = portfolio_value * (
        z * covariance_vol - covariance_mean
    )

    analytical_es = portfolio_value * (
        covariance_vol * norm.pdf(z)
        / (1 - confidence)
        - covariance_mean
    )

    print("\nANALYTICAL NORMAL RISK")
    print("-" * 40)
    print(f"99% VaR: ${analytical_var:,.2f}")
    print(f"99% ES:  ${analytical_es:,.2f}")

    # Assertions: these should hold for consistent calculations.
    assert np.isclose(
        direct_mean,
        covariance_mean,
        atol=1e-10,
    )

    assert np.isclose(
        direct_vol,
        covariance_vol,
        atol=1e-10,
    )

    print("\nPASS: Portfolio calculations are consistent.")


if __name__ == "__main__":
    main()