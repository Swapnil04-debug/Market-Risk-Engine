import numpy as np
import pandas as pd
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "daily_returns.csv"
)

CONFIG_FILE = PROJECT_ROOT / "config" / "portfolio.yaml"


def multivariate_monte_carlo(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99,
    simulations: int = 100_000,
    seed: int = 42,
) -> dict:

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive.")

    if simulations < 100:
        raise ValueError("Use at least 100 simulations.")

    if asset_returns.empty:
        raise ValueError("Asset returns cannot be empty.")

    if asset_returns.isna().any().any():
        raise ValueError("Missing asset returns detected.")

    if not np.isfinite(asset_returns.to_numpy()).all():
        raise ValueError("Non-finite returns detected.")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Non-finite portfolio weights detected.")

    if (weights < 0).any():
        raise ValueError("Negative weights are not supported.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    if not set(weights.index).issubset(asset_returns.columns):
        raise ValueError("Some portfolio assets are missing from returns.")

    # Align columns with portfolio weights.
    returns = asset_returns.loc[:, weights.index]

    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    # Estimate the asset return distribution.
    mean_returns = returns.mean().to_numpy()
    covariance = returns.cov().to_numpy()

    # Check that covariance is positive semidefinite.
    eigenvalues = np.linalg.eigvalsh(covariance)

    if eigenvalues.min() < -1e-10:
        raise ValueError("Covariance matrix is not positive semidefinite.")

    # Simulate correlated asset returns.
    rng = np.random.default_rng(seed)

    simulated_asset_returns = rng.multivariate_normal(
        mean=mean_returns,
        cov=covariance,
        size=simulations,
        check_valid="raise",
    )

    # Calculate portfolio returns for each simulation.
    simulated_portfolio_returns = (
        simulated_asset_returns @ weights.to_numpy()
    )

    # Convert returns to positive losses.
    losses = -portfolio_value * simulated_portfolio_returns

    # Calculate empirical VaR and ES.
    var = float(np.quantile(losses, confidence_level))

    tail_losses = losses[losses >= var]
    es = float(tail_losses.mean())

    return {
        "simulations": simulations,
        "confidence_level": confidence_level,
        "var": var,
        "es": es,
        "var_pct": var / portfolio_value,
        "es_pct": es / portfolio_value,
    }


def main():
    # Load historical asset returns.
    returns = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    # Load portfolio configuration.
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    portfolio_value = float(
        config["portfolio"]["initial_value"]
    )

    weights = pd.Series({
        asset["ticker"]: float(asset["weight"])
        for asset in config["assets"]
    })

    # Select the configured assets.
    returns = returns.loc[:, weights.index]

    result = multivariate_monte_carlo(
        asset_returns=returns,
        weights=weights,
        portfolio_value=portfolio_value,
        confidence_level=0.99,
        simulations=100_000,
        seed=42,
    )

    print("\nMULTIVARIATE MONTE CARLO RISK REPORT")
    print("-" * 45)
    print(f"Simulations: {result['simulations']:,}")
    print(f"Confidence level: {result['confidence_level']:.0%}")
    print(f"One-day VaR: ${result['var']:,.2f}")
    print(f"One-day ES: ${result['es']:,.2f}")
    print(f"VaR (%): {result['var_pct']:.2%}")
    print(f"ES (%): {result['es_pct']:.2%}")


if __name__ == "__main__":
    main()