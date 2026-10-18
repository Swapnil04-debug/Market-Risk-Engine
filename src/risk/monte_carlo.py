import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "portfolio_returns.csv"
)


def monte_carlo_var_es(
    returns: pd.Series,
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

    returns = returns.dropna()

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    if sigma <= 0:
        raise ValueError("Return volatility must be positive.")

    # Reproducible random-number generator
    rng = np.random.default_rng(seed)

    # Simulate one-day portfolio returns
    simulated_returns = rng.normal(
        loc=mu,
        scale=sigma,
        size=simulations,
    )

    # Convert returns into positive losses
    simulated_losses = -simulated_returns * portfolio_value

    # Estimate VaR
    var = np.quantile(
        simulated_losses,
        confidence_level,
    )

    # Estimate ES using losses at or beyond VaR
    tail_losses = simulated_losses[simulated_losses >= var]
    es = tail_losses.mean()

    return {
        "confidence_level": confidence_level,
        "simulations": simulations,
        "seed": seed,
        "var": float(var),
        "es": float(es),
        "var_pct": float(var / portfolio_value),
        "es_pct": float(es / portfolio_value),
    }


def main():
    data = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    result = monte_carlo_var_es(
        returns=data["portfolio_return"],
        portfolio_value=1_000_000,
        confidence_level=0.99,
        simulations=100_000,
        seed=42,
    )

    print("\nMONTE CARLO RISK REPORT")
    print("-" * 40)

    print(f"Simulations: {result['simulations']:,}")
    print(f"Confidence level: {result['confidence_level']:.0%}")
    print(f"One-day VaR: ${result['var']:,.2f}")
    print(f"One-day ES: ${result['es']:,.2f}")
    print(f"VaR (%): {result['var_pct']:.2%}")
    print(f"ES (%): {result['es_pct']:.2%}")


if __name__ == "__main__":
    main()