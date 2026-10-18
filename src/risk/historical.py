import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "portfolio_returns.csv"
)


def historical_var_es(
    returns: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99,
) -> dict:
    """
    Calculate one-day Historical Simulation VaR and ES.

    Returns must be decimal simple returns.
    Losses are reported as positive amounts.
    """

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive.")

    returns = returns.dropna()

    if returns.empty:
        raise ValueError("No valid returns supplied.")

    # Convert returns into positive losses and negative gains.
    losses = -returns.to_numpy() * portfolio_value

    # Empirical loss quantile.
    var = float(
        pd.Series(losses).quantile(
            confidence_level,
            interpolation="linear",
        )
    )

    # Empirical Expected Shortfall.
    # Average losses at or beyond the empirical VaR threshold.
    tail_losses = losses[losses >= var]

    es = float(tail_losses.mean())

    return {
        "confidence_level": confidence_level,
        "observations": len(losses),
        "var": var,
        "es": es,
        "var_pct": var / portfolio_value,
        "es_pct": es / portfolio_value,
    }


def main():
    data = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    portfolio_returns = data["portfolio_return"]

    portfolio_value = 1_000_000

    result = historical_var_es(
        returns=portfolio_returns,
        portfolio_value=portfolio_value,
        confidence_level=0.99,
    )

    print("\nHISTORICAL SIMULATION RISK REPORT")
    print("-" * 40)
    print(f"Observations: {result['observations']}")
    print(f"Confidence level: {result['confidence_level']:.0%}")
    print(f"Portfolio value: ${portfolio_value:,.2f}")
    print(f"One-day VaR: ${result['var']:,.2f}")
    print(f"One-day ES:  ${result['es']:,.2f}")
    print(f"VaR (%): {result['var_pct']:.2%}")
    print(f"ES (%):  {result['es_pct']:.2%}")


if __name__ == "__main__":
    main()