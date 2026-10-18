import pandas as pd
from scipy.stats import norm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "portfolio_returns.csv"
)


def parametric_var_es(
    returns: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99,
) -> dict:

    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive.")

    returns = returns.dropna()

    if len(returns) < 2:
        raise ValueError("At least two observations are required.")

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    if sigma <= 0:
        raise ValueError("Return volatility must be positive.")

    # Standard normal quantile
    z = norm.ppf(confidence_level)

    # Positive loss threshold
    var_return = z * sigma - mu

    # Expected Shortfall for normally distributed returns.
    es_return = (
        sigma * norm.pdf(z) / (1 - confidence_level) - mu
    )

    var = var_return * portfolio_value
    es = es_return * portfolio_value

    return {
        "confidence_level": confidence_level,
        "observations": len(returns),
        "mean_daily_return": mu,
        "daily_volatility": sigma,
        "var": var,
        "es": es,
        "var_pct": var_return,
        "es_pct": es_return,
    }


def main():
    data = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    result = parametric_var_es(
        returns=data["portfolio_return"],
        portfolio_value=1_000_000,
        confidence_level=0.99,
    )

    print("\nPARAMETRIC RISK REPORT")
    print("-" * 40)

    print(f"Observations: {result['observations']}")
    print(f"Daily mean return: {result['mean_daily_return']:.6%}")
    print(f"Daily volatility: {result['daily_volatility']:.6%}")
    print(f"One-day VaR: ${result['var']:,.2f}")
    print(f"One-day ES: ${result['es']:,.2f}")
    print(f"VaR (%): {result['var_pct']:.2%}")
    print(f"ES (%): {result['es_pct']:.2%}")


if __name__ == "__main__":
    main()