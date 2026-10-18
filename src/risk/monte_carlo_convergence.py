import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    ROOT / "data" / "processed" / "portfolio_returns.csv"
)


def main():
    data = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    returns = data["portfolio_return"].dropna()

    portfolio_value = 1_000_000
    confidence = 0.99

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    rng = np.random.default_rng(42)

    simulation_counts = [
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        500_000,
    ]

    # Generate one common sample so estimates at larger
    # simulation counts include the smaller samples.
    max_simulations = max(simulation_counts)

    simulated_returns = rng.normal(
        loc=mu,
        scale=sigma,
        size=max_simulations,
    )

    losses = -portfolio_value * simulated_returns

    results = []

    for count in simulation_counts:
        sample_losses = losses[:count]

        var = np.quantile(sample_losses, confidence)

        tail_losses = sample_losses[sample_losses >= var]
        es = tail_losses.mean()

        results.append({
            "Simulations": count,
            "VaR ($)": round(var, 2),
            "ES ($)": round(es, 2),
        })

    results_df = pd.DataFrame(results)

    print("\nMONTE CARLO CONVERGENCE TEST")
    print("-" * 55)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()