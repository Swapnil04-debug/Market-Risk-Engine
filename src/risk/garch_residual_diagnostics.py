from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch import arch_model
from scipy.stats import jarque_bera, skew, kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox


ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    ROOT / "data" / "processed" / "portfolio_returns.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

LOOKBACK_DAYS = 500
LJUNG_BOX_LAGS = [5, 10, 20]


def load_returns():
    data = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    if "portfolio_return" not in data.columns:
        raise ValueError("portfolio_return column not found.")

    returns = data["portfolio_return"]

    if returns.isna().any() or not np.isfinite(returns).all():
        raise ValueError("Returns contain missing or non-finite values.")

    if not returns.index.is_monotonic_increasing:
        raise ValueError("Dates must be in ascending order.")

    if not returns.index.is_unique:
        raise ValueError("Duplicate dates found.")

    if len(returns) < LOOKBACK_DAYS:
        raise ValueError("Insufficient data for the selected window.")

    return returns


def fit_model(returns, distribution):
    """Fit one GARCH(1,1) model and return standardized residuals."""

    scaled_returns = returns * 100

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist=distribution,
        rescale=False,
    )

    fitted = model.fit(
        disp="off",
        show_warning=False,
    )

    if fitted.convergence_flag != 0:
        raise RuntimeError(
            f"{distribution} GARCH did not converge."
        )

    standardized_residuals = pd.Series(
        fitted.std_resid,
        index=returns.index,
    ).dropna()

    return fitted, standardized_residuals


def diagnose_residuals(name, fitted, residuals):
    """Calculate distribution and autocorrelation diagnostics."""

    squared_residuals = residuals ** 2

    lb_residuals = acorr_ljungbox(
        residuals,
        lags=LJUNG_BOX_LAGS,
        return_df=True,
    )

    lb_squared = acorr_ljungbox(
        squared_residuals,
        lags=LJUNG_BOX_LAGS,
        return_df=True,
    )

    jb = jarque_bera(residuals)

    print(f"\n{name} GARCH(1,1)")
    print("=" * 65)

    print(f"Observations: {len(residuals)}")
    print(f"Convergence flag: {fitted.convergence_flag}")
    print(f"Estimated mean parameter: {fitted.params['mu']:.6f}")
    print(f"Estimated omega: {fitted.params['omega']:.6f}")
    print(f"Estimated alpha[1]: {fitted.params['alpha[1]']:.6f}")
    print(f"Estimated beta[1]: {fitted.params['beta[1]']:.6f}")
    print(f"Alpha + beta: {fitted.params['alpha[1]'] + fitted.params['beta[1]']:.6f}")

    if "nu" in fitted.params:
        print(f"Estimated degrees of freedom: {fitted.params['nu']:.4f}")

    print(f"Residual skewness: {skew(residuals):.4f}")
    print(f"Residual excess kurtosis: {kurtosis(residuals):.4f}")
    print(f"Jarque-Bera statistic: {jb.statistic:.4f}")
    print(f"Jarque-Bera p-value: {jb.pvalue:.6g}")

    print("\nLjung-Box: standardized residuals")
    print(lb_residuals.to_string(float_format=lambda x: f"{x:.6g}"))

    print("\nLjung-Box: squared standardized residuals")
    print(lb_squared.to_string(float_format=lambda x: f"{x:.6g}"))

    return residuals, lb_residuals, lb_squared


def plot_residuals(results):
    """Save residual and squared-residual autocorrelation diagnostics."""

    fig, axes = plt.subplots(
        len(results),
        2,
        figsize=(14, 5 * len(results)),
    )

    if len(results) == 1:
        axes = np.array([axes])

    for row, (name, residuals, _, _) in enumerate(results):
        axes[row, 0].plot(
            residuals.index,
            residuals,
            linewidth=0.7,
        )
        axes[row, 0].axhline(0, linestyle="--", linewidth=0.8)
        axes[row, 0].set_title(f"{name}: Standardized Residuals")
        axes[row, 0].set_xlabel("Date")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(
            residuals.index,
            residuals ** 2,
            linewidth=0.7,
        )
        axes[row, 1].set_title(f"{name}: Squared Standardized Residuals")
        axes[row, 1].set_xlabel("Date")
        axes[row, 1].grid(alpha=0.25)

    fig.tight_layout()

    output = OUTPUT_DIR / "garch_residual_diagnostics.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved diagnostic plot: {output}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()

    # Use the latest 500 observations as one representative window.
    # This is a diagnostic sample, not the full rolling backtest.
    sample = returns.iloc[-LOOKBACK_DAYS:]

    print("GARCH RESIDUAL DIAGNOSTICS")
    print("-" * 65)
    print(f"Sample start: {sample.index.min().date()}")
    print(f"Sample end:   {sample.index.max().date()}")
    print(f"Sample size:  {len(sample)}")

    results = []

    for name, distribution in [
        ("Normal", "normal"),
        ("Student-t", "t"),
    ]:
        fitted, residuals = fit_model(sample, distribution)

        residuals, lb_residuals, lb_squared = diagnose_residuals(
            name,
            fitted,
            residuals,
        )

        results.append(
            (name, residuals, lb_residuals, lb_squared)
        )

    plot_residuals(results)


if __name__ == "__main__":
    main()
