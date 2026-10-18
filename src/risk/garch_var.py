import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import norm
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    ROOT / "data" / "processed" / "portfolio_returns.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "garch_var_backtest.csv"
)

PORTFOLIO_VALUE = 1_000_000
CONFIDENCE_LEVEL = 0.99
LOOKBACK_DAYS = 500


def load_returns():
    """Load and validate daily portfolio returns."""

    data = pd.read_csv(
        RETURNS_FILE,
        index_col="Date",
        parse_dates=True,
    )

    if "portfolio_return" not in data.columns:
        raise ValueError(
            "Column 'portfolio_return' was not found."
        )

    returns = data["portfolio_return"]

    if returns.empty:
        raise ValueError("Portfolio returns are empty.")

    if returns.isna().any():
        raise ValueError(
            "Portfolio returns contain missing values."
        )

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError(
            "Portfolio returns contain non-finite values."
        )

    if not returns.index.is_monotonic_increasing:
        raise ValueError(
            "Dates must be in ascending order."
        )

    if not returns.index.is_unique:
        raise ValueError("Duplicate dates found.")

    if len(returns) <= LOOKBACK_DAYS:
        raise ValueError(
            "Insufficient observations for rolling GARCH."
        )

    return returns


def forecast_garch_var(historical_returns):
    """
    Fit a normal GARCH(1,1) model and forecast one-day-ahead VaR.

    Returns VaR in dollars.
    """

    # Scale decimal returns to percentage units for numerical stability.
    scaled_returns = historical_returns * 100

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            ConvergenceWarning,
        )

        fitted_model = model.fit(
            disp="off",
            show_warning=False,
        )

    # Forecast next-day conditional variance.
    forecast = fitted_model.forecast(
        horizon=1,
        reindex=False,
    )

    variance_percent_squared = float(
        forecast.variance.iloc[-1, 0]
    )

    if (
        not np.isfinite(variance_percent_squared)
        or variance_percent_squared <= 0
    ):
        raise ValueError(
            "GARCH produced an invalid variance forecast."
        )

    # Convert volatility from percentage units to decimal units.
    forecast_volatility = (
        np.sqrt(variance_percent_squared) / 100
    )

    # Convert the estimated mean back to decimal return units.
    forecast_mean = (
        float(fitted_model.params["mu"]) / 100
    )

    z_score = norm.ppf(CONFIDENCE_LEVEL)

    var = PORTFOLIO_VALUE * (
        z_score * forecast_volatility - forecast_mean
    )

    return var, forecast_volatility


def main():
    returns = load_returns()

    losses = -returns * PORTFOLIO_VALUE

    results = []
    failed_fits = 0

    print("\nROLLING GARCH(1,1) VaR")
    print("-" * 55)
    print(f"Observations: {len(returns)}")
    print(f"Lookback window: {LOOKBACK_DAYS} days")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")
    print("Fitting rolling GARCH models...")

    for i in range(LOOKBACK_DAYS, len(returns)):
        historical_returns = returns.iloc[
            i - LOOKBACK_DAYS:i
        ]

        actual_loss = losses.iloc[i]
        forecast_date = returns.index[i]

        try:
            var, forecast_volatility = forecast_garch_var(
                historical_returns
            )

        except (
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
            ConvergenceWarning,
        ) as error:
            failed_fits += 1

            print(
                f"Warning: GARCH fit failed on "
                f"{forecast_date.date()}: {error}"
            )

            continue

        exception = int(actual_loss > var)

        results.append({
            "Date": forecast_date,
            "VaR": var,
            "Actual_Loss": actual_loss,
            "Forecast_Volatility": forecast_volatility,
            "Exception": exception,
        })

    if not results:
        raise RuntimeError(
            "No valid GARCH forecasts were generated."
        )

    backtest = pd.DataFrame(results).set_index("Date")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backtest.to_csv(OUTPUT_FILE)

    total_days = len(backtest)
    total_exceptions = int(
        backtest["Exception"].sum()
    )

    exception_rate = total_exceptions / total_days

    print("\nGARCH VaR RESULTS")
    print("-" * 55)
    print(f"Successful forecasts: {total_days}")
    print(f"Failed model fits: {failed_fits}")
    print(f"VaR exceptions: {total_exceptions}")
    print(f"Observed exception rate: {exception_rate:.2%}")
    print(
        f"Expected exception rate: "
        f"{1 - CONFIDENCE_LEVEL:.2%}"
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()