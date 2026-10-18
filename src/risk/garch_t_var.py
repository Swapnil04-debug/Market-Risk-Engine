import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import t
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    ROOT / "data" / "processed" / "portfolio_returns.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "garch_t_var_backtest.csv"
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
        raise ValueError("Portfolio returns contain missing values.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError(
            "Portfolio returns contain non-finite values."
        )

    if not returns.index.is_monotonic_increasing:
        raise ValueError("Dates must be in ascending order.")

    if not returns.index.is_unique:
        raise ValueError("Duplicate dates found.")

    if len(returns) <= LOOKBACK_DAYS:
        raise ValueError(
            "Insufficient observations for rolling GARCH."
        )

    return returns


def forecast_garch_t_var(historical_returns):
    """
    Fit Student-t GARCH(1,1) and forecast one-day-ahead VaR.

    The arch library's standardized Student-t distribution
    is used for the conditional return innovations.
    """

    # Scale decimal returns to percentage units.
    scaled_returns = historical_returns * 100

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
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

    # Convert forecast volatility to decimal return units.
    forecast_volatility = (
        np.sqrt(variance_percent_squared) / 100
    )

    # Convert estimated mean from percentage to decimal units.
    forecast_mean = (
        float(fitted_model.params["mu"]) / 100
    )

    # Degrees of freedom estimated by the Student-t GARCH model.
    degrees_of_freedom = float(
        fitted_model.params["nu"]
    )

    if degrees_of_freedom <= 2:
        raise ValueError(
            "Student-t degrees of freedom must exceed 2 "
            "for finite variance."
        )

    # scipy's ordinary Student-t has variance nu / (nu - 2).
    # Standardize it to unit variance to match arch's distribution.
    standardized_t_quantile = (
        t.ppf(
            1 - CONFIDENCE_LEVEL,
            df=degrees_of_freedom,
        )
        * np.sqrt(
            (degrees_of_freedom - 2)
            / degrees_of_freedom
        )
    )

    # Convert the 1% return quantile into a positive loss threshold.
    var = PORTFOLIO_VALUE * (
        forecast_mean
        + standardized_t_quantile * forecast_volatility
    ) * -1

    return var, forecast_volatility, degrees_of_freedom


def main():
    returns = load_returns()

    losses = -returns * PORTFOLIO_VALUE

    results = []
    failed_fits = 0

    print("\nROLLING STUDENT-t GARCH(1,1) VaR")
    print("-" * 55)
    print(f"Observations: {len(returns)}")
    print(f"Lookback window: {LOOKBACK_DAYS} days")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")
    print("Fitting rolling Student-t GARCH models...")

    for i in range(LOOKBACK_DAYS, len(returns)):
        historical_returns = returns.iloc[
            i - LOOKBACK_DAYS:i
        ]

        forecast_date = returns.index[i]
        actual_loss = losses.iloc[i]

        try:
            (
                var,
                forecast_volatility,
                degrees_of_freedom,
            ) = forecast_garch_t_var(historical_returns)

        except (
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
            ConvergenceWarning,
        ) as error:
            failed_fits += 1

            print(
                f"Warning: Fit failed on "
                f"{forecast_date.date()}: {error}"
            )
            continue

        exception = int(actual_loss > var)

        results.append({
            "Date": forecast_date,
            "VaR": var,
            "Actual_Loss": actual_loss,
            "Forecast_Volatility": forecast_volatility,
            "Degrees_of_Freedom": degrees_of_freedom,
            "Exception": exception,
        })

    if not results:
        raise RuntimeError(
            "No valid Student-t GARCH forecasts were generated."
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

    print("\nSTUDENT-t GARCH VaR RESULTS")
    print("-" * 55)
    print(f"Successful forecasts: {total_days}")
    print(f"Failed model fits: {failed_fits}")
    print(f"VaR exceptions: {total_exceptions}")
    print(f"Observed exception rate: {exception_rate:.2%}")
    print(
        f"Expected exception rate: "
        f"{1 - CONFIDENCE_LEVEL:.2%}"
    )
    print(
        "Average estimated degrees of freedom: "
        f"{backtest['Degrees_of_Freedom'].mean():.2f}"
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()