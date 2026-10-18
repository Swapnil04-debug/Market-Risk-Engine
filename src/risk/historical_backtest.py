import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]

RETURNS_FILE = (
    ROOT / "data" / "processed" / "portfolio_returns.csv"
)

PORTFOLIO_VALUE = 1_000_000
CONFIDENCE_LEVEL = 0.99
LOOKBACK_DAYS = 500
SIGNIFICANCE_LEVEL = 0.05


# --------------------------------------------------
# 1. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_returns():
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

    if returns.isna().any():
        raise ValueError(
            "Portfolio returns contain missing values."
        )

    if returns.empty:
        raise ValueError("Portfolio returns are empty.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError(
            "Portfolio returns contain invalid values."
        )

    if not returns.index.is_monotonic_increasing:
        raise ValueError(
            "Dates must be in ascending order."
        )

    if not returns.index.is_unique:
        raise ValueError("Duplicate dates found.")

    if len(returns) <= LOOKBACK_DAYS:
        raise ValueError(
            "Insufficient data for rolling VaR backtesting."
        )

    return returns


# --------------------------------------------------
# 2. HISTORICAL SIMULATION VaR
# --------------------------------------------------

def calculate_var(historical_losses):
    if len(historical_losses) == 0:
        raise ValueError(
            "Historical loss sample cannot be empty."
        )

    return np.quantile(
        historical_losses,
        CONFIDENCE_LEVEL,
    )


# --------------------------------------------------
# 3. KUPIEC UNCONDITIONAL COVERAGE TEST
# --------------------------------------------------

def kupiec_test(exceptions, confidence_level):
    """
    H0: The exception probability equals 1 - confidence_level.

    Returns:
        LR statistic
        p-value
    """

    exceptions = np.asarray(exceptions)

    if exceptions.ndim != 1 or len(exceptions) == 0:
        raise ValueError(
            "Exceptions must be a non-empty one-dimensional array."
        )

    if not np.isin(exceptions, [0, 1]).all():
        raise ValueError(
            "Exceptions must contain only 0 and 1."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    n = len(exceptions)
    x = int(np.sum(exceptions))

    p_expected = 1 - confidence_level
    p_observed = x / n

    def log_likelihood(probability):
        if probability == 0:
            return 0.0 if x == 0 else -np.inf

        if probability == 1:
            return 0.0 if x == n else -np.inf

        return (
            x * np.log(probability)
            + (n - x) * np.log1p(-probability)
        )

    lr_stat = 2 * (
        log_likelihood(p_observed)
        - log_likelihood(p_expected)
    )

    # Protect against tiny floating-point errors.
    lr_stat = max(0.0, lr_stat)

    p_value = chi2.sf(lr_stat, df=1)

    return lr_stat, p_value


# --------------------------------------------------
# 4. CHRISTOFFERSEN INDEPENDENCE TEST
# --------------------------------------------------

def christoffersen_independence_test(exceptions):
    """
    H0: VaR exceptions are independent over time.
    H1: VaR exceptions are dependent over time.

    Returns:
        LR statistic
        p-value
        n00, n01, n10, n11
    """

    exceptions = np.asarray(exceptions)

    if exceptions.ndim != 1 or len(exceptions) < 2:
        raise ValueError(
            "At least two exception observations are required."
        )

    if not np.isin(exceptions, [0, 1]).all():
        raise ValueError(
            "Exceptions must contain only 0 and 1."
        )

    exceptions = exceptions.astype(int)

    # Consecutive exception transitions.
    previous = exceptions[:-1]
    current = exceptions[1:]

    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    # Transition probabilities.
    denominator_0 = n00 + n01
    denominator_1 = n10 + n11

    p01 = (
        n01 / denominator_0
        if denominator_0 > 0
        else 0.0
    )

    p11 = (
        n11 / denominator_1
        if denominator_1 > 0
        else 0.0
    )

    total_transitions = n00 + n01 + n10 + n11
    total_exceptions = n01 + n11

    p = total_exceptions / total_transitions

    def bernoulli_log_likelihood(
        successes,
        failures,
        probability,
    ):
        if probability == 0:
            return (
                0.0 if successes == 0 else -np.inf
            )

        if probability == 1:
            return (
                0.0 if failures == 0 else -np.inf
            )

        return (
            successes * np.log(probability)
            + failures * np.log1p(-probability)
        )

    # Restricted model:
    # A single exception probability for every transition.
    ll_restricted = bernoulli_log_likelihood(
        total_exceptions,
        total_transitions - total_exceptions,
        p,
    )

    # Unrestricted model:
    # Separate transition probabilities for previous outcomes 0 and 1.
    ll_unrestricted = (
        bernoulli_log_likelihood(n01, n00, p01)
        + bernoulli_log_likelihood(n11, n10, p11)
    )

    lr_stat = 2 * (
        ll_unrestricted - ll_restricted
    )

    # Protect against floating-point errors.
    lr_stat = max(0.0, lr_stat)

    p_value = chi2.sf(lr_stat, df=1)

    return (
        lr_stat,
        p_value,
        n00,
        n01,
        n10,
        n11,
    )


# --------------------------------------------------
# 5. CONDITIONAL COVERAGE TEST
# --------------------------------------------------

def conditional_coverage_test(
    kupiec_lr,
    independence_lr,
):
    """
    H0: Correct unconditional coverage and independence.

    LR_CC = LR_UC + LR_IND

    Asymptotically follows chi-square with 2 degrees of freedom.
    """

    lr_stat = kupiec_lr + independence_lr

    lr_stat = max(0.0, lr_stat)

    p_value = chi2.sf(lr_stat, df=2)

    return lr_stat, p_value


# --------------------------------------------------
# 6. MAIN BACKTEST
# --------------------------------------------------

def main():
    returns = load_returns()

    # Convert portfolio returns into dollar losses.
    # Positive values represent losses; negative values represent gains.
    losses = -returns * PORTFOLIO_VALUE

    results = []

    # Each forecast uses only the previous 500 observations.
    for i in range(LOOKBACK_DAYS, len(losses)):

        historical_losses = losses.iloc[
            i - LOOKBACK_DAYS:i
        ]

        actual_loss = losses.iloc[i]

        var = calculate_var(historical_losses)

        # Exception occurs when actual loss exceeds VaR.
        exception = int(actual_loss > var)

        results.append({
            "Date": losses.index[i],
            "VaR": var,
            "Actual_Loss": actual_loss,
            "Exception": exception,
        })

    backtest = pd.DataFrame(results).set_index("Date")

    total_days = len(backtest)
    total_exceptions = int(
        backtest["Exception"].sum()
    )

    exception_rate = total_exceptions / total_days
    expected_rate = 1 - CONFIDENCE_LEVEL

    exceptions = backtest["Exception"].to_numpy()

    # Kupiec test.
    kupiec_lr, kupiec_p = kupiec_test(
        exceptions,
        CONFIDENCE_LEVEL,
    )

    # Christoffersen independence test.
    (
        independence_lr,
        independence_p,
        n00,
        n01,
        n10,
        n11,
    ) = christoffersen_independence_test(exceptions)

    # Conditional coverage test.
    coverage_lr, coverage_p = conditional_coverage_test(
        kupiec_lr,
        independence_lr,
    )

    # Save daily backtest results.
    output_file = (
        ROOT / "data" / "processed"
        / "historical_var_backtest.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backtest.to_csv(output_file)

    # --------------------------------------------------
    # 7. PRINT RESULTS
    # --------------------------------------------------

    print("\nROLLING HISTORICAL SIMULATION VaR BACKTEST")
    print("-" * 55)

    print(f"Portfolio value:        ${PORTFOLIO_VALUE:,.2f}")
    print(f"Confidence level:       {CONFIDENCE_LEVEL:.0%}")
    print(f"Lookback window:        {LOOKBACK_DAYS} days")
    print(f"Backtest observations:  {total_days}")
    print(f"VaR exceptions:         {total_exceptions}")
    print(f"Observed exception rate: {exception_rate:.2%}")
    print(f"Expected exception rate: {expected_rate:.2%}")

    print("\nKUPIEC UNCONDITIONAL COVERAGE TEST")
    print("-" * 55)

    print(f"LR statistic:           {kupiec_lr:.4f}")
    print(f"P-value:                {kupiec_p:.4f}")

    if kupiec_p < SIGNIFICANCE_LEVEL:
        print(
            "Result: Reject unconditional coverage "
            "at the 5% significance level."
        )
    else:
        print(
            "Result: Do not reject unconditional coverage "
            "at the 5% significance level."
        )

    print("\nCHRISTOFFERSEN INDEPENDENCE TEST")
    print("-" * 55)

    print(f"0 -> 0 transitions:     {n00}")
    print(f"0 -> 1 transitions:     {n01}")
    print(f"1 -> 0 transitions:     {n10}")
    print(f"1 -> 1 transitions:     {n11}")

    print(f"\nLR statistic:           {independence_lr:.4f}")
    print(f"P-value:                {independence_p:.4f}")

    if independence_p < SIGNIFICANCE_LEVEL:
        print(
            "Result: Reject independence; "
            "evidence of exception clustering."
        )
    else:
        print(
            "Result: Do not reject independence; "
            "no significant evidence of exception clustering."
        )

    print("\nCONDITIONAL COVERAGE TEST")
    print("-" * 55)

    print(f"LR statistic:           {coverage_lr:.4f}")
    print(f"P-value:                {coverage_p:.4f}")

    if coverage_p < SIGNIFICANCE_LEVEL:
        print(
            "Result: Reject correct conditional coverage "
            "at the 5% significance level."
        )
    else:
        print(
            "Result: Do not reject correct conditional coverage "
            "at the 5% significance level."
        )

    print(f"\nSaved results to: {output_file}")

    print(
        "\nNote: Failure to reject a null hypothesis "
        "does not prove the model is correct."
    )


if __name__ == "__main__":
    main()