import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "garch_var_backtest.csv"
)

CONFIDENCE_LEVEL = 0.99
SIGNIFICANCE_LEVEL = 0.05


def kupiec_test(exceptions):
    n = len(exceptions)
    x = int(np.sum(exceptions))

    p_expected = 1 - CONFIDENCE_LEVEL
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

    lr = 2 * (
        log_likelihood(p_observed)
        - log_likelihood(p_expected)
    )

    lr = max(0.0, lr)

    return lr, chi2.sf(lr, df=1)


def christoffersen_test(exceptions):
    previous = exceptions[:-1]
    current = exceptions[1:]

    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    def log_likelihood(successes, failures, probability):
        if probability == 0:
            return 0.0 if successes == 0 else -np.inf

        if probability == 1:
            return 0.0 if failures == 0 else -np.inf

        return (
            successes * np.log(probability)
            + failures * np.log1p(-probability)
        )

    total = n00 + n01 + n10 + n11
    total_exceptions = n01 + n11

    p = total_exceptions / total

    denominator_0 = n00 + n01
    denominator_1 = n10 + n11

    p01 = n01 / denominator_0 if denominator_0 else 0.0
    p11 = n11 / denominator_1 if denominator_1 else 0.0

    ll_restricted = log_likelihood(
        total_exceptions,
        total - total_exceptions,
        p,
    )

    ll_unrestricted = (
        log_likelihood(n01, n00, p01)
        + log_likelihood(n11, n10, p11)
    )

    lr = max(0.0, 2 * (ll_unrestricted - ll_restricted))

    return lr, chi2.sf(lr, df=1)


def main():
    data = pd.read_csv(
        INPUT_FILE,
        index_col="Date",
        parse_dates=True,
    )

    required = ["VaR", "Actual_Loss", "Exception"]

    if data.empty or data[required].isna().any().any():
        raise ValueError("Backtest data is empty or contains missing values.")

    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and ascending.")

    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError("Backtest data contains non-finite values.")

    if not data["Exception"].isin([0, 1]).all():
        raise ValueError("Exception column must contain only 0 and 1.")

    expected_exceptions = (
        data["Actual_Loss"] > data["VaR"]
    ).astype(int)

    if not np.array_equal(
        data["Exception"].to_numpy(dtype=int),
        expected_exceptions.to_numpy(dtype=int),
    ):
        raise ValueError("Exception labels do not match losses and VaR.")

    exceptions = data["Exception"].to_numpy(dtype=int)

    if len(exceptions) < 2:
        raise ValueError("At least two observations are required.")

    total = len(exceptions)
    count = int(exceptions.sum())

    kupiec_lr, kupiec_p = kupiec_test(exceptions)

    independence_lr, independence_p = (
        christoffersen_test(exceptions)
    )

    coverage_lr = kupiec_lr + independence_lr
    coverage_p = chi2.sf(coverage_lr, df=2)

    print("\nGARCH VaR BACKTEST")
    print("-" * 55)
    print(f"Observations:           {total}")
    print(f"VaR exceptions:         {count}")
    print(f"Observed exception rate: {count / total:.2%}")
    print(f"Expected exception rate: {1 - CONFIDENCE_LEVEL:.2%}")

    print("\nKUPIEC UNCONDITIONAL COVERAGE")
    print("-" * 55)
    print(f"LR statistic:           {kupiec_lr:.4f}")
    print(f"P-value:                {kupiec_p:.4f}")
    print(
        "Result:",
        "Reject coverage at 5%."
        if kupiec_p < SIGNIFICANCE_LEVEL
        else "Do not reject coverage at 5%.",
    )

    print("\nCHRISTOFFERSEN INDEPENDENCE")
    print("-" * 55)
    print(f"LR statistic:           {independence_lr:.4f}")
    print(f"P-value:                {independence_p:.4f}")
    print(
        "Result:",
        "Reject independence at 5%."
        if independence_p < SIGNIFICANCE_LEVEL
        else "Do not reject independence at 5%.",
    )

    print("\nCONDITIONAL COVERAGE")
    print("-" * 55)
    print(f"LR statistic:           {coverage_lr:.4f}")
    print(f"P-value:                {coverage_p:.4f}")
    print(
        "Result:",
        "Reject conditional coverage at 5%."
        if coverage_p < SIGNIFICANCE_LEVEL
        else "Do not reject conditional coverage at 5%.",
    )


if __name__ == "__main__":
    main()