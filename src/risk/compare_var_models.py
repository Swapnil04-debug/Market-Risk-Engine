from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import xlogy


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

CONFIDENCE_LEVEL = 0.99
EXPECTED_EXCEPTION_RATE = 1 - CONFIDENCE_LEVEL

FILES = {
    "Historical Simulation": PROCESSED / "historical_var_backtest.csv",
    "Normal GARCH": PROCESSED / "garch_var_backtest.csv",
    "Student-t GARCH": PROCESSED / "garch_t_var_backtest.csv",
}


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

def load_backtests():
    models = {}

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing backtest file: {path}")

        df = pd.read_csv(path, parse_dates=["Date"])

        required = {"Date", "VaR", "Actual_Loss"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

        if df["Date"].duplicated().any():
            raise ValueError(f"{name} contains duplicate dates.")

        if df[["VaR", "Actual_Loss"]].isna().any().any():
            raise ValueError(f"{name} contains missing risk observations.")

        if not np.isfinite(df[["VaR", "Actual_Loss"]].to_numpy()).all():
            raise ValueError(f"{name} contains non-finite values.")

        if (df["VaR"] < 0).any():
            raise ValueError(f"{name} contains negative VaR thresholds.")

        df = df.sort_values("Date").set_index("Date")
        models[name] = df

    # Restrict the comparison to dates available for every model.
    common_dates = models[list(models)[0]].index

    for df in list(models.values())[1:]:
        common_dates = common_dates.intersection(df.index)

    if len(common_dates) == 0:
        raise ValueError("The models have no common dates.")

    common_dates = common_dates.sort_values()

    for name in models:
        models[name] = models[name].loc[common_dates].copy()

    # Actual portfolio losses must agree across models.
    reference_losses = models["Historical Simulation"]["Actual_Loss"]

    for name, df in models.items():
        if not np.allclose(
            reference_losses.to_numpy(),
            df["Actual_Loss"].to_numpy(),
            rtol=1e-9,
            atol=1e-6,
        ):
            raise ValueError(
                f"Actual losses differ between Historical Simulation "
                f"and {name}."
            )

        # Recalculate exceptions instead of trusting stored flags.
        df["Exception"] = (
            df["Actual_Loss"] > df["VaR"]
        ).astype(int)

    return models


# --------------------------------------------------
# Statistical backtests
# --------------------------------------------------

def kupiec_test(exceptions, confidence=0.99):
    """Kupiec unconditional coverage likelihood-ratio test."""

    x = np.asarray(exceptions, dtype=int)
    n = len(x)
    failures = int(x.sum())
    expected_rate = 1 - confidence

    if n == 0:
        raise ValueError("Cannot backtest an empty series.")

    observed_rate = failures / n

    log_likelihood_null = (
        xlogy(failures, expected_rate)
        + xlogy(n - failures, 1 - expected_rate)
    )

    log_likelihood_alt = (
        xlogy(failures, observed_rate)
        + xlogy(n - failures, 1 - observed_rate)
    )

    lr = max(0.0, -2 * (log_likelihood_null - log_likelihood_alt))
    p_value = chi2.sf(lr, df=1)

    return lr, p_value


def christoffersen_independence_test(exceptions):
    """Christoffersen first-order Markov independence test."""

    x = np.asarray(exceptions, dtype=int)

    if len(x) < 2:
        raise ValueError("At least two observations are required.")

    previous = x[:-1]
    current = x[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    # Under independence, both transition probabilities are equal.
    total_transitions = n00 + n01 + n10 + n11
    total_ones = n01 + n11

    pooled_probability = total_ones / total_transitions

    # Under the alternative, transition probabilities differ.
    n0 = n00 + n01
    n1 = n10 + n11

    p01 = n01 / n0 if n0 else 0.0
    p11 = n11 / n1 if n1 else 0.0

    log_likelihood_null = (
        xlogy(total_ones, pooled_probability)
        + xlogy(total_transitions - total_ones, 1 - pooled_probability)
    )

    log_likelihood_alt = (
        xlogy(n01, p01)
        + xlogy(n00, 1 - p01)
        + xlogy(n11, p11)
        + xlogy(n10, 1 - p11)
    )

    lr = max(0.0, -2 * (log_likelihood_null - log_likelihood_alt))
    p_value = chi2.sf(lr, df=1)

    return lr, p_value, n00, n01, n10, n11


# --------------------------------------------------
# Model comparison
# --------------------------------------------------

def compare_models(models):
    results = []

    for name, df in models.items():
        actual_loss = df["Actual_Loss"].to_numpy()
        var = df["VaR"].to_numpy()
        exceptions = df["Exception"].to_numpy()

        exception_count = int(exceptions.sum())
        observation_count = len(df)
        exception_rate = exception_count / observation_count

        # Severity is calculated only on exception days.
        breach_severity = actual_loss[exceptions == 1] - var[exceptions == 1]

        lr_uc, p_uc = kupiec_test(
            exceptions,
            confidence=CONFIDENCE_LEVEL,
        )

        lr_ind, p_ind, n00, n01, n10, n11 = (
            christoffersen_independence_test(exceptions)
        )

        lr_cc = lr_uc + lr_ind
        p_cc = chi2.sf(lr_cc, df=2)

        results.append({
            "Model": name,
            "Observations": observation_count,
            "Exceptions": exception_count,
            "Exception_Rate_Pct": exception_rate * 100,
            "Expected_Rate_Pct": EXPECTED_EXCEPTION_RATE * 100,
            "Mean_VaR": var.mean(),
            "Mean_Breach_Severity": (
                breach_severity.mean()
                if exception_count else np.nan
            ),
            "Max_Breach_Severity": (
                breach_severity.max()
                if exception_count else np.nan
            ),
            "Kupiec_LR": lr_uc,
            "Kupiec_p_value": p_uc,
            "Independence_LR": lr_ind,
            "Independence_p_value": p_ind,
            "Conditional_Coverage_LR": lr_cc,
            "Conditional_Coverage_p_value": p_cc,
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Combined VaR chart
# --------------------------------------------------

def plot_comparison(models):
    first = next(iter(models.values()))
    dates = first.index
    actual_loss = first["Actual_Loss"]

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(
        dates,
        actual_loss,
        label="Actual Loss",
        color="black",
        linewidth=0.8,
        alpha=0.7,
    )

    for name, df in models.items():
        ax.plot(
            dates,
            df["VaR"],
            label=f"{name} VaR",
            linewidth=1.1,
        )

    ax.axhline(0, linewidth=0.7, alpha=0.5)
    ax.set_title(
        f"VaR Model Comparison — {CONFIDENCE_LEVEL:.0%} Confidence"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("One-Day Loss / VaR ($)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    output_path = PROCESSED / "var_model_comparison.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    models = load_backtests()
    results = compare_models(models)

    summary_path = PROCESSED / "var_model_comparison.csv"
    results.to_csv(summary_path, index=False)

    chart_path = plot_comparison(models)

    print("\nCOMMON-DATE MODEL COMPARISON")
    print("=" * 90)
    print(f"Observations per model: {len(next(iter(models.values())))}")
    print(
        f"Date range: {next(iter(models.values())).index.min().date()} "
        f"to {next(iter(models.values())).index.max().date()}"
    )

    display_columns = [
        "Model",
        "Observations",
        "Exceptions",
        "Exception_Rate_Pct",
        "Mean_Breach_Severity",
        "Max_Breach_Severity",
        "Kupiec_p_value",
        "Independence_p_value",
        "Conditional_Coverage_p_value",
    ]

    print("\n", results[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:,.4f}",
    ))

    print("\nInterpretation at the 5% significance level:")
    for _, row in results.iterrows():
        coverage = (
            "REJECTED"
            if row["Kupiec_p_value"] < 0.05
            else "not rejected"
        )
        independence = (
            "REJECTED"
            if row["Independence_p_value"] < 0.05
            else "not rejected"
        )
        conditional = (
            "REJECTED"
            if row["Conditional_Coverage_p_value"] < 0.05
            else "not rejected"
        )

        print(
            f"- {row['Model']}: "
            f"coverage {coverage}; "
            f"independence {independence}; "
            f"conditional coverage {conditional}."
        )

    print(f"\nSaved comparison table: {summary_path}")
    print(f"Saved comparison chart: {chart_path}")


if __name__ == "__main__":
    main()