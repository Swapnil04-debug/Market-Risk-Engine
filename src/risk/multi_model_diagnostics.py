from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "Historical Simulation": PROCESSED / "historical_var_backtest.csv",
    "Normal GARCH": PROCESSED / "garch_var_backtest.csv",
    "Student-t GARCH": PROCESSED / "garch_t_var_backtest.csv",
}

CONFIDENCE_LEVEL = 0.99
ROLLING_WINDOW = 60


def load_models():
    """Load, validate, and align all model backtests."""

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

        values = df[["VaR", "Actual_Loss"]].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")

        if (df["VaR"] < 0).any():
            raise ValueError(f"{name} contains negative VaR thresholds.")

        df = df.sort_values("Date").set_index("Date")

        models[name] = df

    # Use identical dates for every model.
    common_dates = models["Historical Simulation"].index

    for df in models.values():
        common_dates = common_dates.intersection(df.index)

    common_dates = common_dates.sort_values()

    if len(common_dates) == 0:
        raise ValueError("No common dates across the three models.")

    for name in models:
        models[name] = models[name].loc[common_dates].copy()

        # Recalculate exception flags from the loss convention.
        models[name]["Exception"] = (
            models[name]["Actual_Loss"] > models[name]["VaR"]
        ).astype(int)

    # Actual losses must be identical on common dates.
    reference = models["Historical Simulation"]["Actual_Loss"].to_numpy()

    for name, df in models.items():
        if not np.allclose(
            reference,
            df["Actual_Loss"].to_numpy(),
            rtol=1e-9,
            atol=1e-6,
        ):
            raise ValueError(f"Actual losses do not match for {name}.")

    print(f"Common observations: {len(common_dates)}")
    print(f"Date range: {common_dates.min().date()} to {common_dates.max().date()}")

    return models


def save_exception_summary(models):
    """Save model-specific exception records and a summary table."""

    summary = []
    exception_frames = []

    for name, df in models.items():
        exceptions = df[df["Exception"] == 1].copy()
        exceptions["Model"] = name
        exceptions["Excess_Loss"] = (
            exceptions["Actual_Loss"] - exceptions["VaR"]
        )

        exception_frames.append(
            exceptions.reset_index()[
                ["Date", "Model", "VaR", "Actual_Loss", "Excess_Loss"]
            ]
        )

        count = int(df["Exception"].sum())

        summary.append({
            "Model": name,
            "Observations": len(df),
            "Exceptions": count,
            "Exception_Rate_Pct": 100 * count / len(df),
            "Expected_Rate_Pct": 100 * (1 - CONFIDENCE_LEVEL),
            "Mean_Breach_Severity": (
                exceptions["Excess_Loss"].mean()
                if count else np.nan
            ),
            "Max_Breach_Severity": (
                exceptions["Excess_Loss"].max()
                if count else np.nan
            ),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(PROCESSED / "multi_model_exception_summary.csv", index=False)

    all_exceptions = pd.concat(exception_frames, ignore_index=True)
    all_exceptions.to_csv(PROCESSED / "multi_model_exceptions.csv", index=False)

    print("\nEXCEPTION SUMMARY")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print(f"\nSaved summary: {PROCESSED / 'multi_model_exception_summary.csv'}")
    print(f"Saved exceptions: {PROCESSED / 'multi_model_exceptions.csv'}")


def plot_loss_and_var(models):
    """Plot actual losses, all VaR thresholds, and exception markers."""

    reference = next(iter(models.values()))
    dates = reference.index
    actual_losses = reference["Actual_Loss"]

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(
        dates,
        actual_losses,
        color="black",
        linewidth=0.8,
        alpha=0.65,
        label="Actual Loss",
    )

    for name, df in models.items():
        ax.plot(
            dates,
            df["VaR"],
            linewidth=1.2,
            label=f"{name} VaR",
        )

        exceptions = df[df["Exception"] == 1]

        ax.scatter(
            exceptions.index,
            exceptions["Actual_Loss"],
            s=35,
            marker="x",
            label=f"{name} exceptions",
        )

    ax.axhline(0, linewidth=0.7, alpha=0.5)
    ax.set_title("Three-Model VaR Backtest with Exception Markers")
    ax.set_xlabel("Date")
    ax.set_ylabel("One-Day Loss / VaR ($)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    fig.tight_layout()

    output = PROCESSED / "multi_model_var_exceptions.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


def plot_rolling_exceptions(models):
    """Show the number of exceptions in each rolling 60-day window."""

    fig, ax = plt.subplots(figsize=(16, 6))

    for name, df in models.items():
        rolling_count = (
            df["Exception"]
            .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
            .sum()
        )

        ax.plot(
            df.index,
            rolling_count,
            linewidth=1.2,
            label=name,
        )

    # Expected number under a calibrated 99% VaR model:
    # 60 observations * 1% expected exceptions = 0.6.
    ax.axhline(
        ROLLING_WINDOW * (1 - CONFIDENCE_LEVEL),
        linestyle="--",
        linewidth=1,
        label="Expected count at 1%",
    )

    ax.set_title(f"Rolling {ROLLING_WINDOW}-Trading-Day VaR Exceptions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exceptions in Rolling Window")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    output = PROCESSED / "rolling_var_exceptions.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


def plot_volatility_regimes(models):
    """Compare exception rates in low- and high-volatility periods.

    Volatility regimes are defined using a trailing 60-day standard
    deviation of portfolio returns, with the median as the cutoff.
    The volatility estimate uses only information available up to that
    date, including that day's return. This is descriptive analysis,
    not a forecast or a formal statistical test.
    """

    reference = next(iter(models.values()))

    # Actual losses are negative portfolio P&L, so reverse the sign
    # to recover portfolio returns using the known $1,000,000 notional.
    portfolio_returns = -reference["Actual_Loss"] / 1_000_000

    rolling_volatility = portfolio_returns.rolling(
        ROLLING_WINDOW,
        min_periods=ROLLING_WINDOW,
    ).std()

    valid = rolling_volatility.notna()

    if valid.sum() == 0:
        raise ValueError("Not enough data for rolling volatility regimes.")

    cutoff = rolling_volatility[valid].median()

    regimes = pd.Series(index=reference.index, dtype="object")
    regimes.loc[valid & (rolling_volatility <= cutoff)] = "Low volatility"
    regimes.loc[valid & (rolling_volatility > cutoff)] = "High volatility"

    rows = []

    for name, df in models.items():
        aligned_regimes = regimes.loc[df.index]

        for regime_name in ["Low volatility", "High volatility"]:
            mask = aligned_regimes == regime_name
            n = int(mask.sum())
            exceptions = int(df.loc[mask, "Exception"].sum())

            rows.append({
                "Model": name,
                "Regime": regime_name,
                "Observations": n,
                "Exceptions": exceptions,
                "Exception_Rate_Pct": (
                    100 * exceptions / n if n else np.nan
                ),
            })

    result = pd.DataFrame(rows)
    result.to_csv(PROCESSED / "volatility_regime_comparison.csv", index=False)

    pivot = result.pivot(
        index="Model",
        columns="Regime",
        values="Exception_Rate_Pct",
    )

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.axhline(
        100 * (1 - CONFIDENCE_LEVEL),
        linestyle="--",
        linewidth=1,
        label="Expected exception rate (1%)",
    )

    ax.set_title("VaR Exception Rates by Volatility Regime")
    ax.set_xlabel("Model")
    ax.set_ylabel("Exception Rate (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output = PROCESSED / "volatility_regime_comparison.png"
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()

    print("\nVOLATILITY REGIME SUMMARY")
    print(result.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\nSaved: {output}")
    print(f"Rolling-volatility median cutoff: {cutoff:.6f}")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    models = load_models()

    save_exception_summary(models)
    plot_loss_and_var(models)
    plot_rolling_exceptions(models)
    plot_volatility_regimes(models)

    print("\nMulti-model diagnostics completed successfully.")


if __name__ == "__main__":
    main()
