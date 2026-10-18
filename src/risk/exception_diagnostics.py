import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BACKTEST_FILE = (
    ROOT / "data" / "processed"
    / "historical_var_backtest.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

CONFIDENCE_LEVEL = 0.99


def load_backtest():
    """Load and validate the rolling VaR backtest results."""

    if not BACKTEST_FILE.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {BACKTEST_FILE}\n"
            "Run historical_backtest.py first."
        )

    data = pd.read_csv(
        BACKTEST_FILE,
        index_col="Date",
        parse_dates=True,
    )

    required_columns = [
        "VaR",
        "Actual_Loss",
        "Exception",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("Backtest data is empty.")

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Backtest data contains missing values."
        )

    if not np.isfinite(
        data[required_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Backtest data contains non-finite values."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Backtest dates must be in ascending order."
        )

    if not data.index.is_unique:
        raise ValueError("Duplicate backtest dates found.")

    if not data["Exception"].isin([0, 1]).all():
        raise ValueError(
            "Exception column must contain only 0 and 1."
        )

    # The existing backtest defines VaR as a loss threshold.
    if (data["VaR"] < 0).any():
        raise ValueError(
            "Negative VaR thresholds found. "
            "Check the loss convention in the backtest."
        )

    # Verify that exception labels agree with actual losses.
    expected_exceptions = (
        data["Actual_Loss"] > data["VaR"]
    ).astype(int)

    if not np.array_equal(
        data["Exception"].to_numpy(dtype=int),
        expected_exceptions.to_numpy(dtype=int),
    ):
        raise ValueError(
            "Exception labels do not match actual losses and VaR."
        )

    return data


def summarize_exceptions(data):
    """Print a summary and save individual exception records."""

    exceptions = data.loc[
        data["Exception"] == 1
    ].copy()

    total_days = len(data)
    total_exceptions = len(exceptions)

    exception_rate = (
        total_exceptions / total_days
        if total_days > 0
        else 0.0
    )

    print("\nVaR EXCEPTION DIAGNOSTICS")
    print("-" * 60)

    print(f"Backtest observations: {total_days}")
    print(f"VaR exceptions:        {total_exceptions}")
    print(f"Observed exception rate: {exception_rate:.2%}")
    print(
        "Expected exception rate: "
        f"{1 - CONFIDENCE_LEVEL:.2%}"
    )

    if not exceptions.empty:
        exceptions["Excess_Loss"] = (
            exceptions["Actual_Loss"] - exceptions["VaR"]
        )

        print(
            f"\nFirst exception date: "
            f"{exceptions.index.min().date()}"
        )

        print(
            f"Last exception date:  "
            f"{exceptions.index.max().date()}"
        )

        print(
            f"Largest actual loss on an exception day: "
            f"${exceptions['Actual_Loss'].max():,.2f}"
        )

        print(
            f"Largest VaR breach: "
            f"${exceptions['Excess_Loss'].max():,.2f}"
        )

        print(
            f"Average excess loss on exception days: "
            f"${exceptions['Excess_Loss'].mean():,.2f}"
        )

        output_file = (
            OUTPUT_DIR / "var_exceptions.csv"
        )

        exceptions.to_csv(output_file)

        print(
            f"\nSaved exception records to: {output_file}"
        )

        print("\nINDIVIDUAL EXCEPTION DATES")
        print("-" * 60)

        display_columns = [
            "VaR",
            "Actual_Loss",
            "Excess_Loss",
        ]

        print(
            exceptions[display_columns].round(2).to_string()
        )

    else:
        print("\nNo VaR exceptions occurred.")

    return exceptions


def plot_var_diagnostics(data):
    """Plot actual losses against the rolling VaR threshold."""

    fig, ax = plt.subplots(figsize=(15, 7))

    # Rolling VaR threshold.
    ax.plot(
        data.index,
        data["VaR"],
        label="99% Historical Simulation VaR",
        linewidth=1.5,
    )

    # Actual daily losses.
    ax.plot(
        data.index,
        data["Actual_Loss"],
        label="Actual Daily Loss",
        linewidth=0.8,
        alpha=0.75,
    )

    # Highlight VaR exceptions.
    exceptions = data[data["Exception"] == 1]

    if not exceptions.empty:
        ax.scatter(
            exceptions.index,
            exceptions["Actual_Loss"],
            label="VaR Exceptions",
            marker="x",
            s=55,
            zorder=3,
        )

    # Zero-loss reference line.
    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_title(
        "Rolling Historical Simulation VaR Backtest"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Loss ($)")

    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR / "var_exception_diagnostics.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nSaved diagnostic chart to: {output_file}")

    plt.show()


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_backtest()

    exceptions = summarize_exceptions(data)

    plot_var_diagnostics(data)


if __name__ == "__main__":
    main()