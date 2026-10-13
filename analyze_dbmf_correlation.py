"""Calculate and plot DBMF correlations against SPX and IEF."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram
from scipy.stats import norm


INPUT_FILE = Path(__file__).with_name("DBMF_COMP.csv")
OUTPUT_DIR = Path(__file__).with_name("results")
OUTPUT_CYCLE_PLOT = OUTPUT_DIR / "DBMF_COMP_cycle_analysis.png"
DBMF_COLUMN = "DBMF"
COMPARISON_COLUMNS = {
    "SPX": "SPX",
    "IEF": "IEF",
}
ROLLING_WINDOWS = {
    "60 trading days": 60,
    "1 year (252 trading days)": 252,
}
ROLLING_MONTH_COUNT = 12
ANNUAL_RISK_FREE_RATE = 0.03
TRADING_DAYS_PER_YEAR = 252
CYCLE_MIN_PERIOD = 10
CYCLE_MAX_PERIOD = 252
CONFIDENCE_LEVEL = 0.95


def load_prices():
    data = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    required_columns = ["Date", DBMF_COLUMN, *COMPARISON_COLUMNS.values()]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing_columns)}")

    data = data.sort_values("Date").set_index("Date")
    prices = data[[DBMF_COLUMN, *COMPARISON_COLUMNS.values()]].apply(
        pd.to_numeric, errors="coerce"
    )
    if prices.dropna(how="all").empty:
        raise ValueError("The CSV does not contain any valid prices.")
    return prices


def load_returns(prices=None):
    if prices is None:
        prices = load_prices()
    returns = prices.pct_change(fill_method=None)
    if returns.dropna(how="all").empty:
        raise ValueError("The CSV does not contain enough valid prices to calculate returns.")
    return returns


def calculate_performance_metrics(prices, returns):
    """Calculate annualized Sharpe and maximum drawdown for each price series."""
    daily_risk_free_rate = (1 + ANNUAL_RISK_FREE_RATE) ** (
        1 / TRADING_DAYS_PER_YEAR
    ) - 1
    metrics = {}
    for ticker in [DBMF_COLUMN, *COMPARISON_COLUMNS.values()]:
        asset_returns = returns[ticker].dropna()
        excess_returns = asset_returns - daily_risk_free_rate
        return_volatility = excess_returns.std(ddof=1)
        sharpe = (
            np.sqrt(TRADING_DAYS_PER_YEAR) * excess_returns.mean() / return_volatility
            if return_volatility > 0
            else np.nan
        )

        initial_date = prices[ticker].first_valid_index()
        wealth = (1 + asset_returns).cumprod()
        wealth = pd.concat([pd.Series([1.0], index=[initial_date]), wealth])
        drawdown = wealth / wealth.cummax() - 1
        metrics[ticker] = {
            "sharpe_ratio": sharpe,
            "maximum_drawdown": drawdown.min(),
        }
    return pd.DataFrame.from_dict(metrics, orient="index")


def calculate_rolling_correlations(returns):
    rolling = pd.DataFrame(index=returns.index)
    for ticker, column in COMPARISON_COLUMNS.items():
        pair = returns[[DBMF_COLUMN, column]].dropna()
        x = pair[DBMF_COLUMN]
        for label, window in ROLLING_WINDOWS.items():
            key = f"{ticker} {label}"
            correlation = x.rolling(window).corr(pair[column])
            sample_size = x.rolling(window).count()
            clipped = correlation.clip(-1 + 1e-12, 1 - 1e-12)
            fisher_z = np.arctanh(clipped)
            margin = norm.ppf(0.5 + CONFIDENCE_LEVEL / 2) / np.sqrt(sample_size - 3)

            rolling[key] = correlation
            rolling[f"{key} lower"] = np.tanh(fisher_z - margin)
            rolling[f"{key} upper"] = np.tanh(fisher_z + margin)
    return rolling


def calculate_rolling_monthly_return_correlation(monthly_returns, ticker):
    """Calculate a 12-observation rolling correlation of month-end returns."""
    pair_returns = monthly_returns[[DBMF_COLUMN, ticker]].dropna()
    return pair_returns[DBMF_COLUMN].rolling(ROLLING_MONTH_COUNT).corr(
        pair_returns[ticker]
    )


def plot_correlation_series(series, ticker, title, filename, description):
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(series.index, series, color="#176B73", linewidth=1.4)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title(title)
    axis.set_ylabel("Return correlation")
    axis.set_xlabel("Date")
    axis.set_ylim(-1, 1)
    axis.grid(True, alpha=0.25)
    figure.text(0.5, 0.015, description, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))

    output_path = OUTPUT_DIR / filename
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {ticker} correlation figure to {output_path}")


def plot_rebased_prices(prices):
    """Plot DBMF, SPX, and IEF prices rebased to 100 at their first observation."""
    figure, axis = plt.subplots(figsize=(13, 6))
    colors = {DBMF_COLUMN: "#176B73", "SPX": "#C45736", "IEF": "#5268A5"}
    base_prices = prices.apply(lambda column: column.dropna().iloc[0])
    normalized_prices = prices.div(base_prices, axis="columns") * 100
    for ticker in [DBMF_COLUMN, *COMPARISON_COLUMNS.values()]:
        axis.plot(
            normalized_prices.index,
            normalized_prices[ticker],
            color=colors[ticker],
            linewidth=1.2,
            label=ticker,
        )

    axis.axhline(100, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("DBMF, SPX, and IEF Prices (Rebased to 100)")
    axis.set_ylabel("Price index (first observation = 100)")
    axis.set_xlabel("Date")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()

    output_path = OUTPUT_DIR / "DBMF_COMP_rebased_prices.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved rebased price comparison figure to {output_path}")


def save_rolling_correlation_csvs(rolling):
    """Save each comparison's two rolling correlations with their dates."""
    for ticker in COMPARISON_COLUMNS:
        output = pd.DataFrame(
            {
                "Date": rolling.index,
                "60 day rolling": rolling[f"{ticker} 60 trading days"].to_numpy(),
                "1 year rolling": rolling[
                    f"{ticker} 1 year (252 trading days)"
                ].to_numpy(),
            }
        )
        output_path = OUTPUT_DIR / f"DBMF_COMP_{ticker}_rolling_correlations.csv"
        output.to_csv(output_path, index=False)
        print(f"Saved {ticker} rolling correlations to {output_path}")


def plot_rolling_correlations(rolling, ticker):
    """Plot one index's rolling correlations and Fisher confidence bands."""
    figure, correlation_axis = plt.subplots(figsize=(12, 6.5))

    for label in ROLLING_WINDOWS:
        key = f"{ticker} {label}"
        correlation = rolling[key]
        line, = correlation_axis.plot(
            correlation.index,
            correlation,
            linewidth=1.4,
            label=f"{label} correlation",
        )
        correlation_axis.fill_between(
            correlation.index,
            rolling[f"{key} lower"],
            rolling[f"{key} upper"],
            color=line.get_color(),
            alpha=0.14,
            label=f"{label} {CONFIDENCE_LEVEL:.0%} Fisher CI",
        )

    correlation_axis.set_title(f"DBMF vs {ticker}: Rolling Correlation")
    correlation_axis.set_ylabel("Rolling return correlation")
    correlation_axis.set_ylim(-1, 1)
    correlation_axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    correlation_axis.set_xlabel("Date")
    correlation_axis.grid(True, alpha=0.25)

    handles, labels = correlation_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=2,
    )
    figure.text(
        0.5,
        0.015,
        "Shading shows approximate Fisher confidence bands; serial dependence may make them too narrow.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.14, 1, 0.93))

    output_path = OUTPUT_DIR / f"DBMF_COMP_{ticker}_rolling_correlations.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {ticker} rolling correlation chart to {output_path}")


def analyze_correlation_cycles(series, min_period=10, max_period=252, top_n=3):
    """Return a detrended periodogram and its strongest peaks."""
    values = series.dropna().to_numpy(dtype=float)
    max_period = min(max_period, len(values) // 2)
    if max_period < min_period * 2:
        return {"periods": np.array([]), "relative_power": np.array([]), "peaks": []}

    frequencies, power = periodogram(
        values, fs=1.0, window="hann", detrend="linear", scaling="spectrum"
    )
    positive = frequencies > 0
    periods = 1 / frequencies[positive]
    power = power[positive]
    in_range = (periods >= min_period) & (periods <= max_period)
    periods, power = periods[in_range], power[in_range]
    if not len(power):
        return {"periods": np.array([]), "relative_power": np.array([]), "peaks": []}

    peaks, _ = find_peaks(power)
    strongest = int(np.argmax(power))
    if strongest not in peaks:
        peaks = np.append(peaks, strongest)
    peaks = peaks[np.argsort(power[peaks])[::-1]][:top_n]
    total_band_power = power.sum()
    peak_summaries = [
        {
            "period_trading_days": float(periods[index]),
            "band_power_share": float(power[index] / total_band_power),
            "relative_power": float(power[index] / power.max()),
        }
        for index in peaks
    ]
    return {
        "periods": periods,
        "relative_power": power / power.max(),
        "peaks": peak_summaries,
    }


def plot_cycle_analysis(cycle_analyses):
    """Plot relative spectral power against possible cycle lengths."""
    fig, axes = plt.subplots(1, len(cycle_analyses), figsize=(12, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (ticker, analysis) in zip(axes, cycle_analyses.items()):
        periods = analysis["periods"]
        relative_power = analysis["relative_power"]
        peaks = analysis["peaks"]
        if not len(periods):
            axis.text(0.5, 0.5, "Not enough data", ha="center", va="center")
            axis.set_title(f"DBMF vs {ticker}")
            continue

        order = np.argsort(periods)
        axis.plot(periods[order], relative_power[order], color="#176B73", linewidth=1.5)
        axis.fill_between(periods[order], relative_power[order], color="#176B73", alpha=0.14)
        axis.set_xlim(CYCLE_MIN_PERIOD, CYCLE_MAX_PERIOD)
        axis.set_ylim(0, 1.08)
        axis.set_title(f"DBMF vs {ticker}")
        axis.set_xlabel("Candidate cycle length (trading days)")
        axis.grid(True, alpha=0.25)

        if peaks:
            dominant = peaks[0]
            period = dominant["period_trading_days"]
            axis.axvline(period, color="#C45736", linestyle="--", linewidth=1.2)
            axis.scatter(
                [peak["period_trading_days"] for peak in peaks],
                [peak["relative_power"] for peak in peaks],
                color="#C45736",
                zorder=3,
            )
            axis.text(
                0.97,
                0.95,
                f"Strongest peak: ~{period:.0f} days",
                transform=axis.transAxes,
                ha="right",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "#C45736", "alpha": 0.9},
            )

    axes[0].set_ylabel("Relative spectral power\n(strongest peak = 1)")
    fig.suptitle("Cycle Analysis of 60-Day Rolling Correlation", y=0.98)
    fig.text(
        0.5,
        0.015,
        "A peak marks a repeat length with stronger variation in the correlation series. "
        "It is descriptive, not proof that the pattern will repeat.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(OUTPUT_CYCLE_PLOT, dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    prices = load_prices()
    returns = load_returns(prices)
    performance = calculate_performance_metrics(prices, returns)
    print("Performance metrics (daily returns; annual risk-free rate = 3%):")
    for ticker, row in performance.iterrows():
        print(
            f"  {ticker}: Sharpe={row['sharpe_ratio']:.3f}, "
            f"max drawdown={row['maximum_drawdown']:.1%}"
        )

    print("Full-sample Pearson correlations of daily percentage returns:")
    for ticker, column in COMPARISON_COLUMNS.items():
        correlation = returns[DBMF_COLUMN].corr(returns[column])
        print(f"  DBMF vs {ticker}: {correlation:.4f}")
    plot_rebased_prices(prices)

    monthly_prices = prices.resample("ME").last()
    monthly_returns = load_returns(monthly_prices)
    print("\nFull-sample Pearson correlations of monthly percentage returns:")
    for ticker, column in COMPARISON_COLUMNS.items():
        correlation = monthly_returns[DBMF_COLUMN].corr(monthly_returns[column])
        print(f"  DBMF vs {ticker}: {correlation:.4f}")

    rolling = calculate_rolling_correlations(returns)
    save_rolling_correlation_csvs(rolling)
    print("\nDominant cycles in 60-trading-day rolling correlations:")
    print("  The analysis removes a linear trend, then measures repeating variation")
    print("  from 10 to 252 trading days using a Hann-window periodogram.")
    print("  A peak at 40 days means the correlation varied most strongly at about")
    print("  a 40-observation repeat length; it does not predict a future turning point.")
    print("  Peak power share is that peak's portion of all power in the analyzed range.")
    cycle_analyses = {}
    for ticker in COMPARISON_COLUMNS:
        analysis = analyze_correlation_cycles(
            rolling[f"{ticker} 60 trading days"],
            min_period=CYCLE_MIN_PERIOD,
            max_period=CYCLE_MAX_PERIOD,
        )
        cycle_analyses[ticker] = analysis
        if not analysis["peaks"]:
            print(f"  {ticker}: not enough valid observations for cycle analysis")
            continue
        formatted = ", ".join(
            f"{cycle['period_trading_days']:.0f} days "
            f"({cycle['band_power_share']:.1%} of in-range power)"
            for cycle in analysis["peaks"]
        )
        print(f"  {ticker}: {formatted}")
    print("  Note: overlapping rolling windows can create apparent persistence/cycles.")
    plot_cycle_analysis(cycle_analyses)
    print(f"Saved cycle explanation chart to {OUTPUT_CYCLE_PLOT}")

    for ticker in COMPARISON_COLUMNS:
        plot_rolling_correlations(rolling, ticker)

    for ticker in COMPARISON_COLUMNS:
        monthly_rolling_corr = calculate_rolling_monthly_return_correlation(
            monthly_returns, ticker
        )
        plot_correlation_series(
            monthly_rolling_corr,
            ticker,
            f"DBMF vs {ticker}: 12-Month Rolling Correlation of Monthly Returns",
            f"DBMF_COMP_{ticker}_monthly_returns_12m_rolling_correlation.png",
            "Returns use month-end prices; the 12-month window advances one month at a time.",
        )

if __name__ == "__main__":
    main()