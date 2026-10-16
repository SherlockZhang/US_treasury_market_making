"""Calculate DBMF's annualized Sharpe ratio using Bloomberg workbook data."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).with_name("bbg_data.xlsx")
SHEET_NAME = "Sheet1"
OUTPUT_DIR = Path(__file__).with_name("component_analysis") / "results"
OUTPUT_FILE = OUTPUT_DIR / "dbmf_sharpe_stats.csv"
TRADING_DAYS_PER_YEAR = 252


def load_aligned_returns():
    """Load and align DBMF, SPX, and risk-free daily returns by date."""
    frame = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    required_columns = {
        "Date",
        "DBMF",
        "Date.1",
        "risk_free",
        "Date.2",
        "SPX",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE.name}: "
            f"{sorted(missing_columns)}"
        )

    dbmf = frame.loc[:, ["Date", "DBMF"]].dropna().copy()
    risk_free = frame.loc[:, ["Date.1", "risk_free"]].dropna().copy()
    spx = frame.loc[:, ["Date.2", "SPX"]].dropna().copy()
    dbmf["Date"] = pd.to_datetime(dbmf["Date"], errors="raise")
    risk_free["Date.1"] = pd.to_datetime(risk_free["Date.1"], errors="raise")
    spx["Date.2"] = pd.to_datetime(spx["Date.2"], errors="raise")
    dbmf["DBMF"] = pd.to_numeric(dbmf["DBMF"], errors="raise")
    risk_free["risk_free"] = pd.to_numeric(
        risk_free["risk_free"], errors="raise"
    )
    spx["SPX"] = pd.to_numeric(spx["SPX"], errors="raise")

    if dbmf["Date"].duplicated().any():
        raise ValueError("The DBMF data contains duplicate dates.")
    if risk_free["Date.1"].duplicated().any():
        raise ValueError("The risk-free data contains duplicate dates.")
    if spx["Date.2"].duplicated().any():
        raise ValueError("The SPX data contains duplicate dates.")

    dbmf = dbmf.sort_values("Date")
    spx = spx.sort_values("Date.2")
    dbmf["dbmf_daily_return_percent"] = cumulative_to_daily_returns(dbmf["DBMF"])
    dbmf = dbmf.dropna(subset=["dbmf_daily_return_percent"])
    spx["spx_daily_return_percent"] = cumulative_to_daily_returns(spx["SPX"])
    spx = spx.dropna(subset=["spx_daily_return_percent"])

    if (risk_free["risk_free"] <= -100).any():
        raise ValueError("Annualized risk-free yields must be greater than -100%.")
    risk_free["risk_free_daily_return_percent"] = (
        (1 + risk_free["risk_free"] / 100)
        ** (1 / TRADING_DAYS_PER_YEAR)
        - 1
    ) * 100

    aligned = dbmf.loc[:, ["Date", "dbmf_daily_return_percent"]].merge(
        risk_free.loc[:, ["Date.1", "risk_free_daily_return_percent"]],
        left_on="Date",
        right_on="Date.1",
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.merge(
        spx.loc[:, ["Date.2", "spx_daily_return_percent"]],
        left_on="Date",
        right_on="Date.2",
        how="inner",
        validate="one_to_one",
    )
    if aligned.empty:
        raise ValueError(
            "No DBMF daily returns match both SPX and risk-free observations by date."
        )

    aligned["excess_return_percent"] = (
        aligned["dbmf_daily_return_percent"]
        - aligned["risk_free_daily_return_percent"]
    )
    aligned["spx_excess_return_percent"] = (
        aligned["spx_daily_return_percent"]
        - aligned["risk_free_daily_return_percent"]
    )
    return aligned


def cumulative_to_daily_returns(cumulative_returns):
    """Convert cumulative total-return percentages into daily return percentages."""
    if (cumulative_returns <= -100).any():
        raise ValueError("Cumulative returns must be greater than -100%.")
    previous_total_return = cumulative_returns.shift(1)
    return (
        (100 + cumulative_returns) / (100 + previous_total_return) - 1
    ) * 100


def calculate_sharpe_statistics(aligned_returns):
    """Calculate the annualized Sharpe ratio from daily excess returns."""
    excess_returns = aligned_returns["excess_return_percent"]
    if len(excess_returns) < 2:
        raise ValueError("At least two aligned daily returns are required.")

    daily_volatility = excess_returns.std(ddof=1)
    if not np.isfinite(daily_volatility) or daily_volatility == 0:
        raise ValueError("Daily excess-return volatility must be positive.")

    annualization_factor = np.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "observations": len(aligned_returns),
        "start_date": aligned_returns["Date"].min().date(),
        "end_date": aligned_returns["Date"].max().date(),
        "annualized_mean_excess_return_percent": (
            excess_returns.mean() * TRADING_DAYS_PER_YEAR
        ),
        "annualized_excess_return_volatility_percent": (
            daily_volatility * annualization_factor
        ),
        "sharpe_ratio": excess_returns.mean()
        / daily_volatility
        * annualization_factor,
    }


def calculate_beta_alpha(aligned_returns):
    """Estimate DBMF beta and annualized alpha from daily CAPM excess returns."""
    dbmf_excess_returns = aligned_returns["excess_return_percent"]
    spx_excess_returns = aligned_returns["spx_excess_return_percent"]
    if len(aligned_returns) < 3:
        raise ValueError("At least three aligned observations are required.")

    design = np.column_stack(
        [np.ones(len(spx_excess_returns)), spx_excess_returns.to_numpy()]
    )
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("SPX excess returns must have non-zero variance.")

    daily_alpha, beta = np.linalg.lstsq(
        design, dbmf_excess_returns.to_numpy(), rcond=None
    )[0]
    return {
        "beta_vs_spx": beta,
        "daily_alpha_percent": daily_alpha,
        "annualized_alpha_percent": daily_alpha * TRADING_DAYS_PER_YEAR,
    }


def main():
    aligned_returns = load_aligned_returns()
    stats = {
        **calculate_sharpe_statistics(aligned_returns),
        **calculate_beta_alpha(aligned_returns),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(OUTPUT_FILE, index=False, float_format="%.6f")
    print(f"Aligned daily observations: {stats['observations']}")
    print(f"Date range: {stats['start_date']} to {stats['end_date']}")
    print(
        "Annualized DBMF Sharpe ratio: "
        f"{stats['sharpe_ratio']:.4f}"
    )
    print(f"DBMF beta vs. SPX: {stats['beta_vs_spx']:.4f}")
    print(
        "Annualized DBMF CAPM alpha: "
        f"{stats['annualized_alpha_percent']:.4f}%"
    )
    print(f"Saved Sharpe statistics to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
