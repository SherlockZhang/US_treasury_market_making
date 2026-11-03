"""Backtest a normal-CDF momentum signal on 5Y and 10Y-2Y prices."""

import argparse
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_2y_10y_spread_strategy import load_spread
from backtest_5y_trend_strategy import (
    TARGET_ANNUALIZED_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
    calculate_annual_statistics,
    calculate_statistics,
    calculate_trading_volume,
    load_5y_yield,
    load_10y_prices,
    load_spx_index,
    parse_excel_dates,
    run_backtest as run_trend_backtest,
    summarize_5y_yield_change,
    summarize_spx_change,
)


OUTPUT_DIR = Path(__file__).with_name("results_refining_parameter")
FACTOR_DATA_FILE = Path(__file__).with_name("market_data.xlsx")
FACTOR_DATA_SHEET = "Sheet1"
BACKTEST_DATA_FILE = Path(__file__).with_name("US_future_data.xlsx")
BACKTEST_DATA_SHEET = "Sheet1"
FUTURE_PRICE_CSV = Path(__file__).with_name("future_price.csv")
PRESENTATION_OUTPUT_DIR = Path(__file__).with_name("presentation report")
ETF_BENCHMARK_TICKERS = {
    "DBMF US Equity": "DBMF",
    "CTA US Equity": "CTA",
}
DEFAULT_LOOKBACK_DAYS = 252
AVERAGED_SIGNAL_LOOKBACK_DAYS = (32, 64, 126, 252, 504)
NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS = 64
FACTOR_SCORE_LOOKBACK_DAYS = 64
FACTOR_SCORE_ZSCORE_LOOKBACK_DAYS = 64
FACTOR_SIGNAL_BOOST_THRESHOLD = 1.0
FACTOR_SIGNAL_BOOST_MULTIPLIER = 1.1
FACTOR_SIGNAL_REDUCE_MULTIPLIER = 0.9
NORMAL_CDF_SIGNAL_CUTOFF = 0.1
REBALANCE_POSITION_CHANGE_THRESHOLD_FRACTION = 0.25
POSITION_ROUNDING_INCREMENTS = (0.1, 0.2, 0.5)
DEFAULT_POSITION_ROUNDING_INCREMENT = 0.1
RebalanceFrequency = Literal["daily", "weekly", "monthly", "yearly"]
SignalMode = Literal["single", "average", "rounded"]
FactorMode = Literal["none", "market-data"]
DEFAULT_REBALANCE_FREQUENCY: RebalanceFrequency = "weekly"
FACTOR_TICKERS = {
    "JPYUSD": ("JPYUSD BGN Curncy", 1),
    "CL1": ("CL1 COMB Comdty", -1),
    "BBDXY": ("BBDXY Index", -1),
    "TFF2GLFN": ("TFF2GLFN Index", -1),
    "XAU": ("XAU BGN Curncy", 1),
    "DJI": ("DJI", 1),
    "TFF2ILFN": ("TFF2ILFN Index", 1),
}
SPREAD_FACTOR_TICKERS = {
    "CL1": ("CL1 COMB Comdty", 1),
    "DJI": ("DJI", -1),
    "JPY": ("JPYUSD BGN Curncy", -1),
    "SOFR": ("SOFRRATE Index", 1),
    "TFF2GLFN": ("TFF2GLFN Index", 1),
    "DAX": ("DAX", -1),
    "EURUSD": ("EURUSD BGN Curncy", 1),
    "S5INFT": ("S5INFT Index", 1),
    "TFF2ILFN": ("TFF2ILFN Index", -1),
}

STRATEGIES = {
    "10y_future": {
        "instrument": "10Y future",
        "stem": "10y_future_normal_cdf_signal",
        "color": "#245a84",
    },
    "10y_2y_spread": {
        "instrument": "5x2Y - 3x10Y futures basket",
        "stem": "5x2y_minus_3x10y_normal_cdf_signal",
        "color": "#8c4a2f",
    },
}
SINGLE_FUTURE_STRATEGIES = {
    "5Y future": {"stem": "5y_future", "color": "#245a84"},
    "2Y future": {"stem": "2y_future", "color": "#8c4a2f"},
    "10Y future": {"stem": "10y_future", "color": "#4f772d"},
}


def signal_lookbacks_for_mode(
    signal_mode: SignalMode, lookback_days: int
) -> tuple[int, ...]:
    """Return the lookback windows used by a signal mode."""
    if signal_mode == "average":
        return AVERAGED_SIGNAL_LOOKBACK_DAYS
    return (lookback_days,)


def positive_int(value: str) -> int:
    """Parse a positive integer lookback for the command-line option."""
    try:
        days = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "lookback days must be a positive integer"
        ) from error
    if days <= 0:
        raise argparse.ArgumentTypeError("lookback days must be a positive integer")
    return days


def iso_date(value: str) -> pd.Timestamp:
    """Parse a date supplied as YYYY-MM-DD."""
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "start date must use YYYY-MM-DD format"
        ) from error
    if parsed.strftime("%Y-%m-%d") != value:
        raise argparse.ArgumentTypeError("start date must use YYYY-MM-DD format")
    return parsed


def nonnegative_float(value: str) -> float:
    """Parse a finite non-negative position-change threshold fraction."""
    try:
        fraction = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "threshold fraction must be a finite non-negative number"
        ) from error
    if not np.isfinite(fraction) or fraction < 0:
        raise argparse.ArgumentTypeError(
            "threshold fraction must be a finite non-negative number"
        )
    return fraction


def calculate_factor_returns(
    values: pd.Series, zero_crossing_returns: bool
) -> pd.Series:
    """Calculate percentage changes, neutralizing sign-crossing observations."""
    returns = values.pct_change(fill_method=None)
    if zero_crossing_returns:
        previous_values = values.shift(1)
        crosses_zero = (
            previous_values.notna()
            & (values * previous_values <= 0)
            & values.ne(previous_values)
        )
        returns.loc[crosses_zero] = 0.0
    return returns


def load_etf_benchmark_prices() -> dict[str, pd.Series]:
    """Load and validate DBMF and CTA benchmark price series."""
    frame = pd.read_excel(BACKTEST_DATA_FILE, sheet_name=BACKTEST_DATA_SHEET)
    columns = list(frame.columns)
    benchmark_prices: dict[str, pd.Series] = {}

    for ticker, benchmark_name in ETF_BENCHMARK_TICKERS.items():
        if ticker not in columns:
            raise ValueError(
                f"Required benchmark ticker {ticker!r} was not found in "
                f"{BACKTEST_DATA_FILE.name}."
            )
        ticker_column_index = columns.index(ticker)
        if ticker_column_index == 0:
            raise ValueError(f"No date column found for benchmark {ticker!r}.")
        date_column = columns[ticker_column_index - 1]
        if not isinstance(date_column, str) or not date_column.strip().startswith(
            "Date"
        ):
            raise ValueError(
                f"Expected a date column immediately before benchmark {ticker!r}."
            )

        observations = frame.loc[:, [date_column, ticker]].copy()
        observations[date_column] = parse_excel_dates(observations[date_column])
        observations[ticker] = pd.to_numeric(
            observations[ticker], errors="raise"
        )
        partially_blank = observations.isna().any(axis=1) & observations.notna().any(
            axis=1
        )
        if partially_blank.any():
            raise ValueError(
                f"Benchmark {ticker!r} has rows with only a date or price."
            )
        observations = observations.dropna(how="all")
        if observations.empty:
            raise ValueError(f"No observations found for benchmark {ticker!r}.")
        if observations[date_column].duplicated().any():
            raise ValueError(f"Benchmark {ticker!r} contains duplicate dates.")

        prices = observations.sort_values(date_column).set_index(date_column)[ticker]
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(
                f"Benchmark {ticker!r} prices must be finite and positive."
            )
        benchmark_prices[benchmark_name] = prices.rename(benchmark_name)

    return benchmark_prices


def load_factor_scores(
    target_dates: pd.DatetimeIndex,
    factor_tickers: dict[str, tuple[str, int]] = FACTOR_TICKERS,
) -> pd.DataFrame:
    """Load and average signed rolling factor t-statistics on target dates."""
    frame = pd.read_excel(FACTOR_DATA_FILE, sheet_name=FACTOR_DATA_SHEET)
    columns = list(frame.columns)
    factor_scores: dict[str, pd.Series] = {}

    for factor_name, (ticker, sign) in factor_tickers.items():
        if ticker not in columns:
            raise ValueError(
                f"Required factor ticker {ticker!r} for {factor_name!r} "
                f"was not found in {FACTOR_DATA_FILE.name}."
            )
        ticker_column_index = columns.index(ticker)
        if ticker_column_index == 0:
            raise ValueError(f"No date column found for factor ticker {ticker!r}.")
        date_column = columns[ticker_column_index - 1]
        if not isinstance(date_column, str) or not date_column.strip().startswith(
            "Date"
        ):
            raise ValueError(
                f"Expected a date column immediately before factor {ticker!r}."
            )

        observations = frame.loc[:, [date_column, ticker]].copy()
        observations[date_column] = parse_excel_dates(observations[date_column])
        observations[ticker] = pd.to_numeric(
            observations[ticker], errors="raise"
        )
        partially_blank = observations.isna().any(axis=1) & observations.notna().any(
            axis=1
        )
        if partially_blank.any():
            raise ValueError(
                f"Factor ticker {ticker!r} has rows with only a date or value."
            )
        observations = observations.dropna(how="all")
        if observations.empty:
            raise ValueError(f"No observations found for factor ticker {ticker!r}.")
        if observations[date_column].duplicated().any():
            raise ValueError(f"Factor ticker {ticker!r} contains duplicate dates.")
        values = observations.sort_values(date_column).set_index(date_column)[ticker]
        if not np.isfinite(values).all():
            raise ValueError(
                f"Factor ticker {ticker!r} values must be finite."
            )

        returns = calculate_factor_returns(
            values, factor_name in ("CL1", "TFF2GLFN")
        )
        rolling_returns = returns.rolling(
            FACTOR_SCORE_LOOKBACK_DAYS,
            min_periods=FACTOR_SCORE_LOOKBACK_DAYS,
        )
        rolling_mean = rolling_returns.mean()
        rolling_std = rolling_returns.std(ddof=1)
        t_statistic = (
            np.sqrt(FACTOR_SCORE_LOOKBACK_DAYS)
            * rolling_mean
            / rolling_std.where(rolling_std > 0)
        )
        factor_scores[factor_name] = (sign * t_statistic).reindex(
            target_dates, method="ffill"
        )

    signed_scores = pd.DataFrame(factor_scores, index=target_dates)
    signed_scores["factor_score"] = signed_scores.mean(axis=1, skipna=False)
    prior_scores = signed_scores["factor_score"].shift(1)
    rolling_mean = prior_scores.rolling(
        FACTOR_SCORE_ZSCORE_LOOKBACK_DAYS,
        min_periods=FACTOR_SCORE_ZSCORE_LOOKBACK_DAYS,
    ).mean()
    rolling_std = prior_scores.rolling(
        FACTOR_SCORE_ZSCORE_LOOKBACK_DAYS,
        min_periods=FACTOR_SCORE_ZSCORE_LOOKBACK_DAYS,
    ).std(ddof=1)
    signed_scores["factor_score_zscore"] = (
        signed_scores["factor_score"] - rolling_mean
    ) / rolling_std.where(rolling_std > 0)
    return signed_scores


def load_single_future_prices() -> dict[str, pd.Series]:
    """Load individual futures and DBMF prices from the repeated-pair CSV."""
    frame = pd.read_csv(FUTURE_PRICE_CSV)
    price_columns = [*SINGLE_FUTURE_STRATEGIES, "DBMF Price"]
    prices_by_name: dict[str, pd.Series] = {}

    for price_column in price_columns:
        if price_column not in frame.columns:
            raise ValueError(
                f"Required price column {price_column!r} was not found in "
                f"{FUTURE_PRICE_CSV.name}."
            )
        price_column_index = list(frame.columns).index(price_column)
        if price_column_index == 0:
            raise ValueError(f"No date column found for {price_column!r}.")
        date_column_index = price_column_index - (
            2 if price_column == "DBMF Price" else 1
        )
        if date_column_index < 0:
            raise ValueError(f"No date column found for {price_column!r}.")
        date_column = frame.columns[date_column_index]
        if not isinstance(date_column, str) or not date_column.strip().startswith(
            "Date"
        ):
            raise ValueError(
                f"Expected the date column at position {date_column_index} "
                f"before {price_column!r}."
            )

        observations = frame.loc[:, [date_column, price_column]].copy()
        observations[date_column] = parse_csv_dates(observations[date_column])
        observations[price_column] = pd.to_numeric(
            observations[price_column], errors="raise"
        )
        partially_blank = observations.isna().any(axis=1) & observations.notna().any(
            axis=1
        )
        if partially_blank.any():
            raise ValueError(
                f"The {price_column!r} date and price columns contain "
                "partially blank rows."
            )
        observations = observations.dropna(how="all")
        if observations.empty:
            raise ValueError(f"No observations found for {price_column!r}.")
        if observations[date_column].duplicated().any():
            raise ValueError(f"{price_column!r} contains duplicate dates.")
        values = observations[price_column]
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(
                f"{price_column!r} prices must be finite and positive."
            )
        prices_by_name[price_column] = (
            observations.sort_values(date_column)
            .set_index(date_column)[price_column]
            .rename(price_column)
        )

    return prices_by_name


def parse_csv_dates(values: pd.Series) -> pd.Series:
    """Parse CSV dates stored as strings or Excel serial numbers."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_mask = values.notna() & numeric_values.notna()
    parsed_dates = pd.Series(
        pd.NaT,
        index=values.index,
        name=values.name,
        dtype="datetime64[ns]",
    )
    if numeric_mask.any():
        parsed_dates.loc[numeric_mask] = pd.to_datetime(
            numeric_values.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="raise",
        )
    text_mask = values.notna() & ~numeric_mask
    if text_mask.any():
        parsed_dates.loc[text_mask] = pd.to_datetime(
            values.loc[text_mask],
            errors="raise",
        )
    return parsed_dates


def calculate_normal_cdf_signal(
    daily_returns: pd.Series,
    lookback_days: int,
    signal_cutoff: float = NORMAL_CDF_SIGNAL_CUTOFF,
) -> pd.DataFrame:
    """Calculate rolling t-statistics and map them to signals in [-1, 1]."""
    if lookback_days <= 0:
        raise ValueError("Lookback days must be a positive integer.")
    if not np.isfinite(signal_cutoff) or signal_cutoff < 0:
        raise ValueError("Signal cutoff must be finite and non-negative.")

    rolling_returns = daily_returns.rolling(
        window=lookback_days, min_periods=lookback_days
    )
    average_return = rolling_returns.mean()
    return_volatility = rolling_returns.std(ddof=1)
    t_stat = pd.Series(
        np.sqrt(lookback_days) * average_return / return_volatility,
        index=daily_returns.index,
    )
    unfiltered_signal = pd.Series(
        3 * norm.cdf(t_stat) - 1.5, index=daily_returns.index
    )
    unfiltered_signal = unfiltered_signal.where(return_volatility.gt(0))
    signal = unfiltered_signal.where(
        unfiltered_signal.isna()
        | (unfiltered_signal.abs() > signal_cutoff),
        0.0,
    )

    return pd.DataFrame(
        {
            "average_return": average_return,
            "lookback_volatility": return_volatility,
            "t_stat": t_stat,
            "unfiltered_signal": unfiltered_signal,
            "signal": signal,
        },
        index=daily_returns.index,
    )


def run_backtest(
    prices: pd.Series,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    rebalance_frequency: RebalanceFrequency = DEFAULT_REBALANCE_FREQUENCY,
    position_change_threshold_fraction: float = (
        REBALANCE_POSITION_CHANGE_THRESHOLD_FRACTION
    ),
    signal_cutoff: float = NORMAL_CDF_SIGNAL_CUTOFF,
    signal_mode: SignalMode = "single",
    factor_scores: pd.DataFrame | None = None,
    volatility_lookback_days: int = NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS,
    output_start_date: pd.Timestamp | None = None,
    position_rounding_increment: float = DEFAULT_POSITION_ROUNDING_INCREMENT,
) -> pd.DataFrame:
    """Apply the signal with daily, Monday-weekly, month-end, or year-end sizing."""
    if lookback_days <= 0:
        raise ValueError("Lookback days must be a positive integer.")
    if volatility_lookback_days <= 0:
        raise ValueError("Volatility lookback days must be a positive integer.")
    if rebalance_frequency not in ("daily", "weekly", "monthly", "yearly"):
        raise ValueError(
            f"Unsupported rebalance frequency: {rebalance_frequency}"
        )
    if (
        not np.isfinite(position_change_threshold_fraction)
        or position_change_threshold_fraction < 0
    ):
        raise ValueError(
            "Position-change threshold fraction must be finite and non-negative."
        )
    if not np.isfinite(signal_cutoff) or signal_cutoff < 0:
        raise ValueError("Signal cutoff must be finite and non-negative.")
    if signal_mode not in ("single", "average", "rounded"):
        raise ValueError(f"Unsupported signal mode: {signal_mode}")
    if position_rounding_increment not in POSITION_ROUNDING_INCREMENTS:
        raise ValueError(
            "Position-rounding increment must be one of "
            f"{POSITION_ROUNDING_INCREMENTS}."
        )
    if prices.empty:
        raise ValueError("No prices are available for this backtest.")
    if prices.isna().any() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Backtest prices must be finite, non-missing, and positive.")
    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")

    prices = prices.sort_index()
    results = prices.rename("price").to_frame()
    dates = pd.DatetimeIndex(results.index)
    results.index = dates
    results["daily_return"] = results["price"].pct_change(fill_method=None)
    signal_data = calculate_normal_cdf_signal(
        results["daily_return"], lookback_days, signal_cutoff
    )
    if signal_mode == "average":
        component_signals = pd.DataFrame(
            {
                f"signal_t{component_lookback}": calculate_normal_cdf_signal(
                    results["daily_return"],
                    component_lookback,
                    signal_cutoff,
                )["signal"]
                for component_lookback in AVERAGED_SIGNAL_LOOKBACK_DAYS
            },
            index=results.index,
        )
        signal_data["signal"] = component_signals.mean(axis=1, skipna=False)
        signal_data = signal_data.join(component_signals)
    results = results.join(signal_data)
    if factor_scores is not None:
        required_score_column = "factor_score"
        if required_score_column not in factor_scores:
            raise ValueError(
                f"Factor scores must include a {required_score_column!r} column."
            )
        aligned_scores = factor_scores.reindex(results.index)
        score = aligned_scores[required_score_column]
        score_zscore = aligned_scores["factor_score_zscore"]
        signal_boost = (
            (score_zscore > FACTOR_SIGNAL_BOOST_THRESHOLD)
            & (results["signal"] > 0)
        ) | (
            (score_zscore < -FACTOR_SIGNAL_BOOST_THRESHOLD)
            & (results["signal"] < 0)
        )
        signal_reduce = (
            (score_zscore > FACTOR_SIGNAL_BOOST_THRESHOLD)
            & (results["signal"] < 0)
        ) | (
            (score_zscore < -FACTOR_SIGNAL_BOOST_THRESHOLD)
            & (results["signal"] > 0)
        )
        results["base_signal"] = results["signal"]
        results["factor_score"] = score
        results["factor_score_zscore"] = score_zscore
        results["factor_signal_multiplier"] = np.select(
            [signal_boost, signal_reduce],
            [
                FACTOR_SIGNAL_BOOST_MULTIPLIER,
                FACTOR_SIGNAL_REDUCE_MULTIPLIER,
            ],
            default=1.0,
        )
        results["signal"] *= results["factor_signal_multiplier"]
    results["realized_volatility"] = (
        results["daily_return"]
        .rolling(
            volatility_lookback_days,
            min_periods=volatility_lookback_days,
        )
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    if output_start_date is not None:
        results = results.loc[results.index >= output_start_date].copy()
        if results.empty:
            raise ValueError(
                f"No backtest prices available on or after "
                f"{output_start_date:%Y-%m-%d}."
            )
        dates = pd.DatetimeIndex(results.index)

    eligible = results["signal"].notna() & results["realized_volatility"].gt(0)
    desired_position = pd.Series(np.nan, index=results.index, dtype=float)
    desired_position.loc[eligible] = (
        results.loc[eligible, "signal"]
        * TARGET_ANNUALIZED_VOLATILITY
        / results.loc[eligible, "realized_volatility"]
    )
    if signal_mode == "rounded":
        desired_position = (
            (desired_position / position_rounding_increment)
            .round()
            .mul(position_rounding_increment)
            .round(1)
        )
    if rebalance_frequency == "daily":
        rebalance_dates = np.ones(len(dates), dtype=bool)
    elif rebalance_frequency == "weekly":
        rebalance_dates = dates.day_of_week == 0
    elif rebalance_frequency == "monthly":
        months = dates.to_period("M")
        rebalance_dates = np.r_[
            months[:-1] != months[1:],
            True,
        ]
    else:
        years = dates.year
        rebalance_dates = np.r_[years[:-1] != years[1:], True]
    position_change_threshold = (
        position_change_threshold_fraction
        * TARGET_ANNUALIZED_VOLATILITY
        / results["realized_volatility"]
    )
    position_change_threshold = position_change_threshold.where(rebalance_dates)
    target_position = np.zeros(len(results), dtype=float)
    current_target_position = 0.0
    desired_position_values = desired_position.to_numpy()
    threshold_values = position_change_threshold.to_numpy()
    for index, is_rebalance_date in enumerate(rebalance_dates):
        if not is_rebalance_date:
            target_position[index] = current_target_position
            continue

        candidate = desired_position_values[index]
        threshold = threshold_values[index]
        if np.isfinite(candidate) and np.isfinite(threshold):
            if abs(candidate - current_target_position) >= threshold:
                current_target_position = candidate
        target_position[index] = current_target_position

    results["desired_position"] = desired_position
    results["position_change_threshold"] = position_change_threshold
    results["target_position"] = pd.Series(target_position, index=results.index)
    position_at_close = results["target_position"]
    results["position"] = position_at_close.shift(1).fillna(0.0)
    results["strategy_return"] = (
        results["position"] * results["daily_return"]
    ).fillna(0.0)

    results["wealth"] = (1 + results["strategy_return"]).cumprod()
    if (results["wealth"] <= 0).any():
        raise ValueError(
            "The strategy's compounded wealth reached zero or less; "
            "check for extreme returns or volatility-target leverage."
        )
    results["cumulative_pnl_pct"] = (results["wealth"] - 1) * 100
    results["running_peak"] = results["wealth"].cummax()
    results["drawdown"] = results["wealth"] / results["running_peak"] - 1
    results["yearly_wealth"] = results.groupby(dates.year)[
        "strategy_return"
    ].transform(lambda returns: (1 + returns).cumprod())
    results["yearly_cumulative_pnl_pct"] = (
        results["yearly_wealth"] - 1
    ) * 100
    return results


def calculate_stats(
    results: pd.DataFrame,
    instrument: str,
    lookback_days: int,
    rebalance_frequency: RebalanceFrequency,
    position_change_threshold_fraction: float,
    signal_cutoff: float,
    signal_mode: SignalMode,
    factor_mode: str,
    volatility_lookback_days: int,
    position_rounding_increment: float = DEFAULT_POSITION_ROUNDING_INCREMENT,
) -> pd.DataFrame:
    """Calculate full-period and annual statistics."""
    full_stats = calculate_statistics(results)
    annual_stats = calculate_annual_statistics(results)
    full_stats["instrument"] = instrument
    full_stats["signal_mode"] = signal_mode
    full_stats["position_rounding_increment"] = (
        position_rounding_increment if signal_mode == "rounded" else np.nan
    )
    full_stats["factor_mode"] = factor_mode
    full_stats["factor_score_zscore_lookback_days"] = (
        FACTOR_SCORE_ZSCORE_LOOKBACK_DAYS
        if factor_mode == "market-data"
        else np.nan
    )
    full_stats["factor_signal_boost_threshold"] = (
        FACTOR_SIGNAL_BOOST_THRESHOLD if factor_mode == "market-data" else np.nan
    )
    full_stats["factor_signal_boost_multiplier"] = (
        FACTOR_SIGNAL_BOOST_MULTIPLIER
        if factor_mode == "market-data"
        else np.nan
    )
    full_stats["factor_signal_reduce_multiplier"] = (
        FACTOR_SIGNAL_REDUCE_MULTIPLIER
        if factor_mode == "market-data"
        else np.nan
    )
    signal_lookbacks = signal_lookbacks_for_mode(signal_mode, lookback_days)
    full_stats["signal_lookback_days"] = (
        np.nan if signal_mode == "average" else lookback_days
    )
    full_stats["signal_lookbacks_days"] = ",".join(
        str(days) for days in signal_lookbacks
    )
    full_stats["volatility_lookback_days"] = volatility_lookback_days
    full_stats["signal_cutoff"] = signal_cutoff
    full_stats["rebalance_position_change_threshold_fraction"] = (
        position_change_threshold_fraction
    )
    full_stats["rebalance_frequency"] = rebalance_frequency
    for annual_stat in annual_stats:
        annual_stat["instrument"] = instrument
        annual_stat["signal_mode"] = signal_mode
        annual_stat["position_rounding_increment"] = (
            position_rounding_increment if signal_mode == "rounded" else np.nan
        )
        annual_stat["factor_mode"] = factor_mode
        annual_stat["factor_score_zscore_lookback_days"] = (
            FACTOR_SCORE_ZSCORE_LOOKBACK_DAYS
            if factor_mode == "market-data"
            else np.nan
        )
        annual_stat["factor_signal_boost_threshold"] = (
            FACTOR_SIGNAL_BOOST_THRESHOLD
            if factor_mode == "market-data"
            else np.nan
        )
        annual_stat["factor_signal_boost_multiplier"] = (
            FACTOR_SIGNAL_BOOST_MULTIPLIER
            if factor_mode == "market-data"
            else np.nan
        )
        annual_stat["factor_signal_reduce_multiplier"] = (
            FACTOR_SIGNAL_REDUCE_MULTIPLIER
            if factor_mode == "market-data"
            else np.nan
        )
        annual_stat["signal_lookback_days"] = (
            np.nan if signal_mode == "average" else lookback_days
        )
        annual_stat["signal_lookbacks_days"] = full_stats[
            "signal_lookbacks_days"
        ]
        annual_stat["volatility_lookback_days"] = volatility_lookback_days
        annual_stat["signal_cutoff"] = signal_cutoff
        annual_stat["rebalance_position_change_threshold_fraction"] = (
            position_change_threshold_fraction
        )
        annual_stat["rebalance_frequency"] = rebalance_frequency

    sheet1_columns = set(
        pd.read_excel(BACKTEST_DATA_FILE, sheet_name=BACKTEST_DATA_SHEET, nrows=0)
        .columns
    )
    if "5Y Yield" in sheet1_columns:
        yields = load_5y_yield()
        yield_dates = pd.DatetimeIndex(yields.index)
        full_stats.update(summarize_5y_yield_change(yields))
        for annual_stat in annual_stats:
            year = int(cast(str, annual_stat["period"]))
            annual_yields = yields.loc[yield_dates.year == year]
            annual_stat.update(summarize_5y_yield_change(annual_yields))
    else:
        print(
            "5Y yield statistics unavailable: '5Y Yield' is not present "
            "on Sheet1; related CSV fields will be blank."
        )
        for stats in [full_stats, *annual_stats]:
            stats.update(
                {
                    "5y_yield_start_date": np.nan,
                    "5y_yield_end_date": np.nan,
                    "5y_yield_start_pct": np.nan,
                    "5y_yield_end_pct": np.nan,
                    "5y_yield_change_percentage_points": np.nan,
                }
            )

    if "SPX" in sheet1_columns:
        spx = load_spx_index()
        spx_dates = pd.DatetimeIndex(spx.index)
        full_stats.update(summarize_spx_change(spx))
        for annual_stat in annual_stats:
            year = int(cast(str, annual_stat["period"]))
            annual_spx = spx.loc[spx_dates.year == year]
            annual_stat.update(summarize_spx_change(annual_spx))
    else:
        print(
            "SPX statistics unavailable: 'SPX' is not present on Sheet1; "
            "related CSV fields will be blank."
        )
        for stats in [full_stats, *annual_stats]:
            stats.update(
                {
                    "spx_start_date": np.nan,
                    "spx_end_date": np.nan,
                    "spx_start_index": np.nan,
                    "spx_end_index": np.nan,
                    "spx_change_index_points": np.nan,
                }
            )
    return pd.DataFrame([full_stats, *annual_stats])


def calculate_comparison_metrics(returns: pd.Series) -> dict[str, float]:
    """Calculate comparable return, volatility, Sharpe, and drawdown metrics."""
    wealth = (1 + returns).cumprod()
    daily_volatility = returns.std(ddof=1)
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (
        returns.mean() / daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
        if np.isfinite(daily_volatility) and daily_volatility > 0
        else np.nan
    )
    return {
        "total_return_pct": (wealth.iloc[-1] - 1) * 100,
        "annualized_return_pct": returns.mean() * TRADING_DAYS_PER_YEAR * 100,
        "annualized_volatility_pct": annualized_volatility * 100,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": (wealth / wealth.cummax() - 1).min() * 100,
    }


def calculate_relative_rebalance_volume(position: pd.Series) -> pd.Series:
    """Scale gross position changes by the trailing mean absolute position."""
    average_absolute_position = (
        position.abs().shift(1).rolling(window=21, min_periods=1).mean()
    )
    return (
        calculate_trading_volume(position)
        .div(average_absolute_position.where(average_absolute_position.ne(0)))
        .fillna(0.0)
        .mul(100)
    )


def save_plots(
    results: pd.DataFrame,
    benchmark_prices: pd.Series,
    etf_benchmark_prices: dict[str, pd.Series],
    instrument: str,
    color: str,
    rebalance_frequency: RebalanceFrequency,
    position_change_threshold_fraction: float,
    signal_cutoff: float,
    signal_label: str,
    plot_path: Path,
    signal_plot_path: Path,
    comparison_path: Path,
    comparison_table_path: Path,
    factor_mode: FactorMode,
    volatility_lookback_days: int,
) -> None:
    """Save signal, cumulative PnL, and comparison plots."""
    figure, signal_axis = plt.subplots(figsize=(12, 6))
    volume_axis = signal_axis.twinx()
    price_axis = signal_axis.twinx()
    price_axis.spines["right"].set_position(("outward", 60))

    signal_points = signal_axis.scatter(
        results.index,
        results["signal"],
        color="#87CEFA",
        s=10,
        label="Generated signal",
    )
    signal_axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    rebalance_volume = calculate_relative_rebalance_volume(results["position"])
    volume_bars = volume_axis.bar(
        results.index,
        rebalance_volume,
        width=2.0,
        color="#B8860B",
        alpha=0.45,
        label="Rebalance volume",
    )
    price_line, = price_axis.plot(
        results.index,
        results["price"],
        color="#59636E",
        linewidth=1.0,
        alpha=0.8,
        label="Underlying price",
    )
    signal_axis.set_title(
        f"{instrument} Generated Signal, Rebalance Volume, and Price"
    )
    signal_axis.set_xlabel("Date")
    signal_axis.set_ylabel("Generated signal", color=color)
    volume_axis.set_ylabel(
        "Rebalance volume (% of prior 21-day average absolute position)",
        color="#A87912",
    )
    price_axis.set_ylabel("Underlying price", color="#59636E")
    signal_axis.tick_params(axis="y", labelcolor=color)
    volume_axis.tick_params(axis="y", labelcolor="#A87912")
    price_axis.tick_params(axis="y", labelcolor="#59636E")
    signal_axis.grid(True, alpha=0.25)
    signal_axis.legend(
        [signal_points, volume_bars, price_line],
        ["Generated signal", "Rebalance volume", "Underlying price"],
        loc="upper left",
    )
    figure.tight_layout(rect=(0, 0, 0.82, 1))
    figure.savefig(signal_plot_path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        color=color,
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"{instrument} Normal-CDF Signal ({signal_label}) — "
        f"{rebalance_frequency.capitalize()} Rebalance — "
        f"{volatility_lookback_days}-Day Volatility Scaling — "
        f"Signal Cutoff ±{signal_cutoff:.1f} — "
        f"{position_change_threshold_fraction:.0%} Rebalance Threshold — "
        f"{factor_mode} factors — "
        "Full-Period Cumulative PnL"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    original_results = run_trend_backtest(
        benchmark_prices.rename("price").to_frame()
    )
    benchmark_returns = original_results["strategy_return"].reindex(results.index)
    benchmark_cumulative_pnl = (1 + benchmark_returns).cumprod().sub(1).mul(100)
    trend_strategy_name = f"1-year moving-average strategy ({instrument})"
    aligned_etf_prices: dict[str, pd.Series] = {}
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        benchmark_cumulative_pnl.index,
        benchmark_cumulative_pnl,
        label=trend_strategy_name,
        linewidth=1.5,
    )
    for benchmark_name, prices in etf_benchmark_prices.items():
        aligned_prices = prices.reindex(results.index, method="ffill").dropna()
        if aligned_prices.empty:
            raise ValueError(
                f"No {benchmark_name} benchmark prices overlap the backtest dates."
            )
        aligned_etf_prices[benchmark_name] = aligned_prices
        benchmark_cumulative_return = (
            aligned_prices / aligned_prices.iloc[0] - 1
        ) * 100
        axis.plot(
            benchmark_cumulative_return.index,
            benchmark_cumulative_return,
            label=f"{benchmark_name} buy-and-hold",
            linewidth=1.3,
        )
    axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        label=(
            f"Normal-CDF signal ({signal_label}, "
            f"{rebalance_frequency} rebalance, |signal|>{signal_cutoff:.1f}, "
            f"{position_change_threshold_fraction:.0%} threshold, "
            f"{factor_mode} factors)"
        ),
        linewidth=1.5,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(f"{instrument} Trend Strategies — Cumulative PnL Comparison")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative PnL (%)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=150)
    plt.close(figure)

    comparison_start = max(
        [
            results.index[0],
            *(prices.index[0] for prices in aligned_etf_prices.values()),
        ]
    )
    comparison_index = results.index[results.index >= comparison_start]
    comparison_returns: dict[str, pd.Series] = {
        "Normal-CDF signal": results["strategy_return"].reindex(comparison_index),
        trend_strategy_name: benchmark_returns.reindex(comparison_index),
    }
    for benchmark_name, prices in aligned_etf_prices.items():
        buy_and_hold_returns = prices.pct_change(fill_method=None).fillna(0.0)
        buy_and_hold_returns.loc[comparison_start] = 0.0
        comparison_returns[f"{benchmark_name} buy-and-hold"] = (
            buy_and_hold_returns.reindex(comparison_index)
        )

    comparison_rows = []
    for strategy_name, returns in comparison_returns.items():
        comparison_rows.append(
            {
                "instrument": instrument,
                "strategy": strategy_name,
                "comparison_start_date": comparison_index[0].date(),
                "comparison_end_date": comparison_index[-1].date(),
                **calculate_comparison_metrics(returns),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        comparison_table_path,
        index=False,
        float_format="%.6f",
    )


def run_instrument(
    prices: pd.Series,
    strategy: dict[str, str],
    lookback_days: int,
    rebalance_frequency: RebalanceFrequency,
    position_change_threshold_fraction: float,
    signal_cutoff: float,
    signal_mode: SignalMode,
    factor_mode: FactorMode = "none",
    factor_scores: pd.DataFrame | None = None,
    volatility_lookback_days: int = NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS,
    start_date: pd.Timestamp | None = None,
    benchmark_prices: pd.Series | None = None,
    etf_benchmark_prices: dict[str, pd.Series] | None = None,
    position_rounding_increment: float = DEFAULT_POSITION_ROUNDING_INCREMENT,
) -> None:
    """Run one instrument and save daily, summary, and plot outputs."""
    instrument = strategy["instrument"]
    signal_lookbacks = signal_lookbacks_for_mode(signal_mode, lookback_days)
    if signal_mode == "average":
        signal_label = (
            f"average T={','.join(str(days) for days in signal_lookbacks)}"
        )
    elif signal_mode == "rounded":
        signal_label = (
            f"rounded T={lookback_days}, "
            f"position step={position_rounding_increment:g}"
        )
    else:
        signal_label = f"T={lookback_days}"
    if factor_mode == "market-data":
        signal_label = f"{signal_label}, market-data factors (no CTA)"
    if signal_mode == "average":
        lookback_tag = f"avg{'-'.join(str(days) for days in signal_lookbacks)}d"
    elif signal_mode == "rounded":
        rounding_tag = str(position_rounding_increment).replace(".", "p")
        lookback_tag = f"rounded{lookback_days}d_pos{rounding_tag}"
    else:
        lookback_tag = f"{lookback_days}d"
    stem = (
        f"{strategy['stem']}_{rebalance_frequency}_{lookback_tag}_"
        f"vol{volatility_lookback_days}d_"
        f"step{int(position_change_threshold_fraction * 100)}pct_"
        f"sig{int(signal_cutoff * 100)}"
    )
    if factor_mode == "market-data":
        boost_tag = str(FACTOR_SIGNAL_BOOST_MULTIPLIER).replace(".", "p")
        reduce_tag = str(FACTOR_SIGNAL_REDUCE_MULTIPLIER).replace(".", "p")
        stem = (
            f"{stem}_market_factors_zscore_boost{boost_tag}_reduce{reduce_tag}"
        )
    if etf_benchmark_prices:
        stem = f"{stem}_with_etf_benchmarks"
    if start_date is not None:
        stem = f"{stem}_from{start_date:%Y%m%d}"
    results = run_backtest(
        prices,
        lookback_days,
        rebalance_frequency,
        position_change_threshold_fraction,
        signal_cutoff,
        signal_mode,
        factor_scores,
        volatility_lookback_days,
        start_date,
        position_rounding_increment,
    )
    stats_frame = calculate_stats(
        results,
        instrument,
        lookback_days,
        rebalance_frequency,
        position_change_threshold_fraction,
        signal_cutoff,
        signal_mode,
        factor_mode,
        volatility_lookback_days,
        position_rounding_increment,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = OUTPUT_DIR / f"{stem}_daily.csv"
    stats_path = OUTPUT_DIR / f"{stem}_stats.csv"
    simple_stats_path = OUTPUT_DIR / f"{stem}_stats_simple.csv"
    plot_path = OUTPUT_DIR / f"{stem}_cumulative_pnl.png"
    signal_plot_path = OUTPUT_DIR / f"{stem}_signal_rebalance_volume_price.png"
    comparison_path = OUTPUT_DIR / f"{stem}_vs_original_pnl.png"
    comparison_table_path = OUTPUT_DIR / f"{stem}_benchmark_comparison.csv"

    results.to_csv(daily_path, index_label="Date", float_format="%.8f")
    stats_frame.to_csv(stats_path, index=False, float_format="%.6f")
    stats_frame.loc[
        :,
        [
            "period",
            "signal_mode",
            "position_rounding_increment",
            "factor_mode",
            "factor_score_zscore_lookback_days",
            "factor_signal_boost_threshold",
            "factor_signal_boost_multiplier",
            "factor_signal_reduce_multiplier",
            "signal_lookback_days",
            "signal_lookbacks_days",
            "volatility_lookback_days",
            "signal_cutoff",
            "rebalance_position_change_threshold_fraction",
            "rebalance_frequency",
            "sharpe_ratio",
            "annualized_return_pct",
            "annualized_volatility_pct",
            "trading_volume",
            "trading_volume_to_average_absolute_position",
            "max_drawdown_pct",
            "5y_yield_change_percentage_points",
            "spx_change_index_points",
        ],
    ].to_csv(simple_stats_path, index=False, float_format="%.6f")
    save_plots(
        results,
        benchmark_prices if benchmark_prices is not None else prices,
        etf_benchmark_prices if etf_benchmark_prices is not None else {},
        instrument,
        strategy["color"],
        rebalance_frequency,
        position_change_threshold_fraction,
        signal_cutoff,
        signal_label,
        plot_path,
        signal_plot_path,
        comparison_path,
        comparison_table_path,
        factor_mode,
        volatility_lookback_days,
    )

    full_stats = stats_frame.iloc[0]
    print(
        f"{instrument} ({signal_label}, {rebalance_frequency} rebalance): "
        f"{full_stats['data_start_date']} to {full_stats['data_end_date']}, "
        f"Sharpe {full_stats['sharpe_ratio']:.4f}, max drawdown "
        f"{full_stats['max_drawdown_pct']:.2f}%, total return "
        f"{full_stats['total_return_pct']:.2f}%"
    )
    print(f"Daily results: {daily_path}")
    print(f"Stats: {stats_path}")
    print(f"Simplified stats: {simple_stats_path}")
    print(f"PnL plot: {plot_path}")
    print(f"Signal/volume/price plot: {signal_plot_path}")
    print(f"Comparison plot: {comparison_path}")
    print(f"Benchmark comparison table: {comparison_table_path}")


def save_single_future_presentation_plot(
    results: pd.DataFrame,
    dbmf_prices: pd.Series,
    instrument: str,
    output_path: Path,
) -> None:
    """Plot the futures signal above DBMF performance and strategy PnL."""
    aligned_dbmf = dbmf_prices.reindex(results.index, method="ffill")
    aligned_dbmf.loc[aligned_dbmf.index > dbmf_prices.index[-1]] = np.nan
    aligned_dbmf = aligned_dbmf.dropna()
    if aligned_dbmf.empty:
        raise ValueError(
            f"No DBMF Price observations overlap the {instrument} backtest."
        )

    figure, (future_axis, dbmf_axis) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
    )
    signal_axis = future_axis.twinx()
    pnl_axis = dbmf_axis.twinx()
    underlying_change_pct = (
        results["price"] / results["price"].iloc[0] - 1
    ) * 100
    dbmf_change_pct = (aligned_dbmf / aligned_dbmf.iloc[0] - 1) * 100
    underlying_line, = future_axis.plot(
        underlying_change_pct.index,
        underlying_change_pct,
        label=f"{instrument} price change",
        color="#59636E",
        linewidth=1.3,
    )
    signal_line, = signal_axis.plot(
        results.index,
        results["signal"],
        label="Generated normal-CDF signal",
        color="#245a84",
        linewidth=1.1,
    )
    future_axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    signal_axis.axhline(0, color="#245a84", linewidth=0.7, alpha=0.35)
    future_axis.set_title(f"{instrument} Price Change and Generated Signal")
    future_axis.set_ylabel("Future price change (%)")
    signal_axis.set_ylabel("Generated signal")
    future_axis.grid(True, alpha=0.25)
    future_lines = [underlying_line, signal_line]
    future_axis.legend(
        future_lines,
        [line.get_label() for line in future_lines],
        loc="best",
    )

    dbmf_line, = dbmf_axis.plot(
        dbmf_change_pct.index,
        dbmf_change_pct,
        label="DBMF Price change",
        color="#B8860B",
        linewidth=1.3,
    )
    pnl_line, = pnl_axis.plot(
        results.index,
        results["cumulative_pnl_pct"],
        label="Strategy cumulative PnL",
        color="#4f772d",
        linewidth=1.3,
    )
    dbmf_axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    pnl_axis.axhline(0, color="#4f772d", linewidth=0.7, alpha=0.35)
    dbmf_axis.set_title("DBMF Price Change and Strategy PnL")
    dbmf_axis.set_ylabel("DBMF price change (%)")
    pnl_axis.set_ylabel("Strategy cumulative PnL (%)")
    dbmf_axis.set_xlabel("Date")
    dbmf_axis.grid(True, alpha=0.25)
    dbmf_lines = [dbmf_line, pnl_line]
    dbmf_axis.legend(
        dbmf_lines,
        [line.get_label() for line in dbmf_lines],
        loc="best",
    )

    figure.suptitle(f"{instrument} Normal-CDF Trend Backtest")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_single_future_presentation_backtest(
    instrument: str,
    prices: pd.Series,
    dbmf_prices: pd.Series,
    lookback_days: int,
    signal_mode: SignalMode,
    rebalance_frequency: RebalanceFrequency,
    position_change_threshold_fraction: float,
    signal_cutoff: float,
    volatility_lookback_days: int,
    start_date: pd.Timestamp | None,
    factor_mode: FactorMode,
    position_rounding_increment: float = DEFAULT_POSITION_ROUNDING_INCREMENT,
) -> dict[str, object]:
    """Run one futures backtest and save its presentation outputs."""
    strategy = SINGLE_FUTURE_STRATEGIES[instrument]
    factor_scores = (
        load_factor_scores(pd.DatetimeIndex(prices.index))
        if factor_mode == "market-data"
        else None
    )
    results = run_backtest(
        prices,
        lookback_days=lookback_days,
        rebalance_frequency=rebalance_frequency,
        position_change_threshold_fraction=position_change_threshold_fraction,
        signal_cutoff=signal_cutoff,
        signal_mode=signal_mode,
        factor_scores=factor_scores,
        volatility_lookback_days=volatility_lookback_days,
        output_start_date=start_date,
        position_rounding_increment=position_rounding_increment,
    )

    signal_lookbacks = signal_lookbacks_for_mode(signal_mode, lookback_days)
    parameter_values: dict[str, object] = {
        "instrument": instrument,
        "signal_mode": signal_mode,
        "position_rounding_increment": (
            position_rounding_increment if signal_mode == "rounded" else np.nan
        ),
        "factor_mode": factor_mode,
        "signal_lookback_days": (
            np.nan if signal_mode == "average" else lookback_days
        ),
        "signal_lookbacks_days": ",".join(
            str(days) for days in signal_lookbacks
        ),
        "volatility_lookback_days": volatility_lookback_days,
        "signal_cutoff": signal_cutoff,
        "rebalance_position_change_threshold_fraction": (
            position_change_threshold_fraction
        ),
        "rebalance_frequency": rebalance_frequency,
    }
    stats_rows = []
    for stats in [
        calculate_statistics(results),
        *calculate_annual_statistics(results),
    ]:
        stats.update(parameter_values)
        stats_rows.append(stats)
    stats_frame = pd.DataFrame(stats_rows)

    if signal_mode == "average":
        lookback_tag = f"avg{'-'.join(str(days) for days in signal_lookbacks)}d"
    elif signal_mode == "rounded":
        rounding_tag = str(position_rounding_increment).replace(".", "p")
        lookback_tag = f"rounded{lookback_days}d_pos{rounding_tag}"
    else:
        lookback_tag = f"{lookback_days}d"
    stem = (
        f"{strategy['stem']}_normal_cdf_signal_{rebalance_frequency}_"
        f"{lookback_tag}_vol{volatility_lookback_days}d_"
        f"step{int(position_change_threshold_fraction * 100)}pct_"
        f"sig{int(signal_cutoff * 100)}"
    )
    if start_date is not None:
        stem = f"{stem}_from{start_date:%Y%m%d}"
    PRESENTATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = PRESENTATION_OUTPUT_DIR / f"{stem}_daily.csv"
    stats_path = PRESENTATION_OUTPUT_DIR / f"{stem}_stats.csv"
    plot_path = PRESENTATION_OUTPUT_DIR / f"{stem}_presentation.png"
    results.to_csv(daily_path, index_label="Date", float_format="%.8f")
    stats_frame.to_csv(stats_path, index=False, float_format="%.6f")
    save_single_future_presentation_plot(
        results,
        dbmf_prices,
        instrument,
        plot_path,
    )

    full_stats = stats_frame.iloc[0].to_dict()
    if signal_mode == "average":
        signal_label = (
            f"average T={','.join(str(days) for days in signal_lookbacks)}"
        )
    elif signal_mode == "rounded":
        signal_label = (
            f"rounded T={lookback_days}, "
            f"position step={position_rounding_increment:g}"
        )
    else:
        signal_label = f"T={lookback_days}"
    print(
        f"{instrument} ({signal_label}, {rebalance_frequency} rebalance): "
        f"{results.index[0]:%Y-%m-%d} to {results.index[-1]:%Y-%m-%d}, "
        f"Sharpe {full_stats['sharpe_ratio']:.4f}, total return "
        f"{full_stats['total_return_pct']:.2f}%"
    )
    print(f"Daily results: {daily_path}")
    print(f"Stats: {stats_path}")
    print(f"Presentation plot: {plot_path}")
    return full_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instrument",
        choices=tuple(SINGLE_FUTURE_STRATEGIES),
        nargs="+",
        help=(
            "Run the single-instrument futures backtest using future_price.csv "
            "and write presentation outputs. Supply one or more of: "
            + ", ".join(SINGLE_FUTURE_STRATEGIES)
            + ". Omit to run the existing 10Y/basket backtests."
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=positive_int,
        default=DEFAULT_LOOKBACK_DAYS,
        metavar="T",
        help=f"Rolling return lookback in trading days (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--volatility-lookback-days",
        type=positive_int,
        default=NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS,
        metavar="DAYS",
        help=(
            "Rolling realized-volatility window in trading days "
            f"(default: {NORMAL_CDF_VOLATILITY_LOOKBACK_DAYS})."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=iso_date,
        help="First date to include in the backtest, formatted YYYY-MM-DD.",
    )
    parser.add_argument(
        "--signal-mode",
        choices=("single", "average", "rounded"),
        default="single",
        help=(
            "Use one lookback signal, average signals for T="
            f"{','.join(str(days) for days in AVERAGED_SIGNAL_LOOKBACK_DAYS)} "
            "or round volatility-scaled positions to a selected increment "
            "(default: single)."
        ),
    )
    parser.add_argument(
        "--position-rounding-increment",
        type=float,
        choices=POSITION_ROUNDING_INCREMENTS,
        default=DEFAULT_POSITION_ROUNDING_INCREMENT,
        metavar="INCREMENT",
        help=(
            "Position increment for rounded signal mode "
            f"(choices: {', '.join(str(value) for value in POSITION_ROUNDING_INCREMENTS)}; "
            f"default: {DEFAULT_POSITION_ROUNDING_INCREMENT})."
        ),
    )
    parser.add_argument(
        "--factor-mode",
        choices=("none", "market-data"),
        default="none",
        help=(
            "Scale with the signed average market-data factor score: boost "
            f"aligned signals by {FACTOR_SIGNAL_BOOST_MULTIPLIER:g} and reduce "
            f"opposing signals to {FACTOR_SIGNAL_REDUCE_MULTIPLIER:g} when the "
            "factor z-score magnitude exceeds 1."
        ),
    )
    parser.add_argument(
        "--include-etf-benchmarks",
        action="store_true",
        help=(
            "Include DBMF and CTA price-based buy-and-hold benchmarks in the "
            "comparison plot (disabled by default)."
        ),
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=("daily", "weekly", "monthly", "yearly"),
        default=DEFAULT_REBALANCE_FREQUENCY,
        help=(
            "Position rebalance frequency; monthly/yearly rebalance on the "
            "last available date of each period "
            f"(default: {DEFAULT_REBALANCE_FREQUENCY})."
        ),
    )
    parser.add_argument(
        "--rebalance-threshold-fraction",
        type=nonnegative_float,
        default=REBALANCE_POSITION_CHANGE_THRESHOLD_FRACTION,
        metavar="FRACTION",
        help=(
            "Minimum position change as a fraction of target volatility divided "
            f"by realized volatility; use 0 to disable (default: "
            f"{REBALANCE_POSITION_CHANGE_THRESHOLD_FRACTION})."
        ),
    )
    parser.add_argument(
        "--signal-cutoff",
        type=nonnegative_float,
        default=NORMAL_CDF_SIGNAL_CUTOFF,
        metavar="CUTOFF",
        help=(
            "Minimum absolute normal-CDF signal to retain "
            f"(default: {NORMAL_CDF_SIGNAL_CUTOFF})."
        ),
    )
    args = parser.parse_args()
    rebalance_frequency: RebalanceFrequency = args.rebalance_frequency
    signal_mode: SignalMode = args.signal_mode
    factor_mode: FactorMode = args.factor_mode
    if args.instrument:
        if args.include_etf_benchmarks:
            parser.error(
                "--include-etf-benchmarks is not available in single-futures mode; "
                "the presentation plot always compares against DBMF Price."
            )
        presentation_prices = load_single_future_prices()
        comparison_rows = []
        for instrument in args.instrument:
            comparison_rows.append(
                run_single_future_presentation_backtest(
                    instrument,
                    presentation_prices[instrument],
                    presentation_prices["DBMF Price"],
                    args.lookback_days,
                    signal_mode,
                    rebalance_frequency,
                    args.rebalance_threshold_fraction,
                    args.signal_cutoff,
                    args.volatility_lookback_days,
                    args.start_date,
                    factor_mode,
                    args.position_rounding_increment,
                )
            )
        comparison_path = (
            PRESENTATION_OUTPUT_DIR
            / f"normal_cdf_single_future_comparison_{args.start_date:%Y%m%d}.csv"
            if args.start_date is not None
            else PRESENTATION_OUTPUT_DIR / "normal_cdf_single_future_comparison.csv"
        )
        pd.DataFrame(comparison_rows).to_csv(
            comparison_path,
            index=False,
            float_format="%.6f",
        )
        print(f"Instrument comparison: {comparison_path}")
        return

    ten_year_prices = load_10y_prices()["price"]
    spread_prices = load_spread(require_positive_10y_minus_2y=False)[
        "weighted_spread"
    ]
    etf_benchmark_prices = (
        load_etf_benchmark_prices() if args.include_etf_benchmarks else {}
    )
    comparison_benchmark_prices = ten_year_prices
    if args.start_date is not None:
        signal_warmup_observations = (
            max(signal_lookbacks_for_mode(signal_mode, args.lookback_days))
        )
        warmup_observations = max(
            signal_warmup_observations,
            args.volatility_lookback_days,
        )

        def include_warmup(prices: pd.Series) -> pd.Series:
            start_position = int(prices.index.searchsorted(args.start_date))
            warmup_start = max(0, start_position - warmup_observations)
            return prices.iloc[warmup_start:]

        ten_year_prices = include_warmup(ten_year_prices)
        spread_prices = include_warmup(spread_prices)
        if (
            ten_year_prices.empty
            or spread_prices.empty
            or ten_year_prices.index[-1] < args.start_date
            or spread_prices.index[-1] < args.start_date
        ):
            parser.error(
                f"No prices available on or after {args.start_date:%Y-%m-%d} "
                "for both instruments."
            )
    factor_dates = pd.DatetimeIndex(ten_year_prices.index).union(
        pd.DatetimeIndex(spread_prices.index)
    ).sort_values()
    ten_year_factor_scores = None
    spread_factor_scores = None
    if factor_mode == "market-data":
        ten_year_factor_scores = load_factor_scores(
            factor_dates,
            FACTOR_TICKERS,
        )
        spread_factor_scores = load_factor_scores(
            factor_dates,
            SPREAD_FACTOR_TICKERS,
        )
    run_instrument(
        ten_year_prices,
        STRATEGIES["10y_future"],
        args.lookback_days,
        rebalance_frequency,
        args.rebalance_threshold_fraction,
        args.signal_cutoff,
        signal_mode,
        factor_mode,
        ten_year_factor_scores,
        args.volatility_lookback_days,
        args.start_date,
        comparison_benchmark_prices,
        etf_benchmark_prices,
        args.position_rounding_increment,
    )
    run_instrument(
        spread_prices,
        STRATEGIES["10y_2y_spread"],
        args.lookback_days,
        rebalance_frequency,
        args.rebalance_threshold_fraction,
        args.signal_cutoff,
        signal_mode,
        factor_mode,
        spread_factor_scores,
        args.volatility_lookback_days,
        args.start_date,
        comparison_benchmark_prices,
        etf_benchmark_prices,
        args.position_rounding_increment,
    )


if __name__ == "__main__":
    main()
