# Code-to-data file map

This document maps the repository's analysis and backtest code to the data it
reads. Paths below are relative to the repository root unless stated otherwise.
Generated results (for example files under `results/`) are outputs, not source
data, unless a script explicitly reads them.

## Source data files

| Data file | What the code uses it for |
|---|---|
| `US_future_data.xlsx` | Futures prices and related series used by Treasury backtests and analyses. The code reads `Sheet1`, including the 2Y, 5Y, and 10Y futures, 5Y yield, SPX, and CPI series where needed. |
| `market_data.xlsx` | Market/factor time series used for market-data correlations, regression, neural-network modeling, and optional factor signals. The code reads `Sheet1`. |
| `DBMF_COMP.csv` | Date-indexed DBMF, SPX, and IEF price data for DBMF correlation/performance and volatility analyses. |
| `bbg_data.xlsx` | Bloomberg workbook data for DBMF, risk-free yield, and SPX Sharpe-ratio analysis. The code reads `Sheet1`. |
| `future_price.csv` | Repeated date/price column pairs for futures and other components (including DBMF), used to calculate aligned returns and fit component regressions. |
| External/user-provided input to `cta_replication_backtest.py` | By default the script downloads factor and target prices from Yahoo Finance. `--target <file.csv>` instead supplies the target series from a CSV; `--synthetic` generates smoke-test data and avoids downloads. |

## Code-to-source mapping

| Code | Source data used |
|---|---|
| [`analyze_dbmf_correlation.py`](./analyze_dbmf_correlation.py) | `DBMF_COMP.csv` |
| [`analyze_dbmf_garch_volatility.py`](./analyze_dbmf_garch_volatility.py) | `DBMF_COMP.csv` |
| [`analyze_dbmf_sharpe.py`](./analyze_dbmf_sharpe.py) | `bbg_data.xlsx` |
| [`analyze_future_data.py`](./analyze_future_data.py) | `future_price.csv` |
| [`analyze_rolling_future_regression.py`](./analyze_rolling_future_regression.py) | `future_price.csv`, indirectly through `analyze_future_data.py` |
| [`component_analysis/analyze_component_portfolio.py`](./component_analysis/analyze_component_portfolio.py) | `future_price.csv`, indirectly through `analyze_future_data.py` |
| [`analyze_market_data_correlations.py`](./analyze_market_data_correlations.py) | `market_data.xlsx` |
| [`market_data_regression/analyze_market_data_regression.py`](./market_data_regression/analyze_market_data_regression.py) | `market_data.xlsx` |
| [`market_data_regression/analyze_market_data_neural_network.py`](./market_data_regression/analyze_market_data_neural_network.py) | `market_data.xlsx`, indirectly through `analyze_market_data_regression.py` |
| [`backtest_5y_trend_strategy.py`](./backtest_5y_trend_strategy.py) | `US_future_data.xlsx` |
| [`backtest_2y_10y_spread_strategy.py`](./backtest_2y_10y_spread_strategy.py) | `US_future_data.xlsx` |
| [`backtest_factor_modified.py`](./backtest_factor_modified.py) | `US_future_data.xlsx` |
| [`backtest_12_month_average_signal.py`](./backtest_12_month_average_signal.py) | `US_future_data.xlsx`, indirectly through the 5Y trend and 2Y/10Y spread loaders |
| [`analyze_5y_future_yield_correlation.py`](./analyze_5y_future_yield_correlation.py) | `US_future_data.xlsx`, indirectly through the 5Y trend loaders |
| [`backtest_normal_cdf_signal.py`](./backtest_normal_cdf_signal.py) | `US_future_data.xlsx`; `market_data.xlsx` when market-data factor signals are used |
| [`backtest_normal_cdf_sharpe_grid.py`](./backtest_normal_cdf_sharpe_grid.py) | `US_future_data.xlsx`, indirectly through the 10Y futures and spread loaders |
| [`cta_replication_backtest.py`](./cta_replication_backtest.py) | Yahoo Finance downloads by default; optional user-provided target CSV; generated synthetic data with `--synthetic` |
| [`generate_treasury_pdf.py`](./generate_treasury_pdf.py) | No repository data file is loaded; generates a PDF from content defined by the script |

## Notes

- These scripts resolve their bundled input files relative to the script or
  repository location, so run them with the listed files available in the
  repository.
- CSVs, spreadsheets, plots, and PDFs created in `results/`,
  `results_refining_parameter/`, `component_analysis/results/`,
  `market_data_regression/`, or other output folders are analysis artifacts.
  Their presence does not necessarily mean another script consumes them.
- `US_Treasury_Futures_and_Options_Exchanges.pdf` is a reference document in
  the repository; the mapped analysis scripts do not load it as an input.
