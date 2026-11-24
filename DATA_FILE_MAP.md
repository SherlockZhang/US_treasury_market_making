# Code-to-data file map

This document maps the repository's analysis and backtest code to the data it
reads. Paths below are relative to the repository root unless stated otherwise.
Generated results (for example files under `results/`) are outputs, not source
data, unless a script explicitly reads them.

## Source data files

| Data file | What the code uses it for |
|---|---|
| `US_future_data.xlsx` | Futures prices and related series used by Treasury backtests and analyses. The code reads `Sheet1`, including the 2Y, 5Y, and 10Y futures, 5Y yield, SPX, and CPI series where needed. |
| `market_data.xlsx` | Market/factor time series used for correlations, regression, neural-network modeling, and instrument-specific normal-CDF factors. The code reads `Sheet1`. |
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
| [`analyze_dbmf_monthly_returns.py`](./analyze_dbmf_monthly_returns.py) | `future_price.csv` (`DBMF Price` and the date column two positions before it) |
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
| [`backtest_normal_cdf_signal.py`](./backtest_normal_cdf_signal.py) | `US_future_data.xlsx` for the existing 10Y/basket backtests and optional SPX signal factor; `market_data.xlsx` for optional TFF2GLFN, CL1, XAU, and instrument-specific 2Y/5Y/10Y market-factor profiles; `future_price.csv` for the optional single 5Y, 2Y, or 10Y futures backtests and DBMF price comparison |
| [`backtest_normal_cdf_sharpe_grid.py`](./backtest_normal_cdf_sharpe_grid.py) | `US_future_data.xlsx`, indirectly through the 10Y futures and spread loaders |
| [`analyze_best_four_futures_strategies.py`](./analyze_best_four_futures_strategies.py) | `future_price.csv` and the existing 2016–2025 best-parameter tables in `results/` |
| [`analyze_best_cdf_market_factor_comparison.py`](./analyze_best_cdf_market_factor_comparison.py) | `future_price.csv` and `market_data.xlsx` |
| [`cta_replication_backtest.py`](./cta_replication_backtest.py) | Yahoo Finance downloads by default; optional user-provided target CSV; generated synthetic data with `--synthetic` |
| [`generate_treasury_pdf.py`](./generate_treasury_pdf.py) | No repository data file is loaded; generates a PDF from content defined by the script |

## Notes

- These scripts resolve their bundled input files relative to the script or
  repository location, so run them with the listed files available in the
  repository.
- Normal-CDF backtests identify the 2Y future by the input price series name
  `2Y future`. Its unfiltered signal is `(8 / 3) * norm.cdf(t_stat) - 4 / 3`;
  for series named `GC`, it is `2 * norm.cdf(t_stat) - 1`. Other instruments
  retain `4 * norm.cdf(t_stat) - 2`. The formula is applied
  before signal cutoffs, market factors, volatility scaling, and position
  rounding, including to each component of an averaged signal. Previously
  saved results are not updated automatically and must be rerun.
- The Normal-CDF Python backtest supports averaging the T=32, 64, 126, 252,
  and 504 signals with `signal_mode="average"`. Average-mode positions are
  unrounded by default; pass `round_average_positions=True` to round the
  averaged signal's resulting positions using `position_rounding_increment`.
  A 0.25 position increment is supported for both single-lookback and averaged
  signals.
- Run `python analyze_best_four_futures_strategies.py` to generate separate 5Y
  and 10Y four-strategy reports under `results/`, including daily results,
  period and annual statistics, plots, and the scaled CDF selection grid. As in
  the 2Y report, both CDF models use weekly rebalancing and 0.5 position
  increments; the scaled model targets 1% annualized volatility before
  rounding. The scaled CDF uses the user-specified T=64 for 5Y; for 10Y, its
  weekly lookback is selected by Sharpe ratio **after rounding** over 32, 64,
  126, 252, and 504 trading days. The grid records both the training ranking
  and the lookback selected for the report. The existing unscaled rounded CDF,
  EWMA, and KAMA best-parameter tables supply the other models. The 2026 results use prior prices
  for indicator warmup but restart positions according to each backtest's
  period-start convention. The 2016–2025 results are in-sample, 2026 is
  out-of-sample. Returns are based on percentage price changes and normalized
  positions, not contract dollar P&L. Annualized return and volatility use 252
  trading days. CAGR is the compounded annual growth rate of net strategy
  wealth: `((ending wealth) ** (365.25 / elapsed calendar days) - 1) * 100`,
  where ending wealth is the product of `(1 + daily net strategy return)` over
  the report period and elapsed calendar days is the difference between the
  first and last observation dates. CAGR is undefined for a nonpositive ending
  wealth or a period with no elapsed calendar days. Unlike CAGR, the existing
  `annualized_return_pct` is the arithmetic mean daily net return multiplied by
  252 trading days.
- The normal-CDF, EWMA, and KAMA backtests deduct daily trading costs for
  recognized single futures: 2Y 0.0042%, 5Y 0.00813%, and 10Y 0.0155% per
  unit of absolute position change. A position reversal incurs both legs;
  the initial position is measured against flat, including at a fresh period
  start. The cost is subtracted from the daily normalized strategy return
  before compounding wealth and calculating P&L and statistics. Daily outputs
  include `position_change`, `trading_cost`, and `gross_strategy_return`.
  The Python backtest functions infer these rates from the input series name
  (`2Y future`, `5Y future`, or `10Y future`); other series default to zero
  cost. Pass `trading_cost_ratio=0` to reproduce a gross backtest or an
  explicit non-negative decimal rate for other inputs. No costs are assumed
  for the multi-leg spread. Existing saved reports and previously selected
  parameter tables are historical gross results until regenerated or
  reselected; a new report can still use parameters selected from those
  gross tables.
- Run `python analyze_2y_all_rank_sum_six_models.py` and
  `python analyze_5y_10y_all_rank_sum_six_models.py` for separate 2Y, 5Y,
  and 10Y six-model reports using trained rank-sum winners for all four CDF
  modes, EWMA, and KAMA. The scripts validate the current CDF grid against
  saved selections and the one-unit trend results against their saved
  rank-sum winners; run `search_2y_cdf_rank_sum.py`,
  `search_5y_10y_cdf_rank_sum.py`, and `search_ewma_kama_rank_sum.py` first
  when the required current selections are missing or stale. Trained reports
  write to `results/*_six_models_all_rank_sum_trained_trends_unit_position_*`.
  Add `--trend-model averaged` to either report command to generate the
  separate user-specified trend reports in
  `results/*_six_models_all_rank_sum_averaged_trends_unit_position_*`.
  Both variants use the same trained CDF winners. Trained trends use
  positions -1, 0, or 1; averaged trends use a 1-unit maximum. Rerunning
  either variant overwrites its generated report files, including any manual
  edits.
- EWMA and KAMA `run_backtest` functions accept a positive `position_size`
  (default `1`). In the averaged-trend six-model reports, fixed user-specified
  pairs replace trend grid winners. Each component produces its own
  rolling-extreme entry signal (-1, 0, or 1) from the indicated future; their
  equally weighted average is applied on the next trading date to the traded
  future. From flat, a half-strength signal opens a half-sized position; a
  subsequent opposite signal exits to flat, while same-direction signals do
  not resize it. Opposing component signals cancel. The position size is 1,
  so possible exposures are -1, -0.5, 0, 0.5, and 1. Returns and costs use
  the traded future, and the 2026 run resets positions but retains earlier
  observations for indicator warmup. Source and traded price dates must match.
  The pairs, using KAMA's slow, fast, ER, M order, are:
  - 2Y: `EWMA(5,40)@2y + EWMA(10,30)@2y`;
    `KAMA(30,5,30,3)@2y + KAMA(30,5,30,3)@5y`.
  - 5Y: `EWMA(5,40)@2y + EWMA(20,20)@10y`;
    `KAMA(10,5,30,3)@10y + KAMA(30,5,30,3)@2y`.
  - 10Y: `EWMA(25,15)@5y + EWMA(15,20)@10y`;
    `KAMA(10,3,30,3)@2y + KAMA(5,5,30,3)@5y`.
  The saved trend parameter files identify these as user-specified, not
  rank-sum winners. Other report families and parameter searches retain their
  original single-signal behavior.
- Four-strategy report statistics include `trading_volume` (the number of
  days with a position change), `position_turnover` (the sum of absolute
  position changes), and `trading_cost_accumulated` (the sum of daily cost
  deductions, in decimal return units). For a single future, accumulated
  cost equals position turnover times its cost ratio; it does not equal
  trade-day count times the ratio when position changes differ from one unit.
  Annual totals preserve positions across calendar-year boundaries.
- To run the fixed 2Y 1.5%-target models without an existing reference report,
  use `python analyze_2y_four_strategies_target_volatility.py --rounding 1
  --skip-reference-validation --output-dir <directory>`. The flag explicitly
  skips reference-parameter validation and uses the script's fixed parameters.
- Run `python search_ewma_kama_trading_cost_grid.py` for cost-adjusted EWMA
  and KAMA searches for all three futures. The current grids contain 64 EWMA
  combinations (`N` and `M` each in 5, 10, 20, 32, 64, 126, 252, 504) and 560
  KAMA combinations (`N_fast` in 2, 3, 5, 10; `N_slow` in 15, 20, 25, 64,
  126, 252, 504; `ER` in 5, 10, 15, 20, 25; `M` in 3, 32, 64, 252) per
  future. The search starts indicators with prices from 2016, stops on
  2025-12-31, and ranks by annualized Sharpe using net daily returns. It writes
  full ranked grids, best-parameter files, and a parameter-space CSV under
  `results/ewma_kama_cost_adjusted_grid_2016_2025/`. Existing gross grids
  and model-selection tables remain unchanged.
- Run `python search_ewma_kama_rank_sum.py` to rank the same configured
  candidates by the sum of Sharpe, maximum-drawdown, and Calmar ranks. It writes
  the rank-sum grids and selected parameters under
  `results/ewma_kama_rank_sum_cost_adjusted_grid_2016_2025/`.
- Run `python analyze_long_future_tstat_correlation.py` to compare each future's
  selected weekly scaled CDF model with a long-one-position benchmark. It buys
  at the first period close, charges the initial entry cost, and holds without
  an exit charge. It reports Pearson correlations of paired daily net returns,
  including flat and indicator-warmup days, for 2016-2025 and 2026 separately.
  The 2026 backtests restart positions; the CDF indicators retain earlier
  prices for warmup. Outputs are saved under
  `results/long_one_future_vs_weekly_scaled_tstat/`. Generate the rank-sum CDF
  winner files first with `search_2y_cdf_rank_sum.py` and
  `search_5y_10y_cdf_rank_sum.py` if they are missing.
- Run `python analyze_ewma_kama_correlation.py` after the long/T-stat analysis
  to correlate daily net returns of the long strategy with the trained rank-sum
  EWMA and KAMA models at unit position size and the weekly scaled T-stat model.
  The report also includes EWMA/KAMA correlation for the same dates.
  Results and paired daily returns are saved under
  `results/ewma_kama_and_long_tstat_correlations/`. The EWMA and KAMA winner
  CSVs from `search_ewma_kama_rank_sum.py` are required.
- Run `python search_cdf_custom_vol_daily_weekly.py` to rank scaled and
  unscaled normal-CDF models for each future over 2016-2025. It tests daily
  and weekly rebalancing and signal lookbacks 32, 64, 126, 252, and 504,
  using a 64-day volatility lookback, zero signal and rebalance thresholds,
  and the currently configured trading-cost ratios. Scaled volatility targets
  are 1.5%, 3%, and 5% for 2Y, 5Y, and 10Y; final positions for both models
  are rounded to 1, 0.5, and 0.5 respectively. The unscaled model does not
  use a volatility target. Outputs are full and best-per-model grids under
  `results/cdf_custom_vol_daily_weekly_grid_2016_2025/`.
- Run `python search_cdf_rounding_grid.py` for the same CDF search with final
  position rounding increments 0.5, 1, 1.5, 2.5, and 3. Results are saved
  separately under `results/cdf_custom_vol_rounding_daily_weekly_grid_2016_2025/`.
  Candidates with zero return volatility have undefined Sharpe and are retained
  unranked with `sharpe_status=undefined_zero_volatility`; they cannot win.
- Run `python search_best_four_win_ratio_grid.py` to rerun the expanded CDF
  and EWMA/KAMA grids and select, for each instrument and model, the highest
  net Sharpe among candidates with a strictly greater than 50% win ratio.
  Win ratio is winning gross-return days divided by winning plus losing
  gross-return days; zero gross-return days are excluded. Four-strategy
  statistics use this gross win ratio; P&L, Sharpe and risk/reward remain net
  of trading costs. All candidates remain in
  the output with eligibility and exclusion status, and unfiltered rank is
  retained separately. Results are saved under
  `results/best_four_gross_win_ratio_above_50_grid_2016_2025/`; previously
  saved net-win-ratio reports are unchanged.
 - Run `python analyze_2y_best_gross_win_filtered_model.py` for the overall
  highest-net-Sharpe 2Y winner from those grids. It verifies the current model
  settings and training metrics, then writes daily, annual, summary, and plot
  outputs under `results/2y_best_gross_win_filtered_model_2016_2025_and_2026/`.
  The 2026 run resets the position and uses earlier prices for indicator warmup.
- Run `python analyze_2y_best_four_gross_win_filtered.py` for the four 2Y
  winners from the gross-win-ratio-filtered grids, with training Sharpe and
  gross win ratio checked against the selected rows. The comparison and daily
  results are saved under
  `results/2y_best_four_gross_win_filtered_2016_2025_and_2026/`; 2026 resets
  positions and retains earlier prices for indicator warmup.
- Run `python analyze_5y_10y_best_four_gross_win_filtered.py` for the
  equivalent 5Y and 10Y comparisons, with each instrument's target volatility,
  trading cost, and filtered grid winners. Results are saved in the corresponding
  `5y_best_four_gross_win_filtered_2016_2025_and_2026` and
  `10y_best_four_gross_win_filtered_2016_2025_and_2026` directories.
- Run `python analyze_10y_four_weekly_t64_rounding_0p5.py` to replace only the
  10Y scaled CDF model with the eligible rank-3 weekly T=64, rounding=0.5
  candidate. The selected grid row and the four-strategy report are saved under
  `results/10y_four_gross_filtered_weekly_t64_rounding_0p5_2016_2025_and_2026/`.
- Run `python analyze_selected_cdf_market_factor_comparison.py` for the
  2026 factor versus no-factor comparison of the filtered 2Y and 5Y scaled
  winners and the selected rank-3 10Y scaled model. Outputs include factor
  source-date coverage under `results/cdf_2026_gross_filtered_winners_market_factor_comparison/`.
  Factor signals use the existing weekly forward-fill logic, and the enhanced
  variants are not separately optimized.
- Run `python analyze_10y_filtered_four_strategies_market_factors.py` for the
  2026 comparison of the four selected 10Y strategies with market factors
  applied only to the rank-3 scaled CDF model. Results are saved under
  `results/10y_four_gross_filtered_scaled_market_factors_2026/`.
  Four-strategy price/position plots use separate axes for price and position;
  daily green/red bars extend above/below zero in proportion to position units,
  with each strategy's position axis symmetric about zero. No bar is drawn for
  flat positions or dates without observations, including non-trading days.
- Run `python analyze_2y_cost_adjusted_best_strategies.py` to reproduce the
  2Y four-strategy report using the cost-adjusted CDF, EWMA, and KAMA grid
  winners. The script checks training Sharpe and the current cost ratio against
  the saved winners before generating 2016-2025 and 2026 reports; CDF
  frequencies follow the selected models independently.
- Run `python analyze_2y_expanded_cdf_grid_winners.py` for the same report
  using the expanded CDF rounding-grid winners and cost-adjusted EWMA/KAMA
  winners. Scaled and unscaled CDF models use their independently selected
  rounding increments and rebalance frequencies.
- Run `python analyze_5y_10y_expanded_cdf_grid_winners.py` for equivalent
  5Y and 10Y reports using the expanded CDF and cost-adjusted EWMA/KAMA
  winners. Each instrument retains its own target, costs, selected rounding,
  and rebalance frequencies; saved training Sharpe ratios are verified.
- Run `python analyze_best_cdf_market_factor_comparison.py` to compare the
  rounded weekly scaled CDF models with and without their configured
  instrument-specific market factors in 2026. The lookbacks match the
  four-strategy reports (currently 64 days for all three futures); 2Y and 5Y
  use the user-specified overrides and 10Y uses the rounded 2016–2025 training grid.
  The comparison does not select on 2026 performance or separately optimize
  the factor-adjusted model. The summary records model parameters and their
  selection source. Both variants restart their positions in 2026. Outputs
  are written to `results/cdf_four_strategy_model_2026_market_factor_comparison/`.
  The original 64-day comparison remains unchanged. Factor values are aligned
  using the existing weekly forward-fill logic, so the dates and freshness of
  the underlying market data should be checked when interpreting results.
- To generate the single-futures presentation reports, run
  `python backtest_normal_cdf_signal.py --instrument "5Y future" "2Y future" "10Y future" --start-date 2026-01-01 --lookback-days 64 --volatility-lookback-days 64 --rebalance-frequency weekly --rebalance-threshold-fraction 0 --signal-cutoff 0`.
  The daily results, statistics, and plots are written to `presentation report/`.
- To apply the configured 2Y, 5Y, and 10Y instrument-specific market-factor
  profiles, run
  `python backtest_normal_cdf_signal.py --instrument "2Y future" "5Y future" "10Y future" --signal-factor-mode instrument`.
  The factors trigger when their weekly return or point change deviates by more
  than one standard deviation from the preceding 52 weekly observations.
  The configured profiles are:

  | Future | Same-sign increases position (probability) | Different-sign increases position (probability) |
  |---|---|---|
  | 2Y | TFF2GLFN (0.70), CTA US (0.60) | SPX (0.59), TFF2ILFN (0.58), CL1 (0.56) |
  | 5Y | TFF2GLFN (0.71), CTA US (0.57), UKX (0.57) | CL1 (0.54), SPX (0.56), TFF2ILFN (0.58), SOFRRATE (0.68) |
  | 10Y | TFF2GLFN (0.71), UKX (0.60), CTA US (0.57) | SOFRRATE (0.58), TFF2HLFN (0.61) |

  TFF2GLFN, TFF2ILFN, SOFRRATE, and TFF2HLFN use weekly point changes;
  CTA US, SPX, CL1, and UKX use weekly percentage returns. Active multipliers
  combine multiplicatively.
- To calculate monthly DBMF Price returns and the positive-month win rate from
  2016-01-01, run `python analyze_dbmf_monthly_returns.py`. The monthly returns
  and win-rate summary, monthly return plot, and daily-return-based annual
  statistics from 2020 are written to `component_analysis/results/`. The source
  series starts in 2019, so no observations are available for 2016–2019.
- CSVs, spreadsheets, plots, and PDFs created in `results/`,
  `results_refining_parameter/`, `presentation report/`, `component_analysis/results/`,
  `market_data_regression/`, or other output folders are analysis artifacts.
  Their presence does not necessarily mean another script consumes them.
- `US_Treasury_Futures_and_Options_Exchanges.pdf` is a reference document in
  the repository; the mapped analysis scripts do not load it as an input.
