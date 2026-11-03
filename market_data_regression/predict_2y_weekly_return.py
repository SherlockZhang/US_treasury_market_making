"""Forecast 2Y futures weekly returns from lagged JPYUSD and DAX returns."""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_market_data_regression import load_ticker_series


OUTPUT_DIR = Path(__file__).resolve().parent
PREDICTIONS_OUTPUT_FILE = OUTPUT_DIR / "2y_weekly_return_oos_predictions.csv"
SUMMARY_OUTPUT_FILE = OUTPUT_DIR / "2y_weekly_return_model_summary.csv"
TARGET_TICKER = "2Y future"
PREDICTOR_TICKERS = ("JPYUSD BGN Curncy", "DAX")
TRAIN_FRACTION = 0.8
WEEK_FREQUENCY = "W-FRI"


def weekly_returns(prices: pd.Series) -> pd.Series:
    """Calculate returns between consecutive Friday-ending weekly closes."""
    weekly_closes = prices.resample(WEEK_FREQUENCY).last()
    return weekly_closes.pct_change(fill_method=None).mul(100)


def fit_ols(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit OLS with an intercept and return intercept followed by coefficients."""
    design = np.column_stack((np.ones(len(features)), features))
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    if not np.isfinite(coefficients).all():
        raise ValueError("OLS produced non-finite coefficients.")
    return coefficients


def calculate_metrics(
    actual: np.ndarray, predicted: np.ndarray, zero_forecast: np.ndarray
) -> dict[str, float]:
    """Calculate out-of-sample forecast metrics."""
    residuals = actual - predicted
    zero_residuals = actual - zero_forecast
    actual_variance = np.sum((actual - actual.mean()) ** 2)
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 0 and np.std(predicted) > 0
        else np.nan
    )
    return {
        "rmse_pct": float(np.sqrt(np.mean(residuals**2))),
        "mae_pct": float(np.mean(np.abs(residuals))),
        "out_of_sample_r2": (
            float(1 - np.sum(residuals**2) / actual_variance)
            if actual_variance > 0
            else np.nan
        ),
        "prediction_correlation": correlation,
        "directional_accuracy_pct": (
            float(np.mean(np.sign(actual) == np.sign(predicted)) * 100)
            if np.any(predicted != 0)
            else np.nan
        ),
        "zero_forecast_rmse_pct": float(np.sqrt(np.mean(zero_residuals**2))),
        "zero_forecast_mae_pct": float(np.mean(np.abs(zero_residuals))),
    }


def main() -> None:
    price_series = load_ticker_series()
    missing = {TARGET_TICKER, *PREDICTOR_TICKERS}.difference(price_series)
    if missing:
        raise ValueError(f"Missing required series: {sorted(missing)}")

    target_returns = weekly_returns(price_series[TARGET_TICKER]).rename(
        "2y_future_return_pct"
    )
    predictor_returns = pd.DataFrame(
        {
            ticker: weekly_returns(price_series[ticker])
            for ticker in PREDICTOR_TICKERS
        }
    )
    predictor_returns = predictor_returns.shift(1).rename(
        columns={
            ticker: f"prior_week_{ticker}_return_pct"
            for ticker in PREDICTOR_TICKERS
        }
    )
    model_data = pd.concat(
        [predictor_returns, target_returns], axis=1, join="inner"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if model_data.empty:
        raise ValueError("No complete lagged weekly observations are available.")

    training_count = int(len(model_data) * TRAIN_FRACTION)
    if training_count < len(PREDICTOR_TICKERS) + 2:
        raise ValueError("Too few weekly observations for training and testing.")
    if training_count >= len(model_data):
        raise ValueError("Too few weekly observations for an out-of-sample test.")

    feature_columns = list(predictor_returns.columns)
    test = model_data.iloc[training_count:]
    features = model_data.loc[:, feature_columns].to_numpy(dtype=float)
    target = model_data["2y_future_return_pct"].to_numpy(dtype=float)
    predictions = np.empty(len(test), dtype=float)
    coefficients_by_forecast = np.empty(
        (len(test), len(PREDICTOR_TICKERS) + 1), dtype=float
    )

    for test_offset, row_position in enumerate(
        range(training_count, len(model_data))
    ):
        coefficients = fit_ols(
            features[:row_position],
            target[:row_position],
        )
        coefficients_by_forecast[test_offset] = coefficients
        predictions[test_offset] = (
            coefficients[0] + features[row_position] @ coefficients[1:]
        )

    actual = test["2y_future_return_pct"].to_numpy(dtype=float)
    zero_forecast = np.zeros(len(test), dtype=float)
    prediction_frame = test.copy()
    prediction_frame["predicted_2y_future_return_pct"] = predictions
    prediction_frame["zero_return_forecast_pct"] = zero_forecast
    prediction_frame["ols_intercept"] = coefficients_by_forecast[:, 0]
    for predictor_index, ticker in enumerate(PREDICTOR_TICKERS, start=1):
        prediction_frame[f"ols_coef_{ticker}"] = coefficients_by_forecast[
            :, predictor_index
        ]
    prediction_frame.index.name = "week_ending"
    prediction_frame.to_csv(
        PREDICTIONS_OUTPUT_FILE,
        float_format="%.8f",
    )

    model_metrics = calculate_metrics(actual, predictions, zero_forecast)
    zero_metrics = calculate_metrics(actual, zero_forecast, zero_forecast)
    summary_rows = []
    for model_name, metrics in (
        ("Expanding OLS: lagged JPYUSD + DAX", model_metrics),
        ("Zero-return forecast", zero_metrics),
    ):
        summary_rows.append(
            {
                "model": model_name,
                "target": TARGET_TICKER,
                "predictors": ", ".join(PREDICTOR_TICKERS),
                "predictor_timing": "prior week's return predicts target week's return",
                "week_definition": "W-FRI last available close",
                "training_fraction": TRAIN_FRACTION,
                "initial_training_weeks": training_count,
                "out_of_sample_weeks": len(test),
                "out_of_sample_start_week": test.index.min().date(),
                "out_of_sample_end_week": test.index.max().date(),
                **metrics,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
        float_format="%.8f",
    )

    print(
        f"Weekly observations: {len(model_data)}; training: {training_count}; "
        f"out of sample: {len(test)} "
        f"({test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d})"
    )
    print(pd.DataFrame(summary_rows).loc[:, [
        "model",
        "rmse_pct",
        "mae_pct",
        "out_of_sample_r2",
        "prediction_correlation",
        "directional_accuracy_pct",
        "zero_forecast_rmse_pct",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Predictions: {PREDICTIONS_OUTPUT_FILE}")
    print(f"Summary: {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
