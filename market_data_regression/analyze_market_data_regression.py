"""Fit OLS, cross-validated Ridge, and Lasso models to market-data returns."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "market_data.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_DIR = Path(__file__).resolve().parent
STATS_OUTPUT_FILE = OUTPUT_DIR / "regression_fit_stats.csv"
PREDICTIONS_OUTPUT_FILE = OUTPUT_DIR / "holdout_fitted_values.csv"
TOP_TSTAT_OUTPUT_FILE = OUTPUT_DIR / "top_ols_t_statistics.csv"

FIVE_YEAR_TARGET = "5Y future"
TWO_YEAR_TICKER = "2Y future"
TEN_YEAR_TICKER = "10Y future"
SOFR_TICKER = "SOFRRATE Index"
WEIGHTED_BASKET_TARGET = "5x2Y - 3x10Y futures basket"
WEEKLY_TFF_TICKERS = {
    "TFF2ILFN Index",
    "TFF2GLFN Index",
    "TFF2HLFN Index",
}
EXCLUDED_PREDICTORS = {
    FIVE_YEAR_TARGET,
    TWO_YEAR_TICKER,
    TEN_YEAR_TICKER,
    "CTA US Equity",
}
TRAIN_FRACTION = 0.8
TIME_SERIES_CV_SPLITS = 5
RIDGE_ALPHA_CANDIDATES = np.logspace(-4, 4, 33)
LASSO_ALPHA_RATIOS = np.logspace(-4, 0, 33)
LASSO_MAX_ITERATIONS = 10000
LASSO_TOLERANCE = 1e-10
ROLLING_ZSCORE_WINDOW = 63
TOP_PREDICTOR_COUNT = 10


def load_ticker_series() -> dict[str, pd.Series]:
    """Load ticker values from adjacent date/value columns in the workbook."""
    frame = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    columns = list(frame.columns)
    ticker_series: dict[str, pd.Series] = {}

    for column_index, ticker in enumerate(columns):
        if not isinstance(ticker, str) or ticker.startswith("Unnamed:"):
            continue
        if column_index == 0:
            continue
        date_column = columns[column_index - 1]
        if not isinstance(date_column, str) or not date_column.strip().startswith(
            "Date"
        ):
            continue

        observations = frame.loc[:, [date_column, ticker]].copy()
        observations[date_column] = pd.to_datetime(
            observations[date_column], errors="raise"
        )
        observations[ticker] = pd.to_numeric(
            observations[ticker], errors="raise"
        )
        partially_blank = observations.isna().any(axis=1) & observations.notna().any(
            axis=1
        )
        if partially_blank.any():
            raise ValueError(
                f"{ticker} contains rows with only a date or only a value."
            )
        observations = observations.dropna(how="all")
        if observations.empty:
            raise ValueError(f"No observations found for {ticker}.")
        if observations[date_column].duplicated().any():
            raise ValueError(f"{ticker} contains duplicate dates.")
        if not np.isfinite(observations[ticker]).all():
            raise ValueError(f"{ticker} values must be finite.")

        ticker_series[ticker] = (
            observations.sort_values(date_column)
            .set_index(date_column)[ticker]
            .rename(ticker)
        )

    required = {FIVE_YEAR_TARGET, TWO_YEAR_TICKER, TEN_YEAR_TICKER, SOFR_TICKER}
    missing = required.difference(ticker_series)
    if missing:
        raise ValueError(f"Missing required ticker series: {sorted(missing)}")
    return ticker_series


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Scale each value using the preceding observations, excluding itself."""
    prior_values = series.shift(1)
    rolling_mean = prior_values.rolling(window=window, min_periods=window).mean()
    rolling_std = prior_values.rolling(window=window, min_periods=window).std(
        ddof=0
    )
    scaled = (series - rolling_mean) / rolling_std.where(rolling_std > 0)
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    scaled.name = series.name
    return scaled


def build_model_data(
    ticker_series: dict[str, pd.Series],
    exclude_weekly_tff: bool = False,
    scaling_window: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Build complete-case predictors and target daily-return series."""
    predictors: dict[str, pd.Series] = {}
    for ticker, values in ticker_series.items():
        if ticker in EXCLUDED_PREDICTORS or (
            exclude_weekly_tff and ticker in WEEKLY_TFF_TICKERS
        ):
            continue
        if ticker == SOFR_TICKER:
            predictor = values.diff().rename(ticker)
        else:
            predictor = values.pct_change(fill_method=None).rename(ticker)
        predictors[ticker] = (
            rolling_zscore(predictor, scaling_window)
            if scaling_window is not None
            else predictor
        )

    two_year, ten_year = ticker_series[TWO_YEAR_TICKER].align(
        ticker_series[TEN_YEAR_TICKER], join="inner"
    )
    basket_price = 5 * two_year - 3 * ten_year
    if basket_price.empty or not np.isfinite(basket_price).all():
        raise ValueError("The weighted futures basket has invalid price data.")
    if (basket_price <= 0).any():
        raise ValueError("The weighted futures basket must remain positive.")

    targets = {
        FIVE_YEAR_TARGET: ticker_series[FIVE_YEAR_TARGET]
        .pct_change(fill_method=None)
        .rename(FIVE_YEAR_TARGET),
        WEIGHTED_BASKET_TARGET: basket_price.pct_change(fill_method=None).rename(
            WEIGHTED_BASKET_TARGET
        ),
    }
    if scaling_window is not None:
        targets = {
            name: rolling_zscore(series, scaling_window)
            for name, series in targets.items()
        }
    feature_data = pd.concat(list(predictors.values()), axis=1, join="inner")
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(feature_data) <= len(predictors) + TIME_SERIES_CV_SPLITS + 1:
        raise ValueError(
            "Too few complete observations for the requested predictors and "
            "time-series cross-validation."
        )
    return feature_data.loc[:, list(predictors)], targets


def chronological_splits(
    row_count: int, split_count: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window validation splits from a training sample."""
    validation_size = row_count // (split_count + 1)
    if validation_size == 0:
        raise ValueError("Too few training observations for time-series CV.")
    splits = []
    for split_index in range(split_count):
        validation_start = (split_index + 1) * validation_size
        validation_end = (
            (split_index + 2) * validation_size
            if split_index < split_count - 1
            else row_count
        )
        train_indices = np.arange(validation_start)
        validation_indices = np.arange(validation_start, validation_end)
        if len(train_indices) <= 1 or len(validation_indices) == 0:
            raise ValueError("Invalid time-series cross-validation split.")
        splits.append((train_indices, validation_indices))
    return splits


def fit_linear_model(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an intercept model with predictors standardized on its train sample."""
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0, ddof=0)
    if not np.isfinite(feature_scale).all() or (feature_scale == 0).any():
        raise ValueError("Predictors must have finite, non-zero training variance.")
    standardized = (features - feature_mean) / feature_scale
    target_mean = target.mean()
    centered_target = target - target_mean

    if alpha == 0:
        coefficients = np.linalg.lstsq(
            standardized, centered_target, rcond=None
        )[0]
    else:
        coefficients = np.linalg.solve(
            standardized.T @ standardized
            + alpha * np.eye(standardized.shape[1]),
            standardized.T @ centered_target,
        )
    intercept = target_mean - feature_mean @ (coefficients / feature_scale)
    raw_coefficients = coefficients / feature_scale
    return raw_coefficients, np.asarray(intercept), feature_mean


def predict(
    features: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    return features @ coefficients + float(intercept)


def select_ridge_alpha(
    features: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Choose Ridge alpha by mean validation MSE over expanding train windows."""
    splits = chronological_splits(len(target), TIME_SERIES_CV_SPLITS)
    alpha_scores = []
    for alpha in RIDGE_ALPHA_CANDIDATES:
        fold_mse = []
        for train_indices, validation_indices in splits:
            coefficients, intercept, _ = fit_linear_model(
                features[train_indices],
                target[train_indices],
                float(alpha),
            )
            validation_prediction = predict(
                features[validation_indices], coefficients, intercept
            )
            fold_mse.append(
                float(
                    np.mean(
                        (target[validation_indices] - validation_prediction) ** 2
                    )
                )
            )
        alpha_scores.append(float(np.mean(fold_mse)))
    best_index = int(np.argmin(alpha_scores))
    return float(RIDGE_ALPHA_CANDIDATES[best_index]), alpha_scores[best_index]


def lasso_alpha_max(features: np.ndarray, target: np.ndarray) -> float:
    """Return the Lasso penalty that shrinks all standardized slopes to zero."""
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0, ddof=0)
    if not np.isfinite(feature_scale).all() or (feature_scale == 0).any():
        raise ValueError("Predictors must have finite, non-zero training variance.")
    standardized = (features - feature_mean) / feature_scale
    centered_target = target - target.mean()
    return float(np.max(np.abs(standardized.T @ centered_target)) / len(target))


def fit_lasso_model(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardized Lasso by cyclic coordinate descent."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Lasso alpha must be finite and non-negative.")
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0, ddof=0)
    if not np.isfinite(feature_scale).all() or (feature_scale == 0).any():
        raise ValueError("Predictors must have finite, non-zero training variance.")

    standardized = (features - feature_mean) / feature_scale
    centered_target = target - target.mean()
    sample_count = len(target)
    column_norms = np.sum(standardized**2, axis=0) / sample_count
    coefficients = np.zeros(standardized.shape[1], dtype=float)
    residual = centered_target.copy()

    for _ in range(LASSO_MAX_ITERATIONS):
        largest_update = 0.0
        for column_index in range(standardized.shape[1]):
            column = standardized[:, column_index]
            residual += column * coefficients[column_index]
            correlation = float(column @ residual) / sample_count
            updated = np.sign(correlation) * max(
                abs(correlation) - alpha, 0.0
            ) / column_norms[column_index]
            residual -= column * updated
            largest_update = max(
                largest_update, abs(updated - coefficients[column_index])
            )
            coefficients[column_index] = updated
        if largest_update <= LASSO_TOLERANCE:
            break
    else:
        raise RuntimeError(
            f"Lasso coordinate descent did not converge in "
            f"{LASSO_MAX_ITERATIONS} iterations."
        )

    intercept = target.mean() - feature_mean @ (coefficients / feature_scale)
    return coefficients / feature_scale, np.asarray(intercept), feature_mean


def select_lasso_alpha(
    features: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Choose the Lasso penalty by expanding-window validation MSE."""
    splits = chronological_splits(len(target), TIME_SERIES_CV_SPLITS)
    alpha_scores = []
    for alpha_ratio in LASSO_ALPHA_RATIOS:
        fold_mse = []
        for train_indices, validation_indices in splits:
            fold_features = features[train_indices]
            fold_target = target[train_indices]
            fold_alpha = float(
                alpha_ratio * lasso_alpha_max(fold_features, fold_target)
            )
            coefficients, intercept, _ = fit_lasso_model(
                fold_features, fold_target, fold_alpha
            )
            validation_prediction = predict(
                features[validation_indices], coefficients, intercept
            )
            fold_mse.append(
                float(
                    np.mean(
                        (target[validation_indices] - validation_prediction) ** 2
                    )
                )
            )
        alpha_scores.append(float(np.mean(fold_mse)))

    best_index = int(np.argmin(alpha_scores))
    selected_alpha = float(
        LASSO_ALPHA_RATIOS[best_index] * lasso_alpha_max(features, target)
    )
    return selected_alpha, alpha_scores[best_index]


def calculate_metrics(actual: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    residuals = actual - fitted
    total_sum_squares = np.sum((actual - actual.mean()) ** 2)
    r_squared = (
        1 - np.sum(residuals**2) / total_sum_squares
        if total_sum_squares > 0
        else np.nan
    )
    return {
        "r_squared": float(r_squared),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def calculate_ols_t_statistics(
    features: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Calculate classical OLS t-statistics for standardized predictors."""
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0, ddof=0)
    if not np.isfinite(feature_scale).all() or (feature_scale == 0).any():
        raise ValueError("Predictors must have finite, non-zero training variance.")
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(len(target)), standardized])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residuals = target - design @ coefficients
    degrees_of_freedom = len(target) - np.linalg.matrix_rank(design)
    if degrees_of_freedom <= 0:
        raise ValueError("OLS has no residual degrees of freedom.")
    residual_variance = float(residuals @ residuals) / degrees_of_freedom
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_statistics = np.divide(
        coefficients[1:],
        standard_errors[1:],
        out=np.full(len(feature_scale), np.nan),
        where=standard_errors[1:] > 0,
    )
    return t_statistics


def fit_target(
    features: pd.DataFrame,
    target: pd.Series,
    target_name: str,
    scaling_name: str,
) -> tuple[
    list[dict[str, object]],
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
]:
    """Fit all models for one target and return predictions and top predictors."""
    aligned = pd.concat([features, target], axis=1, join="inner").dropna()
    X = aligned.loc[:, features.columns].to_numpy(dtype=float)
    y = aligned[target_name].to_numpy(dtype=float)
    split_index = int(len(aligned) * TRAIN_FRACTION)
    if split_index <= X.shape[1] or len(aligned) - split_index < 2:
        raise ValueError(f"Not enough observations to fit {target_name}.")

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    train_index = aligned.index[:split_index]
    test_index = aligned.index[split_index:]
    rows: list[dict[str, object]] = []
    t_statistics = calculate_ols_t_statistics(X_train, y_train)
    ols_coefficients, ols_intercept, _ = fit_linear_model(X_train, y_train)
    top_predictor_stats: list[dict[str, object]] = []
    train_prediction_frame = pd.DataFrame(
        {"actual": y_train},
        index=train_index,
    )
    prediction_frame = pd.DataFrame(
        {"actual": y_test},
        index=test_index,
    )

    ridge_alpha, ridge_cv_mse = select_ridge_alpha(X_train, y_train)
    lasso_alpha, lasso_cv_mse = select_lasso_alpha(X_train, y_train)
    models = (
        ("OLS", 0.0, np.nan, np.nan),
        ("Ridge", ridge_alpha, ridge_cv_mse, np.nan),
        ("Lasso", lasso_alpha, np.nan, lasso_cv_mse),
    )
    for model_name, penalty, ridge_validation_mse, lasso_validation_mse in models:
        if model_name == "OLS":
            coefficients, intercept = ols_coefficients, ols_intercept
        elif model_name == "Lasso":
            coefficients, intercept, _ = fit_lasso_model(
                X_train, y_train, penalty
            )
        else:
            coefficients, intercept, _ = fit_linear_model(
                X_train, y_train, penalty
            )
        standardized_coefficients = coefficients * X_train.std(axis=0, ddof=0)
        if model_name == "OLS":
            ranking_statistics = t_statistics
            statistic_type = "OLS t-statistic"
        else:
            ranking_statistics = standardized_coefficients
            statistic_type = "standardized coefficient"
        top_indices = np.argsort(np.abs(ranking_statistics))[::-1][
            :TOP_PREDICTOR_COUNT
        ]
        top_predictor_stats.extend(
            {
                "target": target_name,
                "model": model_name,
                "scaling": scaling_name,
                "predictor": features.columns[index],
                "coefficient": float(coefficients[index]),
                "standardized_coefficient": float(
                    standardized_coefficients[index]
                ),
                "statistic_type": statistic_type,
                "ranking_statistic": float(ranking_statistics[index]),
                "absolute_ranking_statistic": float(
                    abs(ranking_statistics[index])
                ),
                "t_statistic": (
                    float(t_statistics[index]) if model_name == "OLS" else np.nan
                ),
                "absolute_t_statistic": (
                    float(abs(t_statistics[index]))
                    if model_name == "OLS"
                    else np.nan
                ),
                "train_observations": len(y_train),
                "train_start_date": train_index.min().date(),
                "train_end_date": train_index.max().date(),
            }
            for index in top_indices
        )
        train_prediction = predict(X_train, coefficients, intercept)
        test_prediction = predict(X_test, coefficients, intercept)
        train_prediction_frame[f"{model_name.lower()}_fitted"] = train_prediction
        prediction_frame[f"{model_name.lower()}_fitted"] = test_prediction

        train_metrics = calculate_metrics(y_train, train_prediction)
        test_metrics = calculate_metrics(y_test, test_prediction)
        rows.append(
            {
                "target": target_name,
                "model": model_name,
                "scaling": scaling_name,
                "predictor_count": X.shape[1],
                "train_observations": len(y_train),
                "test_observations": len(y_test),
                "train_start_date": train_index.min().date(),
                "train_end_date": train_index.max().date(),
                "test_start_date": test_index.min().date(),
                "test_end_date": test_index.max().date(),
                "ridge_alpha": penalty if model_name == "Ridge" else np.nan,
                "ridge_cv_mean_validation_mse": ridge_validation_mse,
                "lasso_alpha": penalty if model_name == "Lasso" else np.nan,
                "lasso_cv_mean_validation_mse": lasso_validation_mse,
                "train_r_squared": train_metrics["r_squared"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "test_r_squared": test_metrics["r_squared"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "target_unit": (
                    "rolling z-score"
                    if scaling_name != "none"
                    else "daily return (decimal)"
                ),
            }
        )
    return rows, train_prediction_frame, prediction_frame, top_predictor_stats


def save_actual_fitted_plot(
    target_name: str,
    predictions: pd.DataFrame,
    output_path: Path,
    sample_name: str = "Holdout",
    scaled: bool = False,
) -> None:
    """Save actual-versus-fitted scatter for each fitted model."""
    scale = 1 if scaled else 100
    actual_values = predictions["actual"] * scale
    figure, axis = plt.subplots(figsize=(8, 7))
    for model_name in ("ols", "ridge", "lasso"):
        axis.scatter(
            actual_values,
            predictions[f"{model_name}_fitted"] * scale,
            alpha=0.7,
            label=model_name.capitalize(),
        )
    lower = float(
        min(
            actual_values.min(),
            (predictions["ols_fitted"] * scale).min(),
            (predictions["ridge_fitted"] * scale).min(),
            (predictions["lasso_fitted"] * scale).min(),
        )
    )
    upper = float(
        max(
            actual_values.max(),
            (predictions["ols_fitted"] * scale).max(),
            (predictions["ridge_fitted"] * scale).max(),
            (predictions["lasso_fitted"] * scale).max(),
        )
    )
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="45° line")
    axis.set_title(
        f"{target_name}: {sample_name} Actual vs. Fitted "
        f"{'Rolling Z-Scores' if scaled else 'Daily Returns'}"
    )
    axis.set_xlabel("Actual rolling z-score" if scaled else "Actual daily return (%)")
    axis.set_ylabel("Fitted rolling z-score" if scaled else "Fitted daily return (%)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_actual_fitted_time_series_plot(
    target_name: str,
    model_set: str,
    model_name: str,
    predictions: pd.DataFrame,
    output_path: Path,
    sample_name: str = "Holdout",
    scaled: bool = False,
) -> None:
    """Plot actual and fitted values in date order for one regression."""
    fitted_column = f"{model_name.lower()}_fitted"
    scale = 1 if scaled else 100
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        predictions.index,
        predictions["actual"] * scale,
        label="Actual",
        linewidth=1.2,
    )
    axis.plot(
        predictions.index,
        predictions[fitted_column] * scale,
        label=f"{model_name} fitted",
        linewidth=1.2,
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"{target_name}: {sample_name} Actual vs. {model_name} Fitted Returns "
        f"({model_set.replace('_', ' ')}, "
        f"{'rolling z-score' if scaled else 'unscaled'})"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Rolling z-score" if scaled else "Daily return (%)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main(use_rolling_zscore: bool = False) -> None:
    ticker_series = load_ticker_series()
    scaling_name = (
        f"rolling_zscore_{ROLLING_ZSCORE_WINDOW}_observations"
        if use_rolling_zscore
        else "none"
    )
    output_dir = (
        OUTPUT_DIR / scaling_name if use_rolling_zscore else OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_output_file = output_dir / STATS_OUTPUT_FILE.name
    predictions_output_file = output_dir / PREDICTIONS_OUTPUT_FILE.name
    top_predictor_output_file = output_dir / TOP_TSTAT_OUTPUT_FILE.name
    all_stats: list[dict[str, object]] = []
    prediction_frames = []
    all_top_predictor_stats: list[dict[str, object]] = []

    model_sets = (
        ("with_weekly_tff", False),
        ("daily_without_weekly_tff", True),
    )
    for model_set, exclude_weekly_tff in model_sets:
        features, target_series = build_model_data(
            ticker_series,
            exclude_weekly_tff=exclude_weekly_tff,
            scaling_window=(
                ROLLING_ZSCORE_WINDOW if use_rolling_zscore else None
            ),
        )
        for target_name, target in target_series.items():
            (
                stats,
                train_predictions,
                predictions,
                top_predictor_stats,
            ) = fit_target(features, target, target_name, scaling_name)
            for row in stats:
                row["model_set"] = model_set
            for row in top_predictor_stats:
                row["model_set"] = model_set
            all_stats.extend(stats)
            all_top_predictor_stats.extend(top_predictor_stats)
            predictions.insert(0, "target", target_name)
            predictions.insert(1, "model_set", model_set)
            predictions.insert(2, "scaling", scaling_name)
            prediction_frames.append(predictions.reset_index(names="date"))
            safe_target = (
                "5y_future"
                if target_name == FIVE_YEAR_TARGET
                else "5x2y_minus_3x10y_basket"
            )
            save_actual_fitted_plot(
                target_name,
                train_predictions,
                output_dir
                / f"{safe_target}_{model_set}_in_sample_actual_vs_fitted.png",
                sample_name="In-Sample",
                scaled=use_rolling_zscore,
            )
            save_actual_fitted_plot(
                target_name,
                predictions,
                output_dir
                / f"{safe_target}_{model_set}_holdout_actual_vs_fitted.png",
                scaled=use_rolling_zscore,
            )
            for model_name in ("OLS", "Ridge", "Lasso"):
                save_actual_fitted_time_series_plot(
                    target_name,
                    model_set,
                    model_name,
                    train_predictions,
                    output_dir
                    / (
                        f"{safe_target}_{model_set}_"
                        f"{model_name.lower()}_in_sample_actual_vs_predicted.png"
                    ),
                    sample_name="In-Sample",
                    scaled=use_rolling_zscore,
                )
                save_actual_fitted_time_series_plot(
                    target_name,
                    model_set,
                    model_name,
                    predictions,
                    output_dir
                    / (
                        f"{safe_target}_{model_set}_"
                        f"{model_name.lower()}_holdout_actual_vs_predicted.png"
                    ),
                    scaled=use_rolling_zscore,
                )

    stats_frame = pd.DataFrame(all_stats)
    stats_frame.to_csv(stats_output_file, index=False, float_format="%.10g")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        predictions_output_file, index=False, float_format="%.10g"
    )
    pd.DataFrame(all_top_predictor_stats).to_csv(
        top_predictor_output_file, index=False, float_format="%.10g"
    )

    for model_set, exclude_weekly_tff in model_sets:
        features, _ = build_model_data(
            ticker_series,
            exclude_weekly_tff=exclude_weekly_tff,
            scaling_window=(
                ROLLING_ZSCORE_WINDOW if use_rolling_zscore else None
            ),
        )
        print(
            f"{model_set}: {len(features)} complete predictor rows "
            f"({features.index.min().date()} to {features.index.max().date()}), "
            f"{len(features.columns)} predictors"
        )
    print(f"Scaling: {scaling_name}")
    print(f"Fit statistics: {stats_output_file}")
    print(f"Holdout fitted values: {predictions_output_file}")
    print(f"Top model predictor statistics: {top_predictor_output_file}")
    for row in all_stats:
        print(
            f"{row['model_set']} | {row['target']} {row['model']}: test R² "
            f"{row['test_r_squared']:.4f}, RMSE {row['test_rmse']:.6f}, "
            f"MAE {row['test_mae']:.6f}"
            + (
                f", alpha={row['ridge_alpha']:g}"
                if row["model"] == "Ridge"
                else (
                    f", alpha={row['lasso_alpha']:g}"
                    if row["model"] == "Lasso"
                    else ""
                )
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rolling-zscore",
        action="store_true",
        help=(
            "Scale each predictor and target using the mean and population "
            f"standard deviation of its previous {ROLLING_ZSCORE_WINDOW} "
            "observations."
        ),
    )
    main(use_rolling_zscore=parser.parse_args().rolling_zscore)
