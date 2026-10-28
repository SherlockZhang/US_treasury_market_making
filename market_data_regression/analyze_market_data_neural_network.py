"""Fit a small neural network to 5Y future daily returns using market data.

Predictors are the daily changes of all market-data series except the 2Y, 5Y
and 10Y futures and CTA US Equity (already excluded by build_model_data). The
sample is split chronologically: the first 80% is used for training (its last
20% is held back for early stopping) and the remaining 20% is the holdout.
Hyperparameters are chosen by expanding-window cross-validation inside the
training period only, so the holdout is used once, for the final evaluation.
"""

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_market_data_regression import (
    FIVE_YEAR_TARGET,
    TRAIN_FRACTION,
    build_model_data,
    calculate_metrics,
    fit_linear_model,
    load_ticker_series,
    predict,
    save_actual_fitted_time_series_plot,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "neural_network"
ENSEMBLE_SIZE = 10
BATCH_SIZE = 64
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 30
VALIDATION_FRACTION = 0.2
BASE_SEED = 42
CV_SPLITS = 4
SEARCH_ENSEMBLE_SIZE = 2
HIDDEN_LAYER_OPTIONS = ((4,), (8,), (16,), (32,), (16, 8), (32, 16))
WEIGHT_DECAY_OPTIONS = (1e-3, 1e-2, 1e-1, 1.0)
LEARNING_RATE_OPTIONS = (1e-3, 3e-3)
DROPOUT_OPTIONS = (0.0, 0.2)


@dataclass(frozen=True)
class Hyperparameters:
    hidden_layers: tuple[int, ...]
    weight_decay: float
    learning_rate: float
    dropout: float


DEFAULT_HYPERPARAMETERS = Hyperparameters((32, 16), 1e-3, 1e-3, 0.0)


class MLPRegressor:
    """Fully connected ReLU network trained with Adam and early stopping."""

    def __init__(self, input_size: int, seed: int, hp: Hyperparameters) -> None:
        self.hp = hp
        self.rng = np.random.default_rng(seed)
        sizes = (input_size, *hp.hidden_layers, 1)
        self.weights = [
            self.rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
        ]
        self.biases = [np.zeros(fan_out) for fan_out in sizes[1:]]

    def forward(
        self, features: np.ndarray, training: bool = False
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray | None]]:
        activations = [features]
        masks: list[np.ndarray | None] = [None]
        keep = 1 - self.hp.dropout
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            output = activations[-1] @ weight + bias
            mask = None
            if layer < len(self.weights) - 1:
                output = np.maximum(output, 0)
                if training and self.hp.dropout > 0:
                    mask = (self.rng.random(output.shape) < keep) / keep
                    output = output * mask
            activations.append(output)
            masks.append(mask)
        return activations[-1].ravel(), activations, masks

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.forward(features)[0]

    def _gradients(
        self, features: np.ndarray, target: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        prediction, activations, masks = self.forward(features, training=True)
        delta = (2 * (prediction - target) / len(target))[:, None]
        weight_grads: list[np.ndarray] = []
        bias_grads: list[np.ndarray] = []
        for layer in reversed(range(len(self.weights))):
            weight_grads.append(
                activations[layer].T @ delta
                + self.hp.weight_decay * self.weights[layer]
            )
            bias_grads.append(delta.sum(axis=0))
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * (activations[layer] > 0)
                if masks[layer] is not None:
                    delta = delta * masks[layer]
        return weight_grads[::-1], bias_grads[::-1]

    def fit(
        self,
        train_features: np.ndarray,
        train_target: np.ndarray,
        validation_features: np.ndarray,
        validation_target: np.ndarray,
    ) -> int:
        """Train with Adam, restoring the best validation weights."""
        parameters = self.weights + self.biases
        first_moment = [np.zeros_like(p) for p in parameters]
        second_moment = [np.zeros_like(p) for p in parameters]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        step = 0
        best_loss = np.inf
        best_epoch = 0
        best_state = None

        for epoch in range(MAX_EPOCHS):
            order = self.rng.permutation(len(train_target))
            for start in range(0, len(order), BATCH_SIZE):
                batch = order[start : start + BATCH_SIZE]
                weight_grads, bias_grads = self._gradients(
                    train_features[batch], train_target[batch]
                )
                step += 1
                for index, gradient in enumerate(weight_grads + bias_grads):
                    first_moment[index] = (
                        beta1 * first_moment[index] + (1 - beta1) * gradient
                    )
                    second_moment[index] = (
                        beta2 * second_moment[index] + (1 - beta2) * gradient**2
                    )
                    corrected_first = first_moment[index] / (1 - beta1**step)
                    corrected_second = second_moment[index] / (1 - beta2**step)
                    parameters[index] -= (
                        self.hp.learning_rate
                        * corrected_first
                        / (np.sqrt(corrected_second) + epsilon)
                    )
            validation_loss = float(
                np.mean((self.predict(validation_features) - validation_target) ** 2)
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = [p.copy() for p in parameters]
            elif epoch + 1 - best_epoch >= EARLY_STOPPING_PATIENCE:
                break

        for parameter, saved in zip(parameters, best_state):
            parameter[...] = saved
        return best_epoch


def fit_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_other: list[np.ndarray],
    hp: Hyperparameters,
    ensemble_size: int = ENSEMBLE_SIZE,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Train the network ensemble and return averaged predictions."""
    validation_start = int(len(y_train) * (1 - VALIDATION_FRACTION))
    X_fit, y_fit = X_train[:validation_start], y_train[:validation_start]
    X_val, y_val = X_train[validation_start:], y_train[validation_start:]

    feature_mean = X_fit.mean(axis=0)
    feature_scale = X_fit.std(axis=0, ddof=0)
    if not np.isfinite(feature_scale).all() or (feature_scale == 0).any():
        raise ValueError("Predictors must have finite, non-zero training variance.")
    target_mean, target_scale = y_fit.mean(), y_fit.std(ddof=0)

    def scale_features(values: np.ndarray) -> np.ndarray:
        return (values - feature_mean) / feature_scale

    scaled_fit, scaled_val = scale_features(X_fit), scale_features(X_val)
    scaled_target_fit = (y_fit - target_mean) / target_scale
    scaled_target_val = (y_val - target_mean) / target_scale

    train_predictions = np.zeros(len(y_train))
    other_predictions = [np.zeros(len(values)) for values in X_other]
    best_epochs = []
    for member in range(ensemble_size):
        model = MLPRegressor(X_train.shape[1], BASE_SEED + member, hp)
        best_epochs.append(
            model.fit(scaled_fit, scaled_target_fit, scaled_val, scaled_target_val)
        )
        train_predictions += model.predict(scale_features(X_train))
        for predictions, values in zip(other_predictions, X_other):
            predictions += model.predict(scale_features(values))

    def unscale(values: np.ndarray) -> np.ndarray:
        return values / ensemble_size * target_scale + target_mean

    return (
        unscale(train_predictions),
        [unscale(predictions) for predictions in other_predictions],
        best_epochs,
    )


def search_hyperparameters(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[Hyperparameters, pd.DataFrame]:
    """Pick hyperparameters by expanding-window CV on the training sample."""
    block = len(y_train) // (CV_SPLITS + 1)
    folds = [
        (np.arange((i + 1) * block), np.arange((i + 1) * block, (i + 2) * block))
        for i in range(CV_SPLITS)
    ]
    results = []
    grid = itertools.product(
        HIDDEN_LAYER_OPTIONS,
        WEIGHT_DECAY_OPTIONS,
        LEARNING_RATE_OPTIONS,
        DROPOUT_OPTIONS,
    )
    for hidden_layers, weight_decay, learning_rate, dropout in grid:
        hp = Hyperparameters(hidden_layers, weight_decay, learning_rate, dropout)
        squared_error = baseline_squared_error = 0.0
        count = 0
        for fit_idx, val_idx in folds:
            _, (fold_pred,), _ = fit_ensemble(
                X_train[fit_idx],
                y_train[fit_idx],
                [X_train[val_idx]],
                hp,
                SEARCH_ENSEMBLE_SIZE,
            )
            squared_error += float(np.sum((y_train[val_idx] - fold_pred) ** 2))
            baseline_squared_error += float(
                np.sum((y_train[val_idx] - y_train[fit_idx].mean()) ** 2)
            )
            count += len(val_idx)
        results.append(
            {
                "hidden_layers": "-".join(map(str, hidden_layers)),
                "weight_decay": weight_decay,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "cv_mse": squared_error / count,
                "cv_r_squared": 1 - squared_error / baseline_squared_error,
                "hp": hp,
            }
        )
    frame = pd.DataFrame(results).sort_values("cv_mse").reset_index(drop=True)
    return frame.loc[0, "hp"], frame.drop(columns="hp")


def save_scatter_plot(
    predictions: pd.DataFrame, title: str, output_path: Path
) -> None:
    actual = predictions["actual"] * 100
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(actual, predictions["nn_fitted"] * 100, alpha=0.7, label="Neural network")
    lower = float(min(actual.min(), (predictions["nn_fitted"] * 100).min()))
    upper = float(max(actual.max(), (predictions["nn_fitted"] * 100).max()))
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1, label="45° line")
    axis.set_title(title)
    axis.set_xlabel("Actual daily return (%)")
    axis.set_ylabel("Fitted daily return (%)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main(include_weekly_tff: bool) -> None:
    ticker_series = load_ticker_series()
    features, targets = build_model_data(
        ticker_series, exclude_weekly_tff=not include_weekly_tff
    )
    target = targets[FIVE_YEAR_TARGET]
    aligned = pd.concat([features, target], axis=1, join="inner").dropna()
    X = aligned.loc[:, features.columns].to_numpy(dtype=float)
    y = aligned[FIVE_YEAR_TARGET].to_numpy(dtype=float)
    split_index = int(len(aligned) * TRAIN_FRACTION)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    train_index, test_index = aligned.index[:split_index], aligned.index[split_index:]

    model_set = "with_weekly_tff" if include_weekly_tff else "daily_without_weekly_tff"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_hp, search_results = search_hyperparameters(X_train, y_train)
    search_file = OUTPUT_DIR / f"nn_5y_future_hyperparameter_search_{model_set}.csv"
    search_results.to_csv(search_file, index=False, float_format="%.10g")

    train_fitted, (test_fitted,), best_epochs = fit_ensemble(
        X_train, y_train, [X_test], best_hp
    )
    default_train, (default_test,), default_epochs = fit_ensemble(
        X_train, y_train, [X_test], DEFAULT_HYPERPARAMETERS
    )
    ols_coefficients, ols_intercept, _ = fit_linear_model(X_train, y_train)
    ols_train = predict(X_train, ols_coefficients, ols_intercept)
    ols_test = predict(X_test, ols_coefficients, ols_intercept)

    rows = []
    for model_name, hp, epochs, train_pred, test_pred in (
        ("Neural network (tuned)", best_hp, best_epochs, train_fitted, test_fitted),
        (
            "Neural network (untuned default)",
            DEFAULT_HYPERPARAMETERS,
            default_epochs,
            default_train,
            default_test,
        ),
        ("OLS baseline", None, None, ols_train, ols_test),
    ):
        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)
        rows.append(
            {
                "target": FIVE_YEAR_TARGET,
                "model": model_name,
                "model_set": model_set,
                "predictor_count": X.shape[1],
                "train_observations": len(y_train),
                "test_observations": len(y_test),
                "train_start_date": train_index.min().date(),
                "train_end_date": train_index.max().date(),
                "test_start_date": test_index.min().date(),
                "test_end_date": test_index.max().date(),
                "train_r_squared": train_metrics["r_squared"],
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "test_r_squared": test_metrics["r_squared"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "hidden_layers": (
                    "-".join(map(str, hp.hidden_layers)) if hp else ""
                ),
                "weight_decay": hp.weight_decay if hp else np.nan,
                "learning_rate": hp.learning_rate if hp else np.nan,
                "dropout": hp.dropout if hp else np.nan,
                "ensemble_size": ENSEMBLE_SIZE if hp else np.nan,
                "mean_best_epoch": float(np.mean(epochs)) if hp else np.nan,
            }
        )

    stats_file = OUTPUT_DIR / f"nn_5y_future_fit_stats_{model_set}.csv"
    pd.DataFrame(rows).to_csv(stats_file, index=False, float_format="%.10g")

    train_frame = pd.DataFrame(
        {"actual": y_train, "nn_fitted": train_fitted, "ols_fitted": ols_train},
        index=train_index,
    )
    test_frame = pd.DataFrame(
        {"actual": y_test, "nn_fitted": test_fitted, "ols_fitted": ols_test},
        index=test_index,
    )
    pd.concat(
        [train_frame.assign(sample="in_sample"), test_frame.assign(sample="holdout")]
    ).reset_index(names="date").to_csv(
        OUTPUT_DIR / f"nn_5y_future_fitted_values_{model_set}.csv",
        index=False,
        float_format="%.10g",
    )

    for sample_name, label, frame in (
        ("in_sample", "In-Sample", train_frame),
        ("holdout", "Holdout", test_frame),
    ):
        save_scatter_plot(
            frame,
            f"{FIVE_YEAR_TARGET}: {label} Actual vs. Neural Network Fitted Returns",
            OUTPUT_DIR / f"5y_future_{model_set}_nn_{sample_name}_actual_vs_fitted.png",
        )
        save_actual_fitted_time_series_plot(
            FIVE_YEAR_TARGET,
            model_set,
            "NN",
            frame,
            OUTPUT_DIR / f"5y_future_{model_set}_nn_{sample_name}_actual_vs_predicted.png",
            sample_name=label,
        )

    print(f"{model_set}: {len(aligned)} observations, {X.shape[1]} predictors")
    print(f"Predictors: {', '.join(features.columns)}")
    print(f"Fit statistics: {stats_file}")
    print(f"Hyperparameter search: {search_file}")
    print(f"Selected hyperparameters: {best_hp}")
    print(search_results.head(5).to_string(index=False))
    for row in rows:
        print(
            f"{row['model']}: train R² {row['train_r_squared']:.4f}, "
            f"test R² {row['test_r_squared']:.4f}, "
            f"test RMSE {row['test_rmse']:.6f}, test MAE {row['test_mae']:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-weekly-tff",
        action="store_true",
        help="Include the weekly TFF predictors (far fewer complete rows).",
    )
    main(include_weekly_tff=parser.parse_args().include_weekly_tff)
