"""Fixed two-component EWMA and KAMA signals for the six-model reports."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest_ewma_extreme_strategy as ewma
import backtest_kama_extreme_strategy as kama
import backtest_normal_cdf_signal as cdf


@dataclass(frozen=True)
class EWMAComponent:
    instrument: str
    n: int
    m: int

    def label(self) -> str:
        return f"EWMA({self.n},{self.m})@{self.instrument.split()[0].lower()}"


@dataclass(frozen=True)
class KAMAComponent:
    instrument: str
    slow: int
    fast: int
    er: int
    m: int

    def label(self) -> str:
        return f"KAMA({self.slow},{self.fast},{self.er},{self.m})@{self.instrument.split()[0].lower()}"


EWMA_MODELS = {
    "2Y future": (EWMAComponent("2Y future", 5, 40), EWMAComponent("2Y future", 10, 30)),
    "5Y future": (EWMAComponent("2Y future", 5, 40), EWMAComponent("10Y future", 20, 20)),
    "10Y future": (EWMAComponent("5Y future", 25, 15), EWMAComponent("10Y future", 15, 20)),
}
KAMA_MODELS = {
    "2Y future": (KAMAComponent("2Y future", 30, 5, 30, 3), KAMAComponent("5Y future", 30, 5, 30, 3)),
    "5Y future": (KAMAComponent("10Y future", 10, 5, 30, 3), KAMAComponent("2Y future", 30, 5, 30, 3)),
    "10Y future": (KAMAComponent("2Y future", 10, 3, 30, 3), KAMAComponent("5Y future", 5, 5, 30, 3)),
}


def model_label(instrument: str, model: str, position_size: int) -> str:
    models = {"EWMA": EWMA_MODELS, "KAMA": KAMA_MODELS}
    components = models[model][instrument]
    return f"{' + '.join(component.label() for component in components)}; position size={position_size}"


def positions_from_signals(
    signal: pd.Series, start: pd.Timestamp, position_size: int
) -> pd.Series:
    """Enter on the preceding day's signal; an opposite signal exits to flat."""
    first = int(signal.index.searchsorted(start))
    if first >= len(signal):
        raise ValueError(f"No prices are available on or after {start:%Y-%m-%d}.")
    positions = np.zeros(len(signal), dtype=float)
    current = 0.0
    for index in range(first, len(signal)):
        if index > 0:
            previous = float(signal.iloc[index - 1])
            if current == 0:
                current = previous
            elif current * previous < 0:
                current = 0.0
        positions[index] = current * position_size
    return pd.Series(positions, index=signal.index, name="position")


def run_backtest(
    prices: dict[str, pd.Series],
    instrument: str,
    model: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    history_start: pd.Timestamp,
    position_size: int = 3,
) -> pd.DataFrame:
    """Average two entry signals and trade the requested future on the next date."""
    if isinstance(position_size, bool) or not isinstance(position_size, (int, np.integer)) or position_size <= 0:
        raise ValueError("Position size must be a positive integer.")
    if model not in ("EWMA", "KAMA"):
        raise ValueError(f"Unknown trend model {model!r}.")
    components = (EWMA_MODELS if model == "EWMA" else KAMA_MODELS)[instrument]
    target = prices[instrument].loc[history_start:end].sort_index()
    if target.empty:
        raise ValueError(f"No prices are available for {instrument}.")
    if target.index.has_duplicates:
        raise ValueError(f"{instrument} price data contains duplicate dates.")
    if target.isna().any() or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError(f"{instrument} prices must be finite, non-missing, and positive.")
    signals = []
    for component in components:
        source = prices[component.instrument].loc[history_start:end].sort_index()
        if not source.index.equals(target.index):
            raise ValueError(
                f"{component.instrument} and {instrument} price dates must match exactly."
            )
        if isinstance(component, EWMAComponent):
            frame = ewma.run_backtest(source, component.n, component.m)
        else:
            frame = kama.run_backtest(source, component.fast, component.slow, component.er, component.m)
        signals.append(frame["long_entry_signal"].astype(int) - frame["short_entry_signal"].astype(int))

    results = pd.DataFrame({"price": target})
    results["component_1_signal"] = signals[0]
    results["component_2_signal"] = signals[1]
    results["entry_signal"] = (signals[0] + signals[1]) / 2
    results["position"] = positions_from_signals(results["entry_signal"], start, position_size)
    results["daily_return"] = target.pct_change(fill_method=None)
    cdf.apply_trading_cost(results, instrument)
    results["wealth"] = (1 + results["strategy_return"]).cumprod()
    if (results.loc[start:, "wealth"] <= 0).any():
        raise ValueError("The strategy's compounded wealth reached zero or less.")
    results["cumulative_pnl_pct"] = (results["wealth"] - 1) * 100
    results["running_peak"] = results["wealth"].cummax()
    results["drawdown"] = results["wealth"] / results["running_peak"] - 1
    return results.loc[start:].copy()


def parameter_rows(instrument: str, model: str, position_size: int) -> pd.DataFrame:
    """Describe fixed components without claiming they are grid winners."""
    models = {"EWMA": EWMA_MODELS, "KAMA": KAMA_MODELS}
    return pd.DataFrame([{
        "instrument": instrument,
        "model": model,
        "selection_source": "user specified",
        "component_1": models[model][instrument][0].label(),
        "component_2": models[model][instrument][1].label(),
        "position_size": position_size,
    }])
