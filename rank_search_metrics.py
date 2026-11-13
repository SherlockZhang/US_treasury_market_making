"""Rank search candidates by Sharpe, maximum drawdown, and Calmar."""

import numpy as np
import pandas as pd


METRICS = {
    "sharpe_ratio": "sharpe_rank",
    "max_drawdown_pct": "max_drawdown_rank",
    "calmar_ratio": "calmar_rank",
}


def rank_metrics(grid: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Rank each metric from best to worst within each group; retain tied winners."""
    ranked = grid.drop(columns=list(METRICS.values()), errors="ignore").copy()
    for metric, rank_name in METRICS.items():
        if not np.isfinite(ranked[metric]).all():
            raise ValueError(f"Cannot rank undefined or non-finite {metric}.")
        ranked[rank_name] = (
            ranked.groupby(groups, sort=False)[metric]
            .rank(ascending=False, method="min")
            .astype(int)
        )
    ranked["rank_sum"] = ranked[list(METRICS.values())].sum(axis=1)
    ranked["selected_best"] = ranked["rank_sum"].eq(
        ranked.groupby(groups, sort=False)["rank_sum"].transform("min")
    )
    return ranked
