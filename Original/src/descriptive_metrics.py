from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def safe_divide(numerator, denominator):
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.div(den.where(den.ne(0)))


def gini_coefficient(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        raise ValueError("Gini hanya menerima nilai nonnegatif.")
    total = x.sum()
    if total == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * total)) - (n + 1) / n)


def theil_t(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        raise ValueError("Theil T hanya menerima nilai nonnegatif.")
    total = x.sum()
    if total == 0:
        return 0.0
    positive = x[x > 0]
    shares = positive / total
    return float(np.sum(shares * np.log(shares * x.size)))


def herfindahl_hirschman(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    total = x.sum()
    if total == 0:
        return 0.0
    shares = x / total
    return float(np.sum(shares**2))


def concentration_share(values: Iterable[float], top_n: int) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    total = x.sum()
    if total == 0:
        return 0.0
    top_n = min(int(top_n), x.size)
    return float(np.sort(x)[-top_n:].sum() / total)


def lorenz_coordinates(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    if np.any(x < 0):
        raise ValueError("Kurva Lorenz hanya menerima nilai nonnegatif.")
    x = np.sort(x)
    total = x.sum()
    population = np.arange(0, x.size + 1) / x.size
    if total == 0:
        outcome = population.copy()
    else:
        outcome = np.insert(np.cumsum(x) / total, 0, 0.0)
    return population, outcome


def inequality_summary(values: Iterable[float], label: str) -> dict:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    return {
        "indicator": label,
        "n_units": int(x.size),
        "total": float(x.sum()) if x.size else np.nan,
        "mean": float(x.mean()) if x.size else np.nan,
        "median": float(np.median(x)) if x.size else np.nan,
        "gini": gini_coefficient(x),
        "theil_t": theil_t(x),
        "hhi": herfindahl_hirschman(x),
        "top_1_share": concentration_share(x, 1),
        "top_5_share": concentration_share(x, 5),
        "top_10_share": concentration_share(x, 10),
    }


def bootstrap_inequality(
    values: Iterable[float],
    *,
    n_boot: int = 1000,
    random_state: int = 42,
) -> dict:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "gini_lower": np.nan,
            "gini_upper": np.nan,
            "theil_lower": np.nan,
            "theil_upper": np.nan,
        }
    rng = np.random.default_rng(random_state)
    gini_boot = np.empty(n_boot)
    theil_boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(x, size=x.size, replace=True)
        gini_boot[i] = gini_coefficient(sample)
        theil_boot[i] = theil_t(sample)
    return {
        "gini_lower": float(np.quantile(gini_boot, 0.025)),
        "gini_upper": float(np.quantile(gini_boot, 0.975)),
        "theil_lower": float(np.quantile(theil_boot, 0.025)),
        "theil_upper": float(np.quantile(theil_boot, 0.975)),
    }


def theil_decomposition(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_col: str,
) -> dict:
    data = frame[[group_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[group_col, value_col])
    data = data.loc[data[value_col] >= 0]
    n_total = len(data)
    y_total = data[value_col].sum()
    total_theil = theil_t(data[value_col])
    if n_total == 0 or y_total == 0:
        return {
            "theil_total": total_theil,
            "theil_between": 0.0,
            "theil_within": 0.0,
            "decomposition_error": 0.0,
        }

    between = 0.0
    within = 0.0
    for _, group in data.groupby(group_col, dropna=False):
        n_g = len(group)
        y_g = group[value_col].sum()
        if y_g <= 0:
            continue
        income_share = y_g / y_total
        population_share = n_g / n_total
        between += income_share * np.log(income_share / population_share)
        within += income_share * theil_t(group[value_col])

    return {
        "theil_total": float(total_theil),
        "theil_between": float(between),
        "theil_within": float(within),
        "between_share": float(between / total_theil) if total_theil > 0 else 0.0,
        "within_share": float(within / total_theil) if total_theil > 0 else 0.0,
        "decomposition_error": float(total_theil - between - within),
    }


def robust_zscore(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float)
    result = np.full(x.shape, np.nan, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return result
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad == 0:
        return result
    result[finite] = 0.67448975 * (x[finite] - median) / mad
    return result


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size == 0:
        return q
    p_finite = p[finite_idx]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q_finite = np.empty_like(adjusted)
    q_finite[order] = adjusted
    q[finite_idx] = q_finite
    return q


def spearman_target_table(
    frame: pd.DataFrame,
    *,
    targets: Sequence[str],
    predictors: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for target in targets:
        for predictor in predictors:
            pair = frame[[target, predictor]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3 or pair[target].nunique() < 2 or pair[predictor].nunique() < 2:
                rho, p_value = np.nan, np.nan
            else:
                result = spearmanr(pair[target], pair[predictor])
                rho, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "n_complete": len(pair),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = np.nan
    for target, indices in result.groupby("target").groups.items():
        result.loc[indices, "q_value_bh"] = benjamini_hochberg(
            result.loc[indices, "p_value"].to_numpy()
        )
    return result


def descriptive_table(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        x = pd.to_numeric(frame[col], errors="coerce")
        valid = x.dropna()
        mean = valid.mean() if len(valid) else np.nan
        variance = valid.var(ddof=1) if len(valid) > 1 else np.nan
        rows.append(
            {
                "variable": col,
                "n": int(valid.size),
                "missing_n": int(x.isna().sum()),
                "missing_pct": float(x.isna().mean() * 100),
                "mean": float(mean) if pd.notna(mean) else np.nan,
                "std": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
                "median": float(valid.median()) if len(valid) else np.nan,
                "q1": float(valid.quantile(0.25)) if len(valid) else np.nan,
                "q3": float(valid.quantile(0.75)) if len(valid) else np.nan,
                "p95": float(valid.quantile(0.95)) if len(valid) else np.nan,
                "p99": float(valid.quantile(0.99)) if len(valid) else np.nan,
                "min": float(valid.min()) if len(valid) else np.nan,
                "max": float(valid.max()) if len(valid) else np.nan,
                "skewness": float(valid.skew()) if len(valid) > 2 else np.nan,
                "variance_to_mean": (
                    float(variance / mean)
                    if pd.notna(variance) and pd.notna(mean) and mean != 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
