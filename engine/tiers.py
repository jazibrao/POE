"""Tiering utilities.

We maintain two tier concepts:

1) Sales-tier (ASP-based)  : reflects realized selling capacity in the analysis period.
2) Add-tier   (RP-based)   : reflects current commercial relevance for additions.

Both use the same percentile cutoffs (Value/Mid/Premium/Ultra) but are computed
from different price signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


TIERS_ORDER = ["Value", "Mid", "Premium", "Ultra"]


@dataclass
class TierCutoffs:
    p_value: float
    p_mid: float
    p_premium: float
    cut_value: float
    cut_mid: float
    cut_premium: float


def _safe_quantile(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    x = x[x > 0]
    if x.empty:
        return float("nan")
    return float(x.quantile(q))


def compute_cutoffs(price_series: pd.Series, p_value: float, p_mid: float, p_premium: float) -> TierCutoffs:
    """Compute tier cutoffs from a price series using percentiles."""
    c_value = _safe_quantile(price_series, p_value)
    c_mid = _safe_quantile(price_series, p_mid)
    c_prem = _safe_quantile(price_series, p_premium)
    return TierCutoffs(p_value=p_value, p_mid=p_mid, p_premium=p_premium, cut_value=c_value, cut_mid=c_mid, cut_premium=c_prem)


def assign_tier_from_price(price: pd.Series, cutoffs: TierCutoffs) -> pd.Series:
    """Assign Value/Mid/Premium/Ultra based on numeric price and cutoffs."""
    p = pd.to_numeric(price, errors="coerce")
    out = pd.Series(index=price.index, dtype="object")
    out[:] = "UNASSIGNED"

    # If cutoffs are nan, everything stays UNASSIGNED
    if not np.isfinite(cutoffs.cut_value) or not np.isfinite(cutoffs.cut_mid) or not np.isfinite(cutoffs.cut_premium):
        return out

    out[p <= cutoffs.cut_value] = "Value"
    out[(p > cutoffs.cut_value) & (p <= cutoffs.cut_mid)] = "Mid"
    out[(p > cutoffs.cut_mid) & (p <= cutoffs.cut_premium)] = "Premium"
    out[p > cutoffs.cut_premium] = "Ultra"
    return out


def normalize_tier_labels(x: pd.Series) -> pd.Series:
    """Normalize common tier label variants to Value/Mid/Premium/Ultra."""
    m = {
        "value": "Value",
        "val": "Value",
        "mid": "Mid",
        "middle": "Mid",
        "premium": "Premium",
        "prem": "Premium",
        "ultra": "Ultra",
        "ulta": "Ultra",
    }
    return x.astype(str).str.strip().str.lower().map(lambda z: m.get(z, z.title()))
