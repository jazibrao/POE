"""Output formatting helpers.

Business users expect *counts* (SKUs, quantities, stock) as integers.
Excel imports frequently convert integer fields to floats (e.g., 12.0),
which makes outputs look confusing and invites unnecessary objections.

This module normalizes output dataframes before export.
"""

from __future__ import annotations

import pandas as pd


_EXCLUDE_INT_COLS = {
    # Scores / rates
    "score",
    "score_addtier",
    "gmroi",
    "ros",
    "str",
    "asp",
    "assigned_tier_share",
    "gmroi_pctl",
    "ros_pctl",
    "gmroi_pctl_add",
    "ros_pctl_add",
    # weights and misc
    "q_score",
    "qscore",
}


def _is_int_like_col(col: str) -> bool:
    c = str(col).strip().lower()
    if c in _EXCLUDE_INT_COLS:
        return False
    # keep ratio/probability style columns as float
    if any(k in c for k in ["mix", "share", "pctl", "alpha", "fraction"]):
        return False
    # keep intermediate raw calculations as float (e.g., Target_SKUs_Tier_raw)
    if c.endswith("_raw"):
        return False
    # explicit quantity/count fields
    if c.endswith("qty"):
        return True
    if "stock" in c and "cost" not in c and "value" not in c:
        return True
    if "sku" in c and "score" not in c:
        return True
    if c in {"gap", "eligibility", "tier_rank_add", "floor", "add1"}:
        return True
    return False


def storeid_first(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure StoreID is the first column when present."""
    if df is None or df.empty:
        return df
    if "StoreID" not in df.columns:
        return df
    cols = list(df.columns)
    if cols and cols[0] == "StoreID":
        return df
    cols = ["StoreID"] + [c for c in cols if c != "StoreID"]
    return df[cols]


def integers_everywhere(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce integer-like columns to pandas nullable Int64."""
    if df is None or df.empty:
        return df

    out = df.copy()
    for col in out.columns:
        if not _is_int_like_col(col):
            continue

        s = pd.to_numeric(out[col], errors="coerce")
        # keep NaN as <NA>
        s = s.round()
        try:
            out[col] = s.astype("Int64")
        except Exception:
            # If conversion fails, leave as-is to avoid crashing exports.
            pass

    return out


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all output formatting rules."""
    if df is None:
        return df
    df2 = integers_everywhere(df)
    df2 = storeid_first(df2)
    return df2
