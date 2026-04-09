"""Quality Control (QC) checks for Portfolio Optimization Engine inputs.

This module is intentionally **non-blocking**: it returns findings as structured
tables so the Streamlit app can surface issues early, without stopping the
workflow.

QC categories
-------------
Errors
  Likely to break the engine or produce meaningless outputs.
Warnings
  Won't necessarily break execution, but can distort results.
Info
  Useful diagnostics/coverage summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------- helpers ------------------------------------


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find_col_fuzzy(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    # exact
    for c in candidates:
        if c in df.columns:
            return c
    norm_map = {_norm(c): c for c in df.columns}
    for c in candidates:
        k = _norm(c)
        if k in norm_map:
            return norm_map[k]
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


@dataclass
class QCResult:
    errors: pd.DataFrame
    warnings: pd.DataFrame
    info: pd.DataFrame
    suggestions: pd.DataFrame
    snapshots: Dict[str, pd.DataFrame]


def _issue_df(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Severity", "Area", "Issue", "Count", "Details"])
    df = pd.DataFrame(rows)
    # stable column order
    for c in ["Severity", "Area", "Issue", "Count", "Details"]:
        if c not in df.columns:
            df[c] = ""
    return df[["Severity", "Area", "Issue", "Count", "Details"]]


# ----------------------------- main --------------------------------------


def run_qc_checks(
    stores: pd.DataFrame,
    tiers: pd.DataFrame,
    articles: pd.DataFrame,
    cfg: object | None = None,
) -> QCResult:
    """Run quality checks on the three required inputs.

    Returns
    -------
    QCResult
        errors, warnings, info: summary tables
        snapshots: key drill-down tables (duplicates, orphans, etc.)
    """

    rows_err: List[Dict] = []
    rows_warn: List[Dict] = []
    rows_info: List[Dict] = []
    rows_suggest: List[Dict] = []
    snaps: Dict[str, pd.DataFrame] = {}

    # ---------------- required columns (hard schema) -----------------
    req_stores = ["StoreID", "Average of SKU_DC", "RFT", "SPF", "Q_SCORE", "GMROI"]
    req_tiers = ["StoreID", "PriceTier", "Average of SKU_DC", "GMROI"]
    # PriceTier can be derived from ASP when not provided, so it's not a hard blocker.
    req_articles = [
        "StoreID",
        "ItemColorName",
        "Sales_Value",
        "Sales_Qty",
        "Gross_Margin",
        "ClosingStockQty",
    ]

    def _missing(df: pd.DataFrame, req: List[str]) -> List[str]:
        return [c for c in req if c not in df.columns]

    miss_s = _missing(stores, req_stores)
    miss_t = _missing(tiers, req_tiers)
    miss_a = _missing(articles, req_articles)

    if miss_s:
        rows_suggest.append({"Severity": "SUGGEST", "Area": "StoresKPIs", "Issue": "Add missing columns", "Count": len(miss_s), "Details": ", ".join(miss_s)})
        rows_err.append({
            "Severity": "ERROR",
            "Area": "StoresKPIs",
            "Issue": "Missing required columns",
            "Count": len(miss_s),
            "Details": ", ".join(miss_s),
        })
    if miss_t:
        rows_suggest.append({"Severity": "SUGGEST", "Area": "TierKPIs", "Issue": "Add missing columns", "Count": len(miss_t), "Details": ", ".join(miss_t)})
        rows_err.append({
            "Severity": "ERROR",
            "Area": "TierKPIs",
            "Issue": "Missing required columns",
            "Count": len(miss_t),
            "Details": ", ".join(miss_t),
        })
    if miss_a:
        rows_suggest.append({"Severity": "SUGGEST", "Area": "Articles", "Issue": "Add missing columns", "Count": len(miss_a), "Details": ", ".join(miss_a)})
        rows_err.append({
            "Severity": "ERROR",
            "Area": "Articles",
            "Issue": "Missing required columns",
            "Count": len(miss_a),
            "Details": ", ".join(miss_a),
        })

    # PriceTier is optional now (can be derived from ASP), but we warn the user.
    if "PriceTier" not in articles.columns:
        asp_col = _find_col_fuzzy(articles, ["ASP", "AvgSellingPrice", "Average Selling Price", "Avg_Selling_Price"])
        if asp_col is None:
            rows_warn.append({
                "Severity": "WARNING",
                "Area": "Articles",
                "Issue": "PriceTier missing and ASP missing",
                "Count": 1,
                "Details": "Provide either PriceTier (Value/Mid/Premium/Ultra) or ASP so the engine can derive tiers.",
            })
        else:
            rows_info.append({
                "Severity": "INFO",
                "Area": "Articles",
                "Issue": "PriceTier missing, will derive from ASP",
                "Count": 1,
                "Details": f"Using ASP column: {asp_col}",
            })

    # UI intelligence checks
    if _find_col_fuzzy(articles, ["RetailPrice", "MRP", "RSP", "ListPrice"]) is None:
        rows_warn.append({
            "Severity": "WARNING",
            "Area": "Articles",
            "Issue": "Retail price signal missing",
            "Count": 1,
            "Details": "Dynamic Add-tier (RetailPrice-based) will fall back to sales tier if RetailPrice/MRP not found.",
        })

    # Even if schema has errors, we continue and compute what we can.

    # ------------------ config-driven checks (if cfg provided) ------------------
    try:
        eff_top_pct = getattr(cfg, "efficient_top_pct", None)
        if eff_top_pct is not None and ("GMROI" in stores.columns):
            gm = _safe_numeric(stores["GMROI"]).fillna(0)
            thr = np.percentile(gm.values, 100 * (1 - float(eff_top_pct)))
            n_eff = int((gm >= thr).sum())
            if n_eff < 5:
                rows_warn.append({
                    "Severity": "WARNING",
                    "Area": "StoresKPIs",
                    "Issue": "Too few efficient stores for stable CV",
                    "Count": n_eff,
                    "Details": f"efficient_top_pct={eff_top_pct} gives only {n_eff} efficient stores. Consider lowering top% or disable auto CV selection.",
                })
                rows_suggest.append({
                    "Severity": "SUGGEST",
                    "Area": "Configuration",
                    "Issue": "Adjust efficient_top_pct",
                    "Count": n_eff,
                    "Details": "If you are testing on a small store subset, reduce efficient_top_pct or set fixed gate weight.",
                })
    except Exception:
        pass

    # ---------------- key identifier nulls ---------------------------
    def _null_rate(df: pd.DataFrame, col: str) -> Tuple[int, float]:
        if col not in df.columns:
            return 0, 0.0
        n = int(df[col].isna().sum())
        r = (n / max(len(df), 1))
        return n, r

    for area, df, cols in [
        ("StoresKPIs", stores, ["StoreID"]),
        ("TierKPIs", tiers, ["StoreID", "PriceTier"]),
        ("Articles", articles, ["StoreID", "ItemColorName", "PriceTier"]),
    ]:
        for c in cols:
            n, r = _null_rate(df, c)
            if n > 0:
                rows_err.append({
                    "Severity": "ERROR",
                    "Area": area,
                    "Issue": f"Nulls in key identifier: {c}",
                    "Count": n,
                    "Details": f"{_pct(r)} of rows",
                })

    # ---------------- duplicates in primary keys ---------------------
    # Stores: StoreID should be unique
    if "StoreID" in stores.columns:
        dup = stores[stores.duplicated(["StoreID"], keep=False)].copy()
        if not dup.empty:
            rows_warn.append({
                "Severity": "WARNING",
                "Area": "StoresKPIs",
                "Issue": "Duplicate store identifiers (StoreID)",
                "Count": len(dup),
                "Details": "Engine expects unique stores; duplicates may distort model.",
            })
            snaps["Duplicate_Stores"] = dup

    # Tiers: StoreID×PriceTier should be unique
    if all(c in tiers.columns for c in ["StoreID", "PriceTier"]):
        dup = tiers[tiers.duplicated(["StoreID", "PriceTier"], keep=False)].copy()
        if not dup.empty:
            rows_warn.append({
                "Severity": "WARNING",
                "Area": "TierKPIs",
                "Issue": "Duplicate StoreID×PriceTier rows",
                "Count": len(dup),
                "Details": "Tier targets may be miscomputed if duplicates exist.",
            })
            snaps["Duplicate_TierKPIs"] = dup

    # Articles: duplicates can be valid (time rows), but warn if no time columns and duplicates exist
    time_cols = [
        _find_col_fuzzy(articles, ["MonthYear", "Month_Year", "YearMonth", "YM"]),
        _find_col_fuzzy(articles, ["WeekStart", "WeekStartDate", "Week", "WeekNo", "WeekNumber"]),
    ]
    time_cols = [c for c in time_cols if c]
    base_key = [c for c in ["StoreID", "ItemColorName", "PriceTier"] if c in articles.columns]
    if base_key and len(base_key) == 3:
        subset = base_key + time_cols
        dup = articles[articles.duplicated(subset, keep=False)].copy()
        if not dup.empty:
            # duplicates on key+time is still odd, but could happen. treat as warning
            rows_warn.append({
                "Severity": "WARNING",
                "Area": "Articles",
                "Issue": "Duplicate rows in Articles on (StoreID, SKU, Tier, Time)",
                "Count": len(dup),
                "Details": "May double-count Sales/Stock unless intended.",
            })
            snaps["Duplicate_Articles_Key"] = dup.head(500)

        if not time_cols:
            # if time granularity not present, duplicates on Store×SKU×Tier should be rare
            dup2 = articles[articles.duplicated(base_key, keep=False)].copy()
            if not dup2.empty:
                rows_warn.append({
                    "Severity": "WARNING",
                    "Area": "Articles",
                    "Issue": "Duplicate StoreID×SKU×Tier rows without time fields",
                    "Count": len(dup2),
                    "Details": "Consider aggregating Articles to one row per Store×SKU×Tier.",
                })
                snaps["Duplicate_Articles_NoTime"] = dup2.head(500)

    # ---------------- referential integrity --------------------------
    stores_ids = set(stores["StoreID"].astype(str)) if "StoreID" in stores.columns else set()
    tiers_ids = set(tiers["StoreID"].astype(str)) if "StoreID" in tiers.columns else set()
    art_ids = set(articles["StoreID"].astype(str)) if "StoreID" in articles.columns else set()

    if stores_ids and tiers_ids:
        orphan_tiers = sorted(list(tiers_ids - stores_ids))
        if orphan_tiers:
            rows_err.append({
                "Severity": "ERROR",
                "Area": "TierKPIs",
                "Issue": "StoreIDs present in TierKPIs but missing in StoresKPIs",
                "Count": len(orphan_tiers),
                "Details": ", ".join(orphan_tiers[:25]) + (" ..." if len(orphan_tiers) > 25 else ""),
            })
            snaps["Orphan_StoreIDs_TierKPIs"] = tiers[tiers["StoreID"].astype(str).isin(orphan_tiers)].head(500)

    if stores_ids and art_ids:
        orphan_art = sorted(list(art_ids - stores_ids))
        if orphan_art:
            rows_err.append({
                "Severity": "ERROR",
                "Area": "Articles",
                "Issue": "StoreIDs present in Articles but missing in StoresKPIs",
                "Count": len(orphan_art),
                "Details": ", ".join(orphan_art[:25]) + (" ..." if len(orphan_art) > 25 else ""),
            })
            snaps["Orphan_StoreIDs_Articles"] = articles[articles["StoreID"].astype(str).isin(orphan_art)].head(500)

    # PriceTier alignment
    tier_vals = set(tiers["PriceTier"].astype(str)) if "PriceTier" in tiers.columns else set()
    art_tier_vals = set(articles["PriceTier"].astype(str)) if "PriceTier" in articles.columns else set()
    if tier_vals and art_tier_vals:
        unknown = sorted(list(art_tier_vals - tier_vals))
        if unknown:
            rows_warn.append({
                "Severity": "WARNING",
                "Area": "Articles",
                "Issue": "PriceTier values in Articles not found in TierKPIs",
                "Count": len(unknown),
                "Details": ", ".join(unknown[:25]) + (" ..." if len(unknown) > 25 else ""),
            })

    # ---------------- numeric sanity checks --------------------------
    def _neg_check(area: str, df: pd.DataFrame, cols: List[str]):
        for c in cols:
            if c not in df.columns:
                continue
            s = _safe_numeric(df[c])
            bad = df[s < 0]
            if not bad.empty:
                rows_warn.append({
                    "Severity": "WARNING",
                    "Area": area,
                    "Issue": f"Negative values in {c}",
                    "Count": len(bad),
                    "Details": "Check refunds/returns or data errors.",
                })
                snaps[f"Negative_{area}_{c}"] = bad.head(500)

    _neg_check("Articles", articles, ["Sales_Qty", "Sales_Value", "Gross_Margin", "ClosingStockQty"])
    _neg_check("StoresKPIs", stores, ["Average of SKU_DC", "RFT", "SPF", "Q_SCORE", "GMROI"])
    _neg_check("TierKPIs", tiers, ["Average of SKU_DC", "GMROI"])

    # GMROI denominator checks (Avg Inventory Cost)
    avg_inv_cost_col = _find_col_fuzzy(
        articles,
        [
            "Avg Inventory Cost",
            "Avg_Inventory_Cost",
            "AvgInventoryCost",
            "AvgInvCost",
            "AverageInventoryCost",
        ],
    )
    if avg_inv_cost_col is None:
        rows_warn.append({
            "Severity": "WARNING",
            "Area": "Articles",
            "Issue": "Avg Inventory Cost column not found",
            "Count": 0,
            "Details": "GMROI will fall back to proxy if not present. Prefer providing Avg Inventory Cost.",
        })
    else:
        denom = _safe_numeric(articles[avg_inv_cost_col]).fillna(0)
        bad = articles[denom <= 0]
        if not bad.empty:
            rows_err.append({
                "Severity": "ERROR",
                "Area": "Articles",
                "Issue": "Avg Inventory Cost <= 0 (GMROI invalid)",
                "Count": len(bad),
                "Details": f"Column: {avg_inv_cost_col}",
            })
            snaps["AvgInvCost_Invalid"] = bad.head(500)

    # Sales consistency checks
    if all(c in articles.columns for c in ["Sales_Qty", "Sales_Value"]):
        q = _safe_numeric(articles["Sales_Qty"]).fillna(0)
        v = _safe_numeric(articles["Sales_Value"]).fillna(0)
        inconsistent = articles[((q == 0) & (v > 0)) | ((q > 0) & (v == 0))]
        if not inconsistent.empty:
            rows_warn.append({
                "Severity": "WARNING",
                "Area": "Articles",
                "Issue": "Sales_Qty and Sales_Value inconsistent",
                "Count": len(inconsistent),
                "Details": "Rows with Qty=0 but Value>0 OR Qty>0 but Value=0.",
            })
            snaps["Sales_Inconsistent"] = inconsistent.head(500)

    # ---------------- coverage diagnostics ----------------------------
    # Detect time granularity
    monthyear_col = _find_col_fuzzy(articles, ["MonthYear", "Month_Year", "YearMonth", "YM"])
    week_col = _find_col_fuzzy(articles, ["WeekStart", "WeekStartDate", "Week", "WeekNo", "WeekNumber"])

    if monthyear_col is not None:
        rows_info.append({
            "Severity": "INFO",
            "Area": "Articles",
            "Issue": "Detected monthly dataset",
            "Count": articles[monthyear_col].nunique(dropna=True),
            "Details": f"Column: {monthyear_col}",
        })
        # coverage by store
        cov = (
            articles.groupby("StoreID")[monthyear_col]
            .nunique(dropna=True)
            .rename("Months_Active")
            .reset_index()
            .sort_values("Months_Active")
        )
        snaps["Monthly_Coverage_By_Store"] = cov
    elif week_col is not None:
        rows_info.append({
            "Severity": "INFO",
            "Area": "Articles",
            "Issue": "Detected weekly dataset",
            "Count": articles[week_col].nunique(dropna=True),
            "Details": f"Column: {week_col}",
        })
        cov = (
            articles.groupby("StoreID")[week_col]
            .nunique(dropna=True)
            .rename("Weeks_Active")
            .reset_index()
            .sort_values("Weeks_Active")
        )
        snaps["Weekly_Coverage_By_Store"] = cov
    else:
        rows_info.append({
            "Severity": "INFO",
            "Area": "Articles",
            "Issue": "Time granularity not detected",
            "Count": 0,
            "Details": "Provide MonthYear or WeekStart/WeekNo to compute Avg ROS more accurately.",
        })

    # Basic dataset sizes
    rows_info.extend([
        {
            "Severity": "INFO",
            "Area": "Sizes",
            "Issue": "Rows in StoresKPIs",
            "Count": int(len(stores)),
            "Details": "",
        },
        {
            "Severity": "INFO",
            "Area": "Sizes",
            "Issue": "Rows in TierKPIs",
            "Count": int(len(tiers)),
            "Details": "",
        },
        {
            "Severity": "INFO",
            "Area": "Sizes",
            "Issue": "Rows in Articles",
            "Count": int(len(articles)),
            "Details": "",
        },
    ])
    # ---------------- QC impact summaries ----------------------------
    # How much sales volume/value is affected by QC-critical rows.
    # This is a diagnostic to quantify data quality risk.
    if articles is not None and not articles.empty and "Sales_Value" in articles.columns:
        a = articles.copy()
        a["StoreID"] = a.get("StoreID", "").astype(str)
        # Define QC-critical conditions (rows that would distort decisions)
        pt = a.get("PriceTier", "").astype(str).str.strip()
        bad_pt = pt.isna() | (pt == "") | (pt.str.lower().isin(["nan", "none", "null"]))
        bad_sid = a["StoreID"].isna() | (a["StoreID"].str.strip() == "") | (a["StoreID"].str.lower().isin(["nan", "none", "null"]))
        # Numeric sanity
        sv = pd.to_numeric(a["Sales_Value"], errors="coerce")
        sq = pd.to_numeric(a.get("Sales_Qty", 0), errors="coerce")
        gm = pd.to_numeric(a.get("Gross_Margin", 0), errors="coerce")
        stock = pd.to_numeric(a.get("ClosingStockQty", 0), errors="coerce")
        avg_inv = pd.to_numeric(a.get("Avg_Inventory_Cost", np.nan), errors="coerce")

        bad_nums = sv.isna() | sq.isna() | gm.isna() | stock.isna()
        bad_inv = (~avg_inv.isna()) & (avg_inv <= 0)

        bad_mask = bad_pt | bad_sid | bad_nums | bad_inv

        a["_Sales_Value"] = sv.fillna(0.0)

        total_by_store = a.groupby("StoreID")["_Sales_Value"].sum().rename("Total_Sales_Value").reset_index()
        impacted_by_store = a.loc[bad_mask].groupby("StoreID")["_Sales_Value"].sum().rename("Impacted_Sales_Value").reset_index()
        impact_store = total_by_store.merge(impacted_by_store, on="StoreID", how="left").fillna({"Impacted_Sales_Value": 0.0})
        impact_store["Impact_%"] = np.where(
            impact_store["Total_Sales_Value"] > 0,
            (impact_store["Impacted_Sales_Value"] / impact_store["Total_Sales_Value"]) * 100.0,
            0.0,
        )
        impact_store = impact_store.sort_values("Impact_%", ascending=False)

        # By Tier
        a["_Tier"] = pt.replace({"": "BLANK"})
        total_by_tier = a.groupby("_Tier")["_Sales_Value"].sum().rename("Total_Sales_Value").reset_index().rename(columns={"_Tier": "PriceTier"})
        impacted_by_tier = a.loc[bad_mask].groupby("_Tier")["_Sales_Value"].sum().rename("Impacted_Sales_Value").reset_index().rename(columns={"_Tier": "PriceTier"})
        impact_tier = total_by_tier.merge(impacted_by_tier, on="PriceTier", how="left").fillna({"Impacted_Sales_Value": 0.0})
        impact_tier["Impact_%"] = np.where(
            impact_tier["Total_Sales_Value"] > 0,
            (impact_tier["Impacted_Sales_Value"] / impact_tier["Total_Sales_Value"]) * 100.0,
            0.0,
        )
        impact_tier = impact_tier.sort_values("Impact_%", ascending=False)

        snaps["QC_Impact_By_Store"] = impact_store
        snaps["QC_Impact_By_Tier"] = impact_tier



    return QCResult(
        errors=_issue_df(rows_err),
        warnings=_issue_df(rows_warn),
        info=_issue_df(rows_info),
        suggestions=_issue_df(rows_suggest),
        snapshots=snaps,
    )
