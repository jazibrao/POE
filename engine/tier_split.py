"""
Portfolio Optimization Engine - Tier Split Module
Version: 2.2.0 (Audit Fixes Applied)

This module handles:
- Splitting store total SKU targets across price tiers
- Alpha blending between current mix and efficient benchmark
- Demand-weighted benchmarking with TVI

Audit Fixes Applied:
- Phase 2.1: Simplified alpha blending with 3-tier system
- Enhanced transparency in alpha calculation documentation
"""

import numpy as np
import pandas as pd
from .config import EngineConfig
from .tiers import TIERS_ORDER as _TIER_ORDER


def _tier_below(tier: str) -> str | None:
    """Return the immediate lower tier label if known."""
    try:
        idx = _TIER_ORDER.index(str(tier))
    except ValueError:
        return None
    if idx <= 0:
        return None
    return _TIER_ORDER[idx - 1]


def _safe_div(a, b):
    b = np.where(np.asarray(b) == 0, np.nan, b)
    return np.asarray(a) / b


def _compute_demand_signals_from_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Compute Store×Tier demand signals from Articles data."""
    if articles is None or articles.empty:
        return pd.DataFrame(columns=[
            "StoreID", "PriceTier", "TierSalesQty", "TierSalesValue",
            "Months_Active_Store", "ASP_Tier", "ASP_Tier_P75",
        ])

    df = articles.copy()
    if "StoreID" not in df.columns or "PriceTier" not in df.columns:
        return pd.DataFrame(columns=[
            "StoreID", "PriceTier", "TierSalesQty", "TierSalesValue",
            "Months_Active_Store", "ASP_Tier", "ASP_Tier_P75",
        ])

    df["StoreID"] = df["StoreID"].astype(str)
    df["PriceTier"] = df["PriceTier"].astype(str)

    if "MonthYear" in df.columns:
        months_s = (
            df.groupby("StoreID")["MonthYear"]
            .nunique(dropna=True)
            .rename("Months_Active_Store")
        )
    else:
        months_s = pd.Series(1, index=df["StoreID"].unique(), name="Months_Active_Store")

    months_s.index = months_s.index.astype(str)
    months_s.index.name = "StoreID"
    months = months_s.reset_index()

    agg = (
        df.groupby(["StoreID", "PriceTier"]).agg(
            TierSalesQty=("Sales_Qty", "sum"),
            TierSalesValue=("Sales_Value", "sum"),
        )
        .reset_index()
    )

    agg["TierSalesQty"] = pd.to_numeric(agg["TierSalesQty"], errors="coerce").fillna(0).round().astype(int)
    agg["TierSalesValue"] = pd.to_numeric(agg["TierSalesValue"], errors="coerce").fillna(0.0)
    agg = agg.merge(months, on="StoreID", how="left")
    agg["Months_Active_Store"] = pd.to_numeric(agg["Months_Active_Store"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    agg["ASP_Tier"] = _safe_div(agg["TierSalesValue"].values, agg["TierSalesQty"].values)

    if "ItemColorName" in df.columns:
        sku = (
            df.groupby(["StoreID", "PriceTier", "ItemColorName"]).agg(
                _q=("Sales_Qty", "sum"),
                _v=("Sales_Value", "sum"),
            )
            .reset_index()
        )
        sku["_q"] = pd.to_numeric(sku["_q"], errors="coerce").fillna(0)
        sku["_v"] = pd.to_numeric(sku["_v"], errors="coerce").fillna(0.0)
        sku = sku[sku["_q"] > 0].copy()
        sku["ASP_SKU"] = _safe_div(sku["_v"].values, sku["_q"].values)
        p75 = sku.groupby(["StoreID", "PriceTier"])["ASP_SKU"].quantile(0.75).rename("ASP_Tier_P75").reset_index()
        agg = agg.merge(p75, on=["StoreID", "PriceTier"], how="left")
    else:
        agg["ASP_Tier_P75"] = np.nan

    return agg


def _default_eligibility(tiers_df: pd.DataFrame) -> dict:
    elig = {}
    for (store, tier), g in tiers_df.groupby(["StoreID", "PriceTier"]):
        cur = float(g["Current_SKUs_Tier"].iloc[0])
        elig.setdefault(str(store), {})[str(tier)] = 1 if cur > 0 else 0
    return elig


def _compute_simplified_alpha(gmroi_pctl: float, cfg: EngineConfig) -> float:
    """
    SIMPLIFIED ALPHA BLENDING (Phase 2.1 - Audit Fix)
    
    Instead of complex continuous alpha calculation, use a 3-tier system:
    
    | Store GMROI Percentile | Alpha Value | Interpretation |
    |------------------------|-------------|----------------|
    | Top 33% (>0.67)        | 0.20        | Keep current mix (high performer) |
    | Middle 33% (0.33-0.67) | 0.50        | Balanced blend |
    | Bottom 33% (<0.33)     | 0.80        | Move toward benchmark (needs improvement) |
    
    This simplification:
    1. Makes the logic transparent and auditable
    2. Reduces complexity while maintaining effectiveness
    3. Provides clear business rationale for each tier
    """
    high_threshold = float(getattr(cfg, "simplified_alpha_high_threshold", 0.67))
    low_threshold = float(getattr(cfg, "simplified_alpha_low_threshold", 0.33))
    
    alpha_high = float(getattr(cfg, "simplified_alpha_high_gmroi", 0.20))
    alpha_mid = float(getattr(cfg, "simplified_alpha_mid_gmroi", 0.50))
    alpha_low = float(getattr(cfg, "simplified_alpha_low_gmroi", 0.80))
    
    if gmroi_pctl >= high_threshold:
        return alpha_high
    elif gmroi_pctl >= low_threshold:
        return alpha_mid
    else:
        return alpha_low


def split_targets_across_tiers(
    tiers: pd.DataFrame,
    stores_enriched: pd.DataFrame,
    cfg: EngineConfig,
    articles_raw: pd.DataFrame | None = None,
) -> dict:
    """
    Split store total targets into tier-level targets.
    
    ALPHA BLENDING DOCUMENTATION:
    -----------------------------
    Alpha controls how much a store's tier mix should move toward the benchmark:
    - Alpha = 0: Keep current mix entirely
    - Alpha = 1: Adopt benchmark mix entirely
    
    Two modes available (controlled by cfg.use_simplified_alpha):
    
    1. SIMPLIFIED MODE (Recommended - Phase 2.1 Fix):
       - 3-tier system based on store GMROI percentile
       - High performers (top 33%): Alpha = 0.20
       - Mid performers (middle 33%): Alpha = 0.50
       - Low performers (bottom 33%): Alpha = 0.80
    
    2. LEGACY MODE:
       - Continuous alpha based on GMROI percentile
       - Alpha = 1 - GMROI_Pctl (with TVI adjustments)
    """

    # Auto-swap if arguments are reversed
    if tiers is not None and stores_enriched is not None:
        if "PriceTier" not in tiers.columns and "PriceTier" in stores_enriched.columns:
            tiers, stores_enriched = stores_enriched, tiers

    df = tiers.copy().rename(columns={"Average of SKU_DC": "Current_SKUs_Tier"})

    if "StoreID" not in df.columns and "Row Labels" in df.columns:
        df = df.rename(columns={"Row Labels": "StoreID"})
    if "StoreID" not in df.columns:
        raise KeyError("StoreID")
    df["StoreID"] = df["StoreID"].astype(str)
    df["PriceTier"] = df["PriceTier"].astype(str)

    df["Current_SKUs_Tier"] = pd.to_numeric(df["Current_SKUs_Tier"], errors="coerce").fillna(0)
    df["Current_SKUs_Tier"] = df["Current_SKUs_Tier"].round().astype(int)

    sid_col = "StoreID" if "StoreID" in stores_enriched.columns else ("Row Labels" if "Row Labels" in stores_enriched.columns else stores_enriched.columns[0])
    stores_enriched[sid_col] = stores_enriched[sid_col].astype(str)
    m_total_raw = stores_enriched.set_index(sid_col)["Target_SKU_Total"].to_dict()
    m_eff_raw = stores_enriched.set_index(sid_col)["Is_Efficient"].to_dict()
    m_total = {str(k): float(v) for k, v in m_total_raw.items()}
    m_eff = {str(k): int(v) for k, v in m_eff_raw.items()}

    df["Target_SKU_Total"] = df["StoreID"].map(lambda x: float(m_total.get(str(x), 0.0)))
    df["Is_Efficient"] = df["StoreID"].map(lambda x: int(m_eff.get(str(x), 0)))

    df = df.merge(df.groupby("StoreID")["Current_SKUs_Tier"].sum().rename("Current_SKU_Total"), on="StoreID", how="left")
    df["Current_SKU_Total"] = pd.to_numeric(df["Current_SKU_Total"], errors="coerce").fillna(0).round().astype(int)
    df["Current_Mix"] = df["Current_SKUs_Tier"] / df["Current_SKU_Total"].replace(0, np.nan)

    # ------------------------------------------------------------
    # Governance: Cap store-level change per run
    # ------------------------------------------------------------
    base_cap = getattr(cfg, "max_store_change_pct", None)
    if base_cap is not None and float(base_cap) < 1.0:
        base_cap = float(base_cap)
        cur_total = pd.to_numeric(df["Current_SKU_Total"], errors="coerce").fillna(0)
        tgt_total = pd.to_numeric(df["Target_SKU_Total"], errors="coerce").fillna(0)

        # Small-store stability: for stores below threshold, cap by absolute SKU count
        # instead of percentage — a 5-SKU store changing 2 SKUs = 40%, which is large
        # in % terms but operationally manageable; however changing 4 (80%) is not.
        small_thr = int(getattr(cfg, "small_store_threshold", 20))
        small_max = int(getattr(cfg, "small_store_max_change", 2))
        is_small = cur_total.between(1, small_thr, inclusive="both")

        # For small stores: cap = small_max absolute SKUs
        # For regular stores: cap = base_cap percentage
        lower = np.where(is_small, cur_total - small_max, np.floor(cur_total * (1.0 - base_cap)))
        upper = np.where(is_small, cur_total + small_max, np.ceil(cur_total * (1.0 + base_cap)))
        lower = np.maximum(lower, 0)

        has_baseline = cur_total > 0
        capped = np.where(has_baseline, np.clip(tgt_total, lower, upper), tgt_total)
        df["Target_SKU_Total"] = np.round(capped).astype(int)

    # ------------------------------------------------------------
    # GMROI Percentile Calculation
    # ------------------------------------------------------------
    if "GMROI" in stores_enriched.columns:
        s_gm = pd.to_numeric(stores_enriched["GMROI"], errors="coerce").fillna(0.0)
        stores_enriched["Store_GMROI_Pctl"] = s_gm.rank(pct=True, method="average")
        m_pctl = stores_enriched.set_index(sid_col)["Store_GMROI_Pctl"].to_dict()
        df["Store_GMROI_Pctl"] = df["StoreID"].map(lambda x: float(m_pctl.get(str(x), 0.0)))
    else:
        df["Store_GMROI_Pctl"] = 0.0

    # ------------------------------------------------------------
    # ALPHA CALCULATION (Simplified or Legacy)
    # ------------------------------------------------------------
    use_simplified = getattr(cfg, "use_simplified_alpha", True)
    
    if use_simplified:
        # SIMPLIFIED 3-TIER ALPHA (Phase 2.1 Fix)
        df["Alpha"] = df["Store_GMROI_Pctl"].apply(lambda x: _compute_simplified_alpha(x, cfg))
        df["Alpha_Mode"] = "Simplified_3Tier"
        
        # Document the alpha assignment
        high_threshold = float(getattr(cfg, "simplified_alpha_high_threshold", 0.67))
        low_threshold = float(getattr(cfg, "simplified_alpha_low_threshold", 0.33))
        df["Alpha_Tier"] = np.where(
            df["Store_GMROI_Pctl"] >= high_threshold, "High_Performer",
            np.where(df["Store_GMROI_Pctl"] >= low_threshold, "Mid_Performer", "Low_Performer")
        )
    else:
        # LEGACY CONTINUOUS ALPHA
        df["Alpha_base_raw"] = 1.0 - df["Store_GMROI_Pctl"]
        df["Alpha_base"] = np.clip(df["Alpha_base_raw"], cfg.alpha_min, cfg.alpha_max)
        df["Alpha"] = df["Alpha_base"]
        df["Alpha_Mode"] = "Legacy_Continuous"
        df["Alpha_Tier"] = "Continuous"

    # ------------------------------------------------------------
    # Efficient_Mix reference
    # ------------------------------------------------------------
    eff_mix = df[df["Is_Efficient"] == 1].groupby("PriceTier")["Current_SKUs_Tier"].sum()
    eff_mix = (eff_mix / eff_mix.sum()).rename("Efficient_Mix").reset_index()
    df = df.merge(eff_mix, on="PriceTier", how="left").fillna({"Efficient_Mix": 0.0})

    elig = cfg.eligibility_matrix or _default_eligibility(df)
    df["Eligibility_Matrix"] = df.apply(lambda r: int(elig.get(r["StoreID"], {}).get(r["PriceTier"], 0)), axis=1)

    # ------------------------------------------------------------
    # Demand-weighted benchmarking
    # ------------------------------------------------------------
    df["DemandMultiplier"] = 1.0
    df["ASP_Pressure"] = np.nan
    df["IntroEligible"] = 1
    k_summary = pd.DataFrame()

    if getattr(cfg, "enable_demand_weighted_benchmark", False) and articles_raw is not None and not articles_raw.empty:
        demand = _compute_demand_signals_from_articles(articles_raw)
        if not demand.empty:
            df = df.merge(
                demand[["StoreID", "PriceTier", "TierSalesQty", "TierSalesValue", "Months_Active_Store", "ASP_Tier", "ASP_Tier_P75"]],
                on=["StoreID", "PriceTier"],
                how="left",
            )
            df[["TierSalesQty", "TierSalesValue", "Months_Active_Store"]] = df[[
                "TierSalesQty", "TierSalesValue", "Months_Active_Store",
            ]].fillna(0)

            df["TierSalesQty"] = pd.to_numeric(df["TierSalesQty"], errors="coerce").fillna(0).round().astype(int)
            df["Months_Active_Store"] = pd.to_numeric(df["Months_Active_Store"], errors="coerce").fillna(1).clip(lower=1).round().astype(int)

            df["Tier_ROS_perSKU_perMonth"] = (
                _safe_div(df["TierSalesQty"].values, np.maximum(df["Current_SKUs_Tier"].values, 1))
                / np.maximum(df["Months_Active_Store"].values, 1)
            )

            net = df.groupby("PriceTier").agg(
                NetSalesQty=("TierSalesQty", "sum"),
                NetCurrentSKUs=("Current_SKUs_Tier", "sum"),
            )
            median_months = int(np.nanmedian(df["Months_Active_Store"].values)) if len(df) else 1
            median_months = max(median_months, 1)
            net["Net_ROS_perSKU_perMonth"] = (
                _safe_div(net["NetSalesQty"].values, np.maximum(net["NetCurrentSKUs"].values, 1)) / median_months
            )
            net = net.reset_index()
            df = df.merge(net[["PriceTier", "Net_ROS_perSKU_perMonth"]], on="PriceTier", how="left")

            df["TVI"] = _safe_div(df["Tier_ROS_perSKU_perMonth"].values, df["Net_ROS_perSKU_perMonth"].values)

            # TVI clamping — first replace infinities, then clamp
            df["TVI"] = df["TVI"].astype(float).replace([np.inf, -np.inf], np.nan)
            clamp_min = float(getattr(cfg, "tvi_clamp_min", 0.70))
            clamp_max = float(getattr(cfg, "tvi_clamp_max", 1.30))
            df["TVI_Clamped"] = np.clip(df["TVI"].fillna(1.0).values, clamp_min, clamp_max)

            df["DemandMultiplier"] = 1.0
            stable = (df["Current_SKUs_Tier"] > 0) & (df["TierSalesQty"] >= int(getattr(cfg, "min_tier_sales_qty_for_tvi", 5)))

            # Evidence-weighted TVI
            k_scale = 0.50
            k_map = demand.groupby("PriceTier")["TierSalesQty"].median() * k_scale
            overall_k = float(np.nanmedian(k_map.values)) if len(k_map) else 10.0
            df["k_tier"] = df["PriceTier"].map(k_map).astype(float)
            df["k_tier"] = df["k_tier"].fillna(overall_k).clip(lower=1.0)

            qty = df["TierSalesQty"].astype(float).fillna(0.0)
            w = _safe_div(qty.values, (qty.values + df["k_tier"].astype(float).values))
            w = np.clip(w, 0.0, 1.0)

            df.loc[stable, "DemandMultiplier"] = (
                1.0 + w[stable.values] * (df.loc[stable, "TVI_Clamped"].astype(float).values - 1.0)
            )

            # k_summary for Model Summary
            k_summary = (
                demand.groupby("PriceTier")["TierSalesQty"]
                .quantile([0.25, 0.50, 0.75])
                .unstack(level=-1)
                .rename(columns={0.25: "Q25", 0.50: "Q50_Median", 0.75: "Q75"})
                .reset_index()
                .rename(columns={"PriceTier": "Tier"})
            )

            # ASP pressure for empty tiers
            net_asp = demand.groupby("PriceTier")["ASP_Tier_P75"].median().rename("Net_ASP_P75").reset_index()
            demand_asp = demand[["StoreID", "PriceTier", "ASP_Tier_P75"]]

            lower_rows = []
            for t in df["PriceTier"].unique():
                lb = _tier_below(t)
                if lb is None:
                    continue
                lower_rows.append((t, lb))
            tier_to_lower = dict(lower_rows)

            df["LowerTier"] = df["PriceTier"].map(lambda x: tier_to_lower.get(str(x)))
            df = df.merge(
                demand_asp.rename(columns={"PriceTier": "LowerTier", "ASP_Tier_P75": "Store_ASP_Lower_P75"}),
                on=["StoreID", "LowerTier"],
                how="left",
            )
            df = df.merge(
                net_asp.rename(columns={"PriceTier": "LowerTier", "Net_ASP_P75": "Net_ASP_Lower_P75"}),
                on=["LowerTier"],
                how="left",
            )

            df["ASP_Pressure"] = _safe_div(df["Store_ASP_Lower_P75"].values, df["Net_ASP_Lower_P75"].values)

            thr = float(getattr(cfg, "asp_pressure_threshold", 1.0))
            is_empty = df["Current_SKUs_Tier"] <= 0
            pressure_ok = (df["ASP_Pressure"].astype(float) > thr)

            df["IntroEligible"] = np.where(is_empty, pressure_ok.astype(int), 1)
            df.loc[is_empty & pressure_ok, "DemandMultiplier"] = 1.0
            df.loc[is_empty & (~pressure_ok), "DemandMultiplier"] = 0.0

            df.drop(columns=["LowerTier"], inplace=True, errors="ignore")

    # Final eligibility
    if getattr(cfg, "empty_tier_requires_matrix", True):
        df["Eligibility"] = (df["Eligibility_Matrix"] * df["IntroEligible"]).astype(int)
    else:
        df["Eligibility"] = np.where(df["Current_SKUs_Tier"] > 0, df["Eligibility_Matrix"], (df["Eligibility_Matrix"] | df["IntroEligible"]).astype(int))

    # Target mix calculation
    eff_mix_adj = df["Efficient_Mix"] * df["DemandMultiplier"].astype(float)
    df["Target_Mix_raw"] = np.where(
        df["Eligibility"] == 0,
        0.0,
        df["Alpha"] * eff_mix_adj + (1 - df["Alpha"]) * df["Current_Mix"].fillna(0.0),
    )

    # Per-tier mix change cap: prevent any single tier's share from shifting more than
    # max_tier_mix_shift in one run (e.g., 30% means a tier at 50% can only go to 20%-80%)
    mix_cap = float(getattr(cfg, "max_tier_mix_shift", 0.30))
    if mix_cap > 0:
        cur_mix = df["Current_Mix"].fillna(0.0)
        raw = df["Target_Mix_raw"]
        # Re-normalize raw to sum to 1 per store before capping
        raw_sum = df.groupby("StoreID")["Target_Mix_raw"].transform("sum").replace(0, np.nan)
        raw_norm = (raw / raw_sum).fillna(0.0)
        # Clamp shift
        df["Target_Mix_raw"] = np.where(
            df["Eligibility"] == 1,
            np.clip(raw_norm, cur_mix - mix_cap, cur_mix + mix_cap),
            0.0,
        )

    denom = df.groupby("StoreID")["Target_Mix_raw"].transform("sum").replace(0, np.nan)
    df["Target_Mix"] = (df["Target_Mix_raw"] / denom).fillna(0.0)

    df["Target_SKUs_Tier_raw"] = df["Target_Mix"] * df["Target_SKU_Total"]
    df["Floor"] = np.floor(df["Target_SKUs_Tier_raw"]).astype(int)
    df["Fraction"] = df["Target_SKUs_Tier_raw"] - df["Floor"]

    def reconcile(g: pd.DataFrame) -> pd.DataFrame:
        total = int(g["Target_SKU_Total"].iloc[0])
        residual = total - int(g["Floor"].sum())
        g = g.copy()
        g["Add1"] = 0
        if residual > 0:
            eligible = g["Eligibility"].values.astype(bool)
            idx_pool = g.index[eligible].to_numpy()
            if len(idx_pool):
                order = np.argsort(-g.loc[idx_pool, "Fraction"].values)
                g.loc[idx_pool[order[:residual]], "Add1"] = 1
        g["Target_SKUs_Tier"] = g["Floor"] + g["Add1"]
        return g

    _storeid_map = df["StoreID"].to_dict()  # Map index -> StoreID for safe recovery
    try:
        df = df.groupby("StoreID", group_keys=False).apply(reconcile, include_groups=False)
    except TypeError:
        df = df.groupby("StoreID", group_keys=False).apply(reconcile)
    if "StoreID" not in df.columns:
        df["StoreID"] = df.index.map(lambda idx: _storeid_map.get(idx, "UNKNOWN")).astype(str)

    df["Gap"] = df["Target_SKUs_Tier"] - df["Current_SKUs_Tier"]
    df["Gap"] = df["Gap"].clip(lower=-df["Current_SKUs_Tier"].fillna(0).astype(int))

    return {"tier_targets": df, "efficient_mix": eff_mix, "k_summary": k_summary}
