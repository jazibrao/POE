"""
Portfolio Optimization Engine - Core Pipeline
Version: 3.0.0

Single-source-of-truth pipeline that is called by:
  - app.py (Streamlit UI)
  - cli_run_engine.py (native CLI)
  - cli_engine.py (app-wrapper CLI)

This avoids duplicating the optimization step sequence across entry points.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import EngineConfig
from .store_model import build_total_target_model
from .tier_split import split_targets_across_tiers
from .articles import aggregate_articles_assign_tier
from .decisions import (
    build_network_stock,
    attach_scores,
    create_remove_list,
    create_peer_pool_adds,
    apply_active_filter,
    flag_circular_recommendations,
    create_transfer_recommendations,
)
from .output_format import format_output

logger = logging.getLogger(__name__)


def run_optimization_pipeline(
    stores_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    cfg: EngineConfig,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """
    Execute the full portfolio optimization pipeline.

    Parameters
    ----------
    stores_df : pd.DataFrame
        Store-level KPIs (GMROI, SPF, Q_SCORE, RFT, Average of SKU_DC).
    tiers_df : pd.DataFrame
        Store × PriceTier current SKU counts.
    articles_df : pd.DataFrame
        Transaction-level line items.
    cfg : EngineConfig
        All optimization parameters.
    progress_cb : callable, optional
        Callback ``(message: str, pct: int) -> None`` for progress reporting.

    Returns
    -------
    dict
        Keys include: stores_enriched, model_info, tier_targets, k_summary,
        sku_master, sku_scored, network_stock, remove_list, add_list,
        transfer_recommendations, shortfall.
    """

    def _cb(msg: str, pct: int = 0) -> None:
        logger.info(msg)
        if progress_cb:
            try:
                progress_cb(msg, pct)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Step 1: Store target model
    # ------------------------------------------------------------------
    _cb("Building store target model...", 10)
    model_res = build_total_target_model(stores_df, cfg)
    stores_enriched = model_res["stores_enriched"]
    model_info = model_res.get("model", {})

    # ------------------------------------------------------------------
    # Step 2: Tier split
    # ------------------------------------------------------------------
    _cb("Splitting targets across tiers...", 30)
    tier_res = split_targets_across_tiers(tiers_df, stores_enriched, cfg, articles_raw=articles_df)
    tier_targets = tier_res["tier_targets"]
    k_summary = tier_res.get("k_summary", pd.DataFrame())

    # ------------------------------------------------------------------
    # Step 3: Article aggregation & scoring
    # ------------------------------------------------------------------
    _cb("Aggregating articles and scoring...", 50)
    agg_res = aggregate_articles_assign_tier(articles_df)
    sku_master = agg_res["sku_master"]

    network_stock = build_network_stock(sku_master)
    sku_master_scoring = sku_master.merge(network_stock, on="ItemColorName", how="left").fillna({"NetworkStock": 0})
    sku_scored = attach_scores(sku_master_scoring, cfg)

    # ------------------------------------------------------------------
    # Step 4: Remove list
    # ------------------------------------------------------------------
    _cb("Generating Remove List...", 70)
    remove_list = create_remove_list(sku_scored, tier_targets, cfg)
    remove_list = apply_active_filter(remove_list, network_stock, cfg)

    # ------------------------------------------------------------------
    # Step 5: Add list (peer pool)
    # ------------------------------------------------------------------
    _cb("Generating Add List from Peer Pool...", 85)
    add_list = create_peer_pool_adds(
        sku_scored,
        stores_enriched,
        tier_targets,
        cfg,
        articles_raw=articles_df,
    )
    add_list = apply_active_filter(add_list, network_stock, cfg, sku_col="ItemColorName")

    # ------------------------------------------------------------------
    # Step 5b: Circular recommendation detection
    # ------------------------------------------------------------------
    remove_list, add_list = flag_circular_recommendations(remove_list, add_list)

    # ------------------------------------------------------------------
    # Step 6: Inter-store transfers (optional)
    # ------------------------------------------------------------------
    _cb("Generating transfer recommendations...", 92)
    transfer_recommendations = pd.DataFrame()
    try:
        transfer_recommendations = create_transfer_recommendations(
            sku_scored, tier_targets, cfg, articles_raw=articles_df,
        )
    except Exception as e:
        logger.warning(f"Transfer recommendations could not be generated: {e}")

    # ------------------------------------------------------------------
    # Step 7: Shortfall report
    # ------------------------------------------------------------------
    _cb("Building shortfall report...", 95)
    shortfall = _build_shortfall_report(tier_targets, add_list)

    _cb("Pipeline complete.", 100)

    return {
        "stores_enriched": stores_enriched,
        "model_info": model_info,
        "tier_targets": tier_targets,
        "k_summary": k_summary,
        "sku_master": sku_master,
        "sku_scored": sku_scored,
        "network_stock": network_stock,
        "remove_list": remove_list,
        "add_list": add_list,
        "transfer_recommendations": transfer_recommendations,
        "shortfall": shortfall,
    }


def _build_shortfall_report(
    tier_targets: pd.DataFrame,
    add_list: pd.DataFrame,
) -> pd.DataFrame:
    """Build shortfall report showing unfilled add gaps per Store × Tier."""
    if tier_targets is None or tier_targets.empty or "Gap" not in tier_targets.columns:
        return pd.DataFrame()

    g = tier_targets[["StoreID", "PriceTier", "Gap"]].copy()
    g["Gap"] = pd.to_numeric(g["Gap"], errors="coerce").fillna(0).astype(int)

    if add_list is not None and not add_list.empty and "StoreID_Target" in add_list.columns:
        adds_cnt = (
            add_list.groupby(["StoreID_Target", "PriceTier"])
            .size()
            .rename("Add_Count")
            .reset_index()
            .rename(columns={"StoreID_Target": "StoreID"})
        )
        merged = g.merge(adds_cnt, on=["StoreID", "PriceTier"], how="left").fillna({"Add_Count": 0})
    else:
        merged = g.copy()
        merged["Add_Count"] = 0

    merged["Add_Count"] = merged["Add_Count"].astype(int)
    merged["Shortfall"] = np.where(
        merged["Gap"] > 0,
        np.maximum(merged["Gap"] - merged["Add_Count"], 0),
        0,
    )
    shortfall = merged[merged["Shortfall"] > 0].copy()

    if not shortfall.empty:
        def _rec(row):
            prov = int(row.get("Add_Count", 0) or 0)
            gap = int(row.get("Gap", 0) or 0)
            fr = (prov / gap) if gap > 0 else 1.0
            if prov == 0:
                return "expand peer pool"
            if fr < 0.50:
                return "relax lifecycle restriction"
            return "accept tier mix adjustment"

        shortfall["Recommended_Action"] = shortfall.apply(_rec, axis=1)

    return shortfall
