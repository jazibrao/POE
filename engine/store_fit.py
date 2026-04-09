"""Store-fit scoring with confidence weighting.

Hybrid score (ROS + GMROI blended percentiles) answers:
  "Is this article generally strong?"

Store-fit answers:
  "Is this article strong *for this store*?"

We compute store-fit using only fields available in the input dataset.

Similarity definition
---------------------
Two items are considered similar if they share:
  - Dept
  - Category
  - AddTier_RP (dynamic tier based on Retail Price, Active-only cutoffs)

Store-fit computation
--------------------
For a candidate (Store S, Item A), we compute:
  Hybrid_store = median(Hybrid of similar items already present in S)
  Hybrid_peers = median(Hybrid of similar items across S's peer stores)

Then blend using confidence based on evidence count n:
  conf(n) = 1 - exp(-n / scale)
  StoreFit = conf * Hybrid_store + (1-conf) * Hybrid_peers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import EngineConfig


@dataclass
class StoreFitTables:
    store_stats: pd.DataFrame
    peer_stats: pd.DataFrame


def _conf(n: float, scale: float) -> float:
    try:
        n = float(n)
    except Exception:
        n = 0.0
    if n <= 0:
        return 0.0
    return float(1.0 - np.exp(-n / max(scale, 1e-6)))


def build_store_fit_tables(
    sku_scored: pd.DataFrame,
    peer_map: Dict[str, list],
    cfg: EngineConfig,
    dept_col: str = "Dept",
    cat_col: str = "Category",
    tier_col: str = "Add_Tier_RP",
    hybrid_col: str = "Score_AddTier",
) -> StoreFitTables:
    """Precompute store and peer similarity stats for fast lookups."""

    df = sku_scored.copy()
    df["StoreID"] = df["StoreID"].astype(str)

    # If fields are missing, return empty tables (caller will fallback)
    for c in [dept_col, cat_col, tier_col, hybrid_col]:
        if c not in df.columns:
            empty = pd.DataFrame(columns=["StoreID", dept_col, cat_col, tier_col, "Hybrid_median", "n"])
            return StoreFitTables(empty, empty)

    # Store-level stats
    g = df.groupby(["StoreID", dept_col, cat_col, tier_col], dropna=False)[hybrid_col]
    store_stats = g.agg(Hybrid_median="median", n="count").reset_index()

    # Peer stats: aggregate over each store's peer set
    # To keep it fast and stable, we compute peer stats per (cluster) by pooling peer stores.
    # If peer_map is empty, peer stats stays empty.
    rows = []
    if peer_map:
        # Build quick lookup of store->subset df index
        for store, peers in peer_map.items():
            store = str(store)
            peers = [str(p) for p in (peers or [])]
            if not peers:
                continue
            sub = df[df["StoreID"].isin(peers)]
            if sub.empty:
                continue
            gg = sub.groupby([dept_col, cat_col, tier_col], dropna=False)[hybrid_col]
            tmp = gg.agg(Hybrid_peer_median="median").reset_index()
            tmp["StoreID"] = store
            rows.append(tmp)

    peer_stats = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["StoreID", dept_col, cat_col, tier_col, "Hybrid_peer_median"]
    )

    return StoreFitTables(store_stats=store_stats, peer_stats=peer_stats)


def score_store_fit_for_candidates(
    candidates: pd.DataFrame,
    tables: StoreFitTables,
    cfg: EngineConfig,
    dept_col: str = "Dept",
    cat_col: str = "Category",
    tier_col: str = "Add_Tier_RP",
) -> pd.DataFrame:
    """Attach StoreFitScore, Confidence_n, SimilarCountInStore to candidates."""

    if candidates is None or candidates.empty:
        return candidates

    cand = candidates.copy()
    cand["StoreID_Target"] = cand["StoreID_Target"].astype(str)

    # Default fallbacks
    cand["SimilarCountInStore"] = 0
    cand["Confidence"] = 0.0
    cand["StoreFitScore"] = 0.0

    if tables.store_stats.empty and tables.peer_stats.empty:
        return cand

    # Merge store stats
    ss = tables.store_stats.rename(columns={"StoreID": "StoreID_Target"})
    key_cols = ["StoreID_Target", dept_col, cat_col, tier_col]
    cand = cand.merge(ss[key_cols + ["Hybrid_median", "n"]], on=key_cols, how="left")
    cand["Hybrid_median"] = cand["Hybrid_median"].fillna(0.0)
    cand["n"] = cand["n"].fillna(0)

    # Merge peer stats
    ps = tables.peer_stats.rename(columns={"StoreID": "StoreID_Target"})
    cand = cand.merge(ps[key_cols + ["Hybrid_peer_median"]], on=key_cols, how="left")
    cand["Hybrid_peer_median"] = cand["Hybrid_peer_median"].fillna(cand["Hybrid_median"])

    # Confidence blend
    conf_vals = cand["n"].map(lambda x: _conf(x, cfg.store_fit_conf_scale))
    cand["SimilarCountInStore"] = cand["n"].astype(int)
    cand["Confidence"] = conf_vals.astype(float)

    cand["StoreFitScore"] = cand["Confidence"] * cand["Hybrid_median"] + (1 - cand["Confidence"]) * cand["Hybrid_peer_median"]

    # Cleanup
    cand.drop(columns=["Hybrid_median", "Hybrid_peer_median", "n"], inplace=True, errors="ignore")
    return cand
