"""Peer store selection based on Q-score behaviour.

Business definition
------------------
Q_score(Store, Tier) = ASP_tier(Store, Tier) × STR_tier(Store, Tier)

Where ASP_tier and STR_tier are calculated over the analysis period.

We cluster stores using tier-wise behaviour signals:
 - ASP_tier
 - STR_tier
 - UnitsSold_tier (stabilizer so tiny stores don't get matched with flagships)

The cluster ID becomes the peer group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .tiers import TierCutoffs, compute_cutoffs, assign_tier_from_price, normalize_tier_labels, TIERS_ORDER
from .config import EngineConfig
from .articles import _find_col_fuzzy


@dataclass
class PeerResult:
    store_clusters: pd.DataFrame  # StoreID, PeerClusterID
    peer_map: Dict[str, List[str]]  # StoreID -> list of peer stores
    store_tier_metrics: pd.DataFrame
    sales_tier_cutoffs: TierCutoffs


def _compute_str(sales_qty: pd.Series, closing_stock_qty: pd.Series) -> pd.Series:
    """Sell-through proxy.

    STR = SalesQty / (SalesQty + ClosingStockQty)

    This is a practical proxy when opening stock / receipts are not available.
    """
    s = pd.to_numeric(sales_qty, errors="coerce").fillna(0)
    c = pd.to_numeric(closing_stock_qty, errors="coerce").fillna(0)
    denom = (s + c).replace(0, np.nan)
    return (s / denom).fillna(0)


def build_peer_groups_from_articles(articles_raw: pd.DataFrame, cfg: EngineConfig) -> PeerResult:
    """Build peer groups using only features available in input data."""

    df = articles_raw.copy()
    df["StoreID"] = df["StoreID"].astype(str)

    # --- Determine ASP column ---
    asp_col = _find_col_fuzzy(df, ["ASP", "AvgSellingPrice", "Average Selling Price", "Avg_Selling_Price"])
    if asp_col is None:
        # Fallback: try derive ASP = Sales_Value / Sales_Qty
        if "Sales_Value" in df.columns and "Sales_Qty" in df.columns:
            df["ASP"] = (pd.to_numeric(df["Sales_Value"], errors="coerce") / pd.to_numeric(df["Sales_Qty"], errors="coerce").replace(0, np.nan)).fillna(0)
            asp_col = "ASP"
        else:
            # No ASP signal -> return a single cluster
            stores = sorted(df["StoreID"].unique())
            sc = pd.DataFrame({"StoreID": stores, "PeerClusterID": 0})
            peer_map = {s: [x for x in stores if x != s] for s in stores}
            dummy_cut = compute_cutoffs(pd.Series([np.nan]), cfg.sales_tier_p_value, cfg.sales_tier_p_mid, cfg.sales_tier_p_premium)
            return PeerResult(sc, peer_map, pd.DataFrame(), dummy_cut)

    # --- Assign SalesTier_ASP using network-wide percentiles (sold articles) ---
    sales_cut = compute_cutoffs(df[asp_col], cfg.sales_tier_p_value, cfg.sales_tier_p_mid, cfg.sales_tier_p_premium)
    df["SalesTier_ASP"] = assign_tier_from_price(df[asp_col], sales_cut)
    df["SalesTier_ASP"] = normalize_tier_labels(df["SalesTier_ASP"])

    # --- STR needs Sales_Qty and ClosingStockQty ---
    sales_qty_col = _find_col_fuzzy(df, ["Sales_Qty", "SalesQty", "QtySold", "UnitsSold"])
    stock_col = _find_col_fuzzy(df, ["ClosingStockQty", "StoreStock", "Closing_Stock_Qty", "StockQty"])
    if sales_qty_col is None:
        df["Sales_Qty"] = 0
        sales_qty_col = "Sales_Qty"
    if stock_col is None:
        df["ClosingStockQty"] = 0
        stock_col = "ClosingStockQty"

    df["STR"] = _compute_str(df[sales_qty_col], df[stock_col])

    # --- Compute tier-wise metrics per store ---
    grp = df.groupby(["StoreID", "SalesTier_ASP"], dropna=False)
    units = grp[sales_qty_col].sum().rename("UnitsSold_Tier")
    def _wavg(g: pd.DataFrame) -> float:
        vals = pd.to_numeric(g[asp_col], errors="coerce").fillna(0)
        w = pd.to_numeric(g[sales_qty_col], errors="coerce").fillna(0) + 1e-6
        return float(np.average(vals, weights=w)) if len(vals) else 0.0

    # Pandas 2.1+ supports include_groups=False to prevent FutureWarnings.
    try:
        asp_w = grp.apply(_wavg, include_groups=False).rename("ASP_Tier")
    except TypeError:
        asp_w = grp.apply(_wavg).rename("ASP_Tier")
    str_m = grp["STR"].mean().rename("STR_Tier")
    met = pd.concat([units, asp_w, str_m], axis=1).reset_index()
    met["Q_Score_Tier"] = met["ASP_Tier"] * met["STR_Tier"]

    # --- Wide features for clustering ---
    feats = []
    for tier in TIERS_ORDER:
        sub = met[met["SalesTier_ASP"] == tier][["StoreID", "ASP_Tier", "STR_Tier", "UnitsSold_Tier"]].copy()
        sub = sub.set_index("StoreID")
        sub.columns = [f"{c}_{tier}" for c in sub.columns]
        feats.append(sub)
    X = pd.concat(feats, axis=1).fillna(0)

    # Standardize columns for clustering stability
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0).replace(0, 1))
    Xn = Xn.fillna(0)

    stores = Xn.index.to_list()
    if len(stores) < 3:
        sc = pd.DataFrame({"StoreID": stores, "PeerClusterID": 0})
        peer_map = {s: [x for x in stores if x != s] for s in stores}
        return PeerResult(sc, peer_map, met, sales_cut)

    k = int(getattr(cfg, "peer_cluster_k", 0) or 0)
    if k <= 0:
        # Data-driven K: wide-enough peer pools without being too generic
        # Rule of thumb: k ≈ sqrt(N/3) clamped to [4, 10]
        import math
        k = int(round(math.sqrt(max(len(stores), 1) / 3.0)))
        k = max(4, min(10, k))
    k = max(2, min(k, len(stores)))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(Xn.values)

    sc = pd.DataFrame({"StoreID": stores, "PeerClusterID": labels.astype(int)})

    # Merge tiny clusters into nearest bigger cluster
    counts = sc["PeerClusterID"].value_counts().to_dict()
    big_clusters = {cid for cid, cnt in counts.items() if cnt >= cfg.peer_min_cluster_size}
    if big_clusters:
        centroids = {cid: km.cluster_centers_[cid] for cid in range(k)}
        for idx, row in sc.iterrows():
            cid = int(row["PeerClusterID"])
            if counts.get(cid, 0) >= cfg.peer_min_cluster_size:
                continue
            # find nearest big cluster centroid
            v = Xn.loc[row["StoreID"]].values
            best = None
            best_d = 1e18
            for bcid in big_clusters:
                d = float(np.linalg.norm(v - centroids[bcid]))
                if d < best_d:
                    best_d = d
                    best = bcid
            if best is not None:
                sc.at[idx, "PeerClusterID"] = int(best)

    # Build peer map using cosine similarity within cluster
    peer_map: Dict[str, List[str]] = {}
    for cid, g in sc.groupby("PeerClusterID"):
        ids = g["StoreID"].astype(str).tolist()
        if len(ids) <= 1:
            for s in ids:
                peer_map[s] = []
            continue

        M = Xn.loc[ids].values
        sim = cosine_similarity(M)
        for i, s in enumerate(ids):
            order = np.argsort(-sim[i])
            peers = [ids[j] for j in order if ids[j] != s]
            peer_map[s] = peers[: int(cfg.peer_top_k_within_cluster)]

    return PeerResult(sc, peer_map, met, sales_cut)