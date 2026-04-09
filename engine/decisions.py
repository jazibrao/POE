"""
Portfolio Optimization Engine - Decisions Module
Version: 2.2.0 (Audit Fixes Applied)

This module handles:
- SKU scoring with GMROI, ROS, and Discount Penalty
- Remove list generation with documented logic
- Add list generation with deduplication and expanded candidate pool
- Inter-store transfer recommendations (new feature)

Audit Fixes Applied:
- Fix C1: Duplicate SKU deduplication in Add_List
- Fix C2: Expanded candidate pool with fallback logic
- Fix C3: Documented remove list logic with transparency
- Phase 2.2: Discount penalty in scoring
- Phase 2.3: Inter-store transfer recommendations
"""

import logging
import pandas as pd
import numpy as np
from .config import EngineConfig
from .tiers import normalize_tier_labels

logger = logging.getLogger(__name__)


def _pct_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True, method="average")


def _build_presence_dicts(
    sku_df: pd.DataFrame,
    articles_raw: pd.DataFrame | None = None,
) -> tuple[dict[str, set], dict[str, set]]:
    """Build per-store SKU presence dictionaries (computed once, used by Add + Transfers).

    Returns
    -------
    present_any : dict[StoreID, set[ItemColorName]]
        All SKU records regardless of stock.
    present_stock : dict[StoreID, set[ItemColorName]]
        Only SKUs with positive store stock.
    """
    present_any = sku_df.groupby("StoreID")["ItemColorName"].apply(set).to_dict()
    present_stock = sku_df[sku_df["StoreStock"] > 0].groupby("StoreID")["ItemColorName"].apply(set).to_dict()

    # Prefer full Articles table for presence (safer than scored subset)
    if articles_raw is not None and not articles_raw.empty:
        try:
            ar = articles_raw.copy()
            ar["StoreID"] = ar["StoreID"].astype(str)
            ar["ItemColorName"] = ar["ItemColorName"].astype(str)
            present_any = ar.groupby("StoreID")["ItemColorName"].apply(set).to_dict()

            stock_col = None
            for c in ["StoreStock", "ClosingStockQty", "StockQty", "Stock"]:
                if c in ar.columns:
                    stock_col = c
                    break
            if stock_col is not None:
                ar[stock_col] = pd.to_numeric(ar[stock_col], errors="coerce").fillna(0)
                present_stock = ar[ar[stock_col] > 0].groupby("StoreID")["ItemColorName"].apply(set).to_dict()
        except Exception as e:
            logger.warning(f"Failed to build presence from raw articles, using scored subset: {e}")

    return present_any, present_stock


def build_network_stock(sku_master: pd.DataFrame) -> pd.DataFrame:
    return sku_master.groupby("ItemColorName")["StoreStock"].sum().rename("NetworkStock").reset_index()


def attach_scores(sku_master: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    """
    Attach performance scores to SKUs.
    
    Scoring Components:
    - GMROI: Gross Margin Return on Investment
    - ROS: Rate of Sale
    - PricePower: Inverse of discount (optional)
    - DiscountPenalty: Penalty for high-tier SKUs sold at heavy discount (NEW)
    
    The final Score is a weighted combination of these metrics.
    """
    df = sku_master.copy()
    
    # ------------------------------------------------------------------
    # ROS Calculation
    # ------------------------------------------------------------------
    if "ROS" not in df.columns:
        qty = pd.to_numeric(df.get("Sales_Qty", 0), errors="coerce").fillna(0)

        if "Weeks_Active" in df.columns:
            w = pd.to_numeric(df["Weeks_Active"], errors="coerce").fillna(1).clip(lower=1)
            df["ROS"] = (qty / w).fillna(0)
        elif "Weeks" in df.columns:
            w = pd.to_numeric(df["Weeks"], errors="coerce").fillna(1).clip(lower=1)
            df["ROS"] = (qty / w).fillna(0)
        elif "Months_Active" in df.columns:
            m = pd.to_numeric(df["Months_Active"], errors="coerce").fillna(1).clip(lower=1)
            df["ROS"] = (qty / m).fillna(0)
        elif "CountofMonthYear" in df.columns:
            m = pd.to_numeric(df["CountofMonthYear"], errors="coerce").fillna(1).clip(lower=1)
            df["ROS"] = (qty / m).fillna(0)
        elif "Months" in df.columns:
            m = pd.to_numeric(df["Months"], errors="coerce").fillna(1).clip(lower=1)
            df["ROS"] = (qty / m).fillna(0)
        else:
            w = float(cfg.period_weeks) if cfg.period_weeks else 52.0
            df["ROS"] = qty / w

    # ------------------------------------------------------------------
    # GMROI Calculation
    # ------------------------------------------------------------------
    if "GMROI" not in df.columns:
        gm = pd.to_numeric(df.get("Gross_Margin", 0), errors="coerce").fillna(0)

        denom_col_candidates = [
            "Avg_Inventory_Cost",
            "Avg Inventory Cost",
            "AvgInventoryCost",
            "AvgInvCost",
            "AverageInventoryCost",
            "InventoryInvestment",
            "AvgInventoryValue",
            "Closing_Inv_Cost_Est",
            "ClosingStockValue",
        ]

        denom = None
        for c in denom_col_candidates:
            if c in df.columns:
                denom = pd.to_numeric(df[c], errors="coerce")
                break

        if denom is None:
            denom = pd.to_numeric(df.get("StoreStock", 0), errors="coerce")

        gm = pd.to_numeric(gm, errors="coerce").fillna(0)
        denom = pd.to_numeric(denom, errors="coerce").fillna(0)
        if (denom <= 0).all():
            logger.warning(
                "GMROI denominator has no positive values — all GMROI scores will be 0. "
                "Check that inventory cost columns contain valid data."
            )
        gmroi_raw = gm / denom.replace(0, np.nan)
        df["GMROI"] = pd.to_numeric(gmroi_raw, errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------
    # Add-tier mapping
    # ------------------------------------------------------------------
    df["Add_Tier_RP"] = df.get("Assigned_Tier", "UNASSIGNED").astype(str)

    # ------------------------------------------------------------------
    # Discount Processing and Price Power
    # ------------------------------------------------------------------
    if "DiscountPct" in df.columns:
        disc = pd.to_numeric(df["DiscountPct"], errors="coerce").fillna(0.0)
        # Auto-detect format: if 95th percentile > 1.5, data is in percentage (0-100) format
        # Otherwise, data is already in fractional (0-1) format
        p95 = float(disc.quantile(0.95))
        if p95 > 1.5:
            logger.info(f"DiscountPct appears in percentage format (p95={p95:.1f}), dividing by 100")
            disc_frac = disc / 100.0
        else:
            disc_frac = disc
        disc_frac = disc_frac.clip(lower=0.0, upper=1.0)
        df["DiscountFrac"] = disc_frac
        df["PricePower"] = (1.0 - disc_frac).astype(float)
    else:
        df["DiscountFrac"] = 0.0
        df["PricePower"] = 0.0

    # ------------------------------------------------------------------
    # DISCOUNT PENALTY (Phase 2.2 - NEW)
    # Premium/Ultra tier SKUs sold at heavy discount are penalized
    # ------------------------------------------------------------------
    discount_penalty_threshold = float(getattr(cfg, "discount_penalty_threshold", 0.25))
    discount_penalty_factor = float(getattr(cfg, "discount_penalty_factor", 0.20))
    
    high_tiers = {"PREMIUM", "ULTRA"}
    tier_upper = df["Assigned_Tier"].astype(str).str.upper().str.strip()
    is_high_tier = tier_upper.isin(high_tiers)
    is_heavy_discount = df["DiscountFrac"] > discount_penalty_threshold
    
    # Penalty: reduce score by penalty_factor for high-tier SKUs with heavy discount
    df["DiscountPenalty"] = np.where(
        is_high_tier & is_heavy_discount,
        discount_penalty_factor * df["DiscountFrac"],
        0.0
    )

    # ------------------------------------------------------------------
    # Weight Normalization
    # ------------------------------------------------------------------
    w_g = float(getattr(cfg, "score_weight_gmroi", 0.6))
    w_r = float(getattr(cfg, "score_weight_ros", 0.4))
    w_p = float(getattr(cfg, "score_weight_price_power", 0.0))
    w_sum = (w_g + w_r + w_p) if (w_g + w_r + w_p) > 0 else 1.0
    w_g, w_r, w_p = w_g / w_sum, w_r / w_sum, w_p / w_sum

    # ------------------------------------------------------------------
    # Score Calculation within SALES tier
    # ------------------------------------------------------------------
    # Compute within-tier percentile ranks
    df["GMROI_pctl_tier"] = df.groupby("Assigned_Tier")["GMROI"].transform(_pct_rank).fillna(0)
    df["ROS_pctl_tier"] = df.groupby("Assigned_Tier")["ROS"].transform(_pct_rank).fillna(0)
    # Compute global percentile ranks (for blending when tiers are sparse)
    df["GMROI_pctl_global"] = _pct_rank(df["GMROI"]).fillna(0)
    df["ROS_pctl_global"] = _pct_rank(df["ROS"]).fillna(0)
    # Blend: for tiers with <5 SKUs, mix 70% within-tier + 30% global to stabilize scores
    _MIN_TIER_SIZE_FOR_PURE_RANK = 5
    tier_sizes = df.groupby("Assigned_Tier")["GMROI"].transform("count")
    blend_w = np.where(tier_sizes < _MIN_TIER_SIZE_FOR_PURE_RANK, 0.70, 1.0)
    df["GMROI_pctl"] = blend_w * df["GMROI_pctl_tier"] + (1 - blend_w) * df["GMROI_pctl_global"]
    df["ROS_pctl"] = blend_w * df["ROS_pctl_tier"] + (1 - blend_w) * df["ROS_pctl_global"]
    if w_p > 0:
        df["PricePower_pctl"] = df.groupby("Assigned_Tier")["PricePower"].transform(_pct_rank).fillna(0)
    else:
        df["PricePower_pctl"] = 0.0
    
    # Base score before penalty
    base_score = w_g * df["GMROI_pctl"] + w_r * df["ROS_pctl"] + w_p * df["PricePower_pctl"]
    
    # Apply discount penalty
    df["Score"] = (base_score - df["DiscountPenalty"]).clip(lower=0.0)

    # Score_AddTier: Since Add_Tier_RP == Assigned_Tier (line 115), the
    # within-tier percentile ranks are identical.  Alias to avoid duplicate
    # computation (~6 groupby transforms saved per run).
    df["Score_AddTier"] = df["Score"]
    
    return df


def create_remove_list(sku_scored: pd.DataFrame, tier_targets: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    """
    Generate the Remove List for SKUs that should be removed from stores.
    
    REMOVAL LOGIC DOCUMENTATION (Audit Fix C3):
    -------------------------------------------
    1. Only Store×Tier combinations with negative Gap AND Eligibility=1 are considered
    2. Within each Store×Tier, SKUs are ranked by Score (ascending - lowest first)
    3. Only SKUs with stock in store (StoreStock > 0) are candidates for removal
    4. Protected SKUs (do_not_remove list) are never recommended for removal
    5. The number of removals = abs(Gap), capped by available candidates
    
    Transparency: The Reason field explains why each SKU was selected for removal.
    """
    gaps = tier_targets[["StoreID", "PriceTier", "Gap", "Eligibility"]].copy()
    gaps["StoreID"] = gaps["StoreID"].astype(str)
    gaps["PriceTier"] = gaps["PriceTier"].astype(str)

    df = sku_scored.copy()
    df["StoreID"] = df["StoreID"].astype(str)
    df["Assigned_Tier"] = df["Assigned_Tier"].astype(str)

    merged = df.merge(gaps, left_on=["StoreID", "Assigned_Tier"], right_on=["StoreID", "PriceTier"], how="left")
    merged["Gap"] = merged["Gap"].fillna(0)
    merged["Eligibility"] = merged["Eligibility"].fillna(0)

    protect = set(cfg.do_not_remove or [])
    out = []
    removal_stats = []  # Track removal statistics for transparency
    
    for (store, tier), g in merged.groupby(["StoreID", "Assigned_Tier"]):
        gap = int(g["Gap"].iloc[0])
        eligible = int(g["Eligibility"].iloc[0])
        
        if eligible != 1 or gap >= 0:
            continue
            
        need = abs(gap)

        # Only consider SKUs with stock in store
        stock_col = None
        if "ClosingStockQty" in g.columns:
            stock_col = "ClosingStockQty"
        elif "StoreStock" in g.columns:
            stock_col = "StoreStock"

        g_pool = g
        if stock_col is not None:
            stock = pd.to_numeric(g_pool[stock_col], errors="coerce").fillna(0)
            g_pool = g_pool[stock > 0].copy()

        # Respect Protected SKUs
        protected_count = g_pool["ItemColorName"].isin(protect).sum()
        g_pool = g_pool[~g_pool["ItemColorName"].isin(protect)].copy()

        available = len(g_pool)
        pick = g_pool.sort_values("Score").head(need).copy()
        actual_removed = len(pick)
        
        # Document the removal decision
        pick["Action"] = "REMOVE"
        pick["Reason"] = f"Lowest Score in Store×Tier (Gap={gap}, Available={available}, Protected={protected_count})"
        pick["Removal_Logic"] = "Score-based ranking within tier, stock>0 filter applied"
        
        removal_stats.append({
            "StoreID": store,
            "Tier": tier,
            "Gap": gap,
            "Needed": need,
            "Available": available,
            "Protected": protected_count,
            "Removed": actual_removed
        })
        
        out.append(pick)
    
    if out:
        result = pd.concat(out, ignore_index=True)
        # Add summary statistics as metadata
        result.attrs["removal_stats"] = pd.DataFrame(removal_stats)
        return result

    base_cols = [
        "StoreID", "ItemColorName", "Assigned_Tier", "Action", "Reason",
        "Removal_Logic", "Score", "Score_AddTier", "ROS", "GMROI", "NetworkStock",
    ]
    empty_df = pd.DataFrame(columns=base_cols)
    empty_df.attrs["removal_stats"] = pd.DataFrame()
    return empty_df


def create_peer_pool_adds(
    sku_scored: pd.DataFrame,
    stores_enriched: pd.DataFrame,
    tier_targets: pd.DataFrame,
    cfg: EngineConfig,
    articles_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Create add list using peer pools with DEDUPLICATION and EXPANDED POOL.
    
    AUDIT FIXES APPLIED:
    - Fix C1: Deduplicate Add_List to ensure each Store×SKU appears only once
    - Fix C2: Expand candidate pool with fallback to all efficient stores
    
    ADD LOGIC DOCUMENTATION:
    ------------------------
    1. For each Store×Tier with positive Gap, find candidate SKUs
    2. Primary source: Peer stores (based on Q-score clustering)
    3. Fallback source: All efficient stores (if peer pool insufficient)
    4. Secondary fallback: All stores in the universe
    5. Candidates are ranked by FinalScore (blend of Score_AddTier and StoreFitScore)
    6. DEDUPLICATION: Only the highest-scored recommendation per Store×SKU is kept
    """

    from .peers import build_peer_groups_from_articles
    from .store_fit import build_store_fit_tables, score_store_fit_for_candidates

    sid_col = "StoreID" if "StoreID" in stores_enriched.columns else "Row Labels"
    eff_ids = set(stores_enriched.loc[stores_enriched["Is_Efficient"] == 1, sid_col].astype(str))
    all_store_ids = set(stores_enriched[sid_col].astype(str))
    
    gaps = tier_targets[["StoreID", "PriceTier", "Gap", "Eligibility"]].copy()
    gaps["StoreID"] = gaps["StoreID"].astype(str)
    gaps["PriceTier"] = gaps["PriceTier"].astype(str)

    df = sku_scored.copy()
    df["StoreID"] = df["StoreID"].astype(str)
    df["Assigned_Tier"] = df["Assigned_Tier"].astype(str)

    # Build peer groups
    peer_map = {}
    store_clusters = None

    # Option A: 2D clustering using Store KPIs (Q_SCORE + SPF) for more realistic peers
    if cfg.use_peer_clusters and str(getattr(cfg, "peer_cluster_mode", "tier_behavior")).lower() == "qscore_spf":
        try:
            import numpy as np
            from sklearn.cluster import KMeans

            from .articles import _find_col_fuzzy
            s = stores_enriched.copy()
            # Fuzzy columns
            q_col = _find_col_fuzzy(s, ["Q_SCORE", "QScore", "Q Score", "Q-Score"])
            spf_col = _find_col_fuzzy(s, ["SPF", "SalesPerFoot", "Sales per Foot", "Sales/Foot"])

            if q_col and spf_col:
                x = s[[q_col, spf_col]].copy()
                x[q_col] = pd.to_numeric(x[q_col], errors="coerce")
                x[spf_col] = pd.to_numeric(x[spf_col], errors="coerce")
                x = x.dropna()

                n = len(x)
                if n >= 8:
                    k = int(getattr(cfg, "peer_cluster_k", 0) or 0)
                    if k <= 0:
                        k = int(round((n / 3) ** 0.5))
                    k = max(4, min(12, k))
                    k = min(k, max(2, n // 4))

                    # Standardize
                    mu = x[[q_col, spf_col]].mean()
                    sd = x[[q_col, spf_col]].std().replace(0, 1.0)
                    z = (x[[q_col, spf_col]] - mu) / sd

                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = km.fit_predict(z.values)

                    store_clusters = pd.DataFrame({
                        "StoreID": x.index.map(lambda i: s.loc[i, "StoreID"]).astype(str),
                        "PeerClusterID": labels.astype(int),
                    })

                    # Build peer_map
                    grp = store_clusters.groupby("PeerClusterID")["StoreID"].apply(list).to_dict()
                    peer_map = {sid: [p for p in grp.get(cid, []) if p != sid]
                                for sid, cid in zip(store_clusters["StoreID"], store_clusters["PeerClusterID"])}
        except Exception as e:
            logger.warning(f"Q-score/SPF clustering failed, falling back to article-based peers: {e}")
            peer_map = {}
            store_clusters = None

    # Option B (default): Article-based clustering (tier behavior)
    if (not peer_map) and cfg.use_peer_clusters and articles_raw is not None and not articles_raw.empty:
        try:
            pres = build_peer_groups_from_articles(articles_raw, cfg)
            peer_map = pres.peer_map
            store_clusters = pres.store_clusters
        except Exception as e:
            logger.warning(f"Article-based peer clustering failed: {e}")
            peer_map = {}
            store_clusters = None

    # Prepare candidate sources at different levels
    candidate_all = df.copy()
    candidate_efficient = df[df["StoreID"].isin(eff_ids)].copy()

    # Lifecycle filter — exclude statuses specified in config (fully dynamic)
    _lc_exclusions = cfg.lifecycle_exclusions or []
    if _lc_exclusions and "LifeCycle_Status" in candidate_all.columns:
        _ban = {s.strip().lower() for s in _lc_exclusions}
        lc = candidate_all["LifeCycle_Status"].astype(str).str.strip().str.lower()
        candidate_all = candidate_all[~lc.isin(_ban)].copy()

        lc_eff = candidate_efficient["LifeCycle_Status"].astype(str).str.strip().str.lower()
        candidate_efficient = candidate_efficient[~lc_eff.isin(_ban)].copy()

    # Do-not-add filter: exclude SKUs that should never be recommended for addition
    _do_not_add = set(cfg.do_not_add or [])
    if _do_not_add:
        candidate_all = candidate_all[~candidate_all["ItemColorName"].isin(_do_not_add)].copy()
        candidate_efficient = candidate_efficient[~candidate_efficient["ItemColorName"].isin(_do_not_add)].copy()

    # Network stock filter on add candidates: don't recommend SKUs with low network availability
    _min_add_ns = float(cfg.active_network_stock_threshold)
    if _min_add_ns > 0 and "NetworkStock" in candidate_all.columns:
        candidate_all = candidate_all[candidate_all["NetworkStock"] >= _min_add_ns].copy()
        candidate_efficient = candidate_efficient[candidate_efficient["NetworkStock"] >= _min_add_ns].copy()

    # Precompute tier rankings
    for cand_df in [candidate_all, candidate_efficient]:
        if not cand_df.empty:
            cand_df.sort_values(["Add_Tier_RP", "Score_AddTier"], ascending=[True, False], inplace=True)
            cand_df["TierRank_Add"] = cand_df.groupby("Add_Tier_RP")["Score_AddTier"].rank(method="first", ascending=False)

    # Presence dictionaries (computed once via shared helper)
    present_any, present_stock = _build_presence_dicts(df, articles_raw)

    # Target store stock lookup (for Add validation and output columns)
    _stock_lookup = df.copy()
    _stock_lookup["StoreID"] = _stock_lookup["StoreID"].astype(str)
    _stock_lookup["ItemColorName"] = _stock_lookup["ItemColorName"].astype(str)
    _stock_lookup["StoreStock"] = pd.to_numeric(_stock_lookup["StoreStock"], errors="coerce").fillna(0)
    target_stock_map = _stock_lookup.set_index(["StoreID", "ItemColorName"])["StoreStock"].to_dict()

    # Tier order for closest-tier fallback (sourced from tiers.py canonical order)
    from .tiers import TIERS_ORDER
    _tier_order = [t.upper() for t in TIERS_ORDER]

    def _closest_tiers(tier: str) -> list[str]:
        t = normalize_tier_labels(pd.Series([tier])).iloc[0]
        t = str(t).upper().strip()
        if t not in _tier_order:
            return []
        i = _tier_order.index(t)
        candidates = []
        for step in range(1, int(getattr(cfg, "closest_tier_max_steps", 2)) + 1):
            left = i - step
            right = i + step
            if left >= 0:
                candidates.append(_tier_order[left])
            if right < len(_tier_order):
                candidates.append(_tier_order[right])
        out = []
        seen = set()
        for x in candidates:
            if x == t:
                continue
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    # Store-fit tables
    fit_tables = None
    if cfg.use_store_fit and peer_map:
        try:
            fit_tables = build_store_fit_tables(df, peer_map, cfg)
        except Exception as e:
            logger.warning(f"Store-fit table construction failed: {e}")
            fit_tables = None

    out = []
    add_stats = []  # Track add statistics for transparency
    
    for (store, tier), g in gaps.groupby(["StoreID", "PriceTier"]):
        gap = int(g["Gap"].iloc[0])
        eligible = int(g["Eligibility"].iloc[0])
        if eligible != 1 or gap <= 0:
            continue

        # Inventory Health Guardrail (Insights): block adds if the tier is already over-stocked
        if bool(getattr(cfg, "enable_inventory_prudence_filter", True)):
            try:
                cover_thr = float(getattr(cfg, "stock_cover_threshold_months", 6.0))
                # Prefer tier-level signals present in tier_targets
                row0 = g.iloc[0]
                closing = float(row0.get("ClosingStockQty", np.nan))
                ros_per_sku_m = float(row0.get("Tier_ROS_perSKU_perMonth", np.nan))
                cur_skus_t = float(row0.get("Current_SKUs_Tier", np.nan))
                if not np.isnan(closing) and not np.isnan(ros_per_sku_m) and not np.isnan(cur_skus_t) and cur_skus_t > 0 and ros_per_sku_m > 0:
                    tier_monthly_units = ros_per_sku_m * cur_skus_t
                    months_cover = closing / max(tier_monthly_units, 1e-6)
                    if months_cover > cover_thr:
                        # Skip adds for this store-tier due to over-stock
                        continue
            except Exception as e:
                logger.debug(f"Inventory prudence check skipped for store={store}, tier={tier}: {e}")

        target_tier = normalize_tier_labels(pd.Series([tier])).iloc[0]
        store_present = present_stock.get(store, set())
        
        # EXPANDED CANDIDATE POOL LOGIC (Audit Fix C2 — reordered)
        # Priority: Store similarity > Tier proximity > Pool breadth
        #   Level 1: Peer stores, exact tier
        #   Level 2: Peer stores, closest tier (if enabled)
        #   Level 3: All efficient stores, exact tier
        #   Level 4: All stores, exact tier (last resort)
        peers = peer_map.get(str(store), []) if peer_map else []
        cand = pd.DataFrame()
        source_level = "None"
        tt_upper = str(target_tier).upper()

        # ── Level 1: Peer stores, exact tier ──
        if peers:
            cand = candidate_efficient[candidate_efficient["StoreID"].isin(peers)].copy()
            cand = cand[cand["Add_Tier_RP"].astype(str).str.upper() == tt_upper]
            cand = cand[~cand["ItemColorName"].isin(store_present)]
            if not cand.empty:
                source_level = f"Peer Pool (n={len(peers)})"

        # ── Level 2: Closest tier from peers first, then efficient stores ──
        if (cand.empty or len(cand) < gap) and getattr(cfg, "enable_closest_tier_fallback", True):
            for alt in _closest_tiers(str(target_tier)):
                alt_upper = str(alt).upper()
                # Try peers first for the adjacent tier
                if peers:
                    cand_alt = candidate_efficient[candidate_efficient["StoreID"].isin(peers)].copy()
                    cand_alt = cand_alt[cand_alt["Add_Tier_RP"].astype(str).str.upper() == alt_upper]
                    cand_alt = cand_alt[~cand_alt["ItemColorName"].isin(store_present)]
                    if not cand_alt.empty:
                        if cand.empty:
                            cand = cand_alt
                            source_level = f"Peer Closest Tier ({target_tier}->{alt})"
                        else:
                            cand = pd.concat([cand, cand_alt]).drop_duplicates(subset=["ItemColorName"], keep="first")
                            source_level = f"Peer + Closest Tier Fallback"
                        break
                # Then try all efficient stores for the adjacent tier
                cand_alt = candidate_efficient.copy()
                cand_alt = cand_alt[cand_alt["Add_Tier_RP"].astype(str).str.upper() == alt_upper]
                cand_alt = cand_alt[~cand_alt["ItemColorName"].isin(store_present)]
                cand_alt = cand_alt[cand_alt["StoreID"] != store]
                if not cand_alt.empty:
                    if cand.empty:
                        cand = cand_alt
                        source_level = f"Efficient Closest Tier ({target_tier}->{alt})"
                    else:
                        cand = pd.concat([cand, cand_alt]).drop_duplicates(subset=["ItemColorName"], keep="first")
                        source_level = f"Multi-level Closest Tier Fallback"
                    break

        # ── Level 3: All efficient stores, exact tier ──
        if cand.empty or len(cand) < gap:
            cand_eff = candidate_efficient.copy()
            cand_eff = cand_eff[cand_eff["Add_Tier_RP"].astype(str).str.upper() == tt_upper]
            cand_eff = cand_eff[~cand_eff["ItemColorName"].isin(store_present)]
            cand_eff = cand_eff[cand_eff["StoreID"] != store]
            if not cand_eff.empty:
                if cand.empty:
                    cand = cand_eff
                    source_level = f"All Efficient Stores (n={len(eff_ids)})"
                else:
                    cand = pd.concat([cand, cand_eff]).drop_duplicates(subset=["ItemColorName"], keep="first")
                    source_level = f"Peer + Efficient Fallback"

        # ── Level 4: All stores, exact tier (last resort) ──
        if cand.empty or len(cand) < gap:
            cand_all = candidate_all.copy()
            cand_all = cand_all[cand_all["Add_Tier_RP"].astype(str).str.upper() == tt_upper]
            cand_all = cand_all[~cand_all["ItemColorName"].isin(store_present)]
            cand_all = cand_all[cand_all["StoreID"] != store]
            if not cand_all.empty:
                if cand.empty:
                    cand = cand_all
                    source_level = f"All Stores Fallback (n={len(all_store_ids)})"
                else:
                    cand = pd.concat([cand, cand_all]).drop_duplicates(subset=["ItemColorName"], keep="first")
                    source_level = f"Multi-level Fallback"

        if cand.empty:
            add_stats.append({
                "StoreID": store, "Tier": tier, "Gap": gap,
                "Candidates": 0, "Added": 0, "Source": "No candidates found"
            })
            continue

        # Prepare candidates
        cand["StoreID_Target"] = str(store)
        # Exclude self-store as a source (source and target must not be same)
        cand = cand[cand["StoreID"].astype(str) != str(store)]
        cand["Action"] = "ADD"
        cand["Reason"] = source_level

        # Attach store-fit if enabled
        if fit_tables is not None:
            cand = score_store_fit_for_candidates(cand, fit_tables, cfg)
            cand["FinalScore"] = (1 - cfg.store_fit_weight) * cand["Score_AddTier"] + cfg.store_fit_weight * cand["StoreFitScore"]
        else:
            cand["FinalScore"] = cand["Score_AddTier"]
            cand["StoreFitScore"] = 0.0
            cand["Confidence"] = "NA"
            cand["SimilarCountInStore"] = 0

        cand = cand.sort_values("FinalScore", ascending=False)
        # Ensure unique SKUs per target store-tier (same SKU can exist in many peer stores)
        cand = cand.drop_duplicates(subset=["ItemColorName"], keep="first")
        pick = cand.head(gap).copy()
        # Output should be target-store centric
        pick["StoreID_Target"] = str(store)
        pick["SourceStoreID"] = pick["StoreID"]
        # Stock columns for clarity
        pick["SourceStoreStock"] = pd.to_numeric(pick.get("StoreStock", 0), errors="coerce").fillna(0)
        pick["TargetStoreStock"] = pick["ItemColorName"].astype(str).map(lambda sku: float(target_stock_map.get((str(store), sku), 0)))
        # Hard safety: never recommend adding an SKU already available in the target store
        pick = pick[pick["TargetStoreStock"] <= 0].copy()
        # Keep StoreStock as SourceStoreStock for backward compatibility
        pick["StoreStock"] = pick["SourceStoreStock"]
        pick["StoreID"] = str(store)
        # Hard safety: source and target must not be the same
        pick = pick[pick["SourceStoreID"].astype(str) != pick["StoreID"].astype(str)].copy()
        pick["PriceTier"] = str(target_tier).upper()

        # Human-readable rationale
        def _rationale(r: pd.Series) -> str:
            reason = str(r.get("Reason", source_level)).strip()
            src_store = str(r.get("SourceStoreID", r.get("StoreID", "NA")))
            fs = f"{float(r.get('FinalScore', 0)):.3f}"
            gm_p = f"{float(r.get('GMROI_pctl_add', 0)):.2f}"
            ros_p = f"{float(r.get('ROS_pctl_add', 0)):.2f}"
            dp = f"{float(r.get('DiscountPenalty', 0)):.2f}"
            return f"Gap:+{gap} | Source:{reason} | FinalScore:{fs} | GMROI_pctl:{gm_p} | ROS_pctl:{ros_p} | DiscountPenalty:{dp} | From:{src_store}"

        pick["Add_Rationale_Text"] = pick.apply(_rationale, axis=1)
        
        add_stats.append({
            "StoreID": store, "Tier": tier, "Gap": gap,
            "Candidates": len(cand), "Added": len(pick), "Source": source_level
        })
        
        out.append(pick)

    if not out:
        base_cols = [
            "StoreID_Target", "StoreID", "Assigned_Tier", "Add_Tier_RP",
            "ItemColorName", "Action", "Reason", "Add_Rationale_Text",
            "Score_AddTier", "StoreFitScore", "FinalScore", "NetworkStock",
            "ROS", "GMROI", "DiscountPenalty", "PriceTier",
        ]
        return pd.DataFrame(columns=base_cols)

    adds = pd.concat(out, ignore_index=True)
    
    # ------------------------------------------------------------------
    # DEDUPLICATION (Audit Fix C1)
    # Keep only the highest-scored recommendation per Store×SKU
    # ------------------------------------------------------------------
    before_dedup = len(adds)
    adds = adds.sort_values("FinalScore", ascending=False)
    adds = adds.drop_duplicates(subset=["StoreID_Target", "ItemColorName"], keep="first")
    after_dedup = len(adds)
    
    # Add deduplication metadata
    adds.attrs["dedup_stats"] = {
        "before": before_dedup,
        "after": after_dedup,
        "duplicates_removed": before_dedup - after_dedup
    }
    adds.attrs["add_stats"] = pd.DataFrame(add_stats)

    # Handle force_add
    force = set(cfg.force_add or [])
    if force:
        forced = pd.DataFrame([
            {
                "StoreID_Target": "ALL",
                "Assigned_Tier": "FORCE",
                "ItemColorName": x,
                "Action": "ADD",
                "Reason": "ForceAdd",
                "Score": 1.0,
                "FinalScore": 1.0,
                "Add_Rationale_Text": "ForceAdd | Merchant override",
            }
            for x in force
        ])
        adds = pd.concat([adds, forced], ignore_index=True)

    # Final safety: ensure no duplicates per TargetStore × SKU
    if not adds.empty and "StoreID_Target" in adds.columns:
        adds = adds.drop_duplicates(subset=["StoreID_Target", "ItemColorName"], keep="first")

    return adds


def flag_circular_recommendations(
    remove_list: pd.DataFrame,
    add_list: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect and flag circular recommendations where a SKU is being removed from
    Store A while simultaneously being recommended for addition to Store B
    sourced from Store A.

    Returns updated (remove_list, add_list) with a 'CircularFlag' column.
    """
    if remove_list is None or remove_list.empty or add_list is None or add_list.empty:
        if remove_list is not None and not remove_list.empty:
            remove_list["CircularFlag"] = ""
        if add_list is not None and not add_list.empty:
            add_list["CircularFlag"] = ""
        return remove_list, add_list

    # Build set of (SourceStoreID, ItemColorName) being removed
    removals = set()
    for _, r in remove_list.iterrows():
        removals.add((str(r.get("StoreID", "")), str(r.get("ItemColorName", ""))))

    # Flag adds whose source store is also removing that same SKU
    flags = []
    for _, r in add_list.iterrows():
        source = str(r.get("SourceStoreID", ""))
        sku = str(r.get("ItemColorName", ""))
        if (source, sku) in removals:
            flags.append("CIRCULAR: source store removing this SKU")
        else:
            flags.append("")

    add_list = add_list.copy()
    add_list["CircularFlag"] = flags
    remove_list = remove_list.copy()
    remove_list["CircularFlag"] = ""

    circular_count = sum(1 for f in flags if f)
    if circular_count > 0:
        logger.warning(
            f"Detected {circular_count} circular recommendations "
            f"(add from a store that is also removing the same SKU)"
        )

    return remove_list, add_list


def apply_active_filter(action_df: pd.DataFrame, network_stock: pd.DataFrame, cfg: EngineConfig, sku_col: str = "ItemColorName") -> pd.DataFrame:
    """Apply network stock filter to ensure only active SKUs are in recommendations."""
    if action_df is None or action_df.empty:
        return action_df
    net_map = network_stock.set_index("ItemColorName")["NetworkStock"].to_dict()
    df = action_df.copy()
    df["NetworkStock"] = df[sku_col].map(lambda x: float(net_map.get(x, 0)))
    return df[df["NetworkStock"] > cfg.active_network_stock_threshold].copy()


# ==============================================================================
# INTER-STORE TRANSFER RECOMMENDATIONS (Phase 2.3 - NEW FEATURE)
# ==============================================================================

def create_transfer_recommendations(
    sku_scored: pd.DataFrame,
    tier_targets: pd.DataFrame,
    cfg: EngineConfig,
    articles_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate inter-store transfer recommendations.
    
    This feature identifies opportunities to transfer excess inventory from
    stores with surplus SKUs (negative gap) to stores that need them (positive gap).
    
    TRANSFER LOGIC:
    ---------------
    1. Identify stores with surplus (Gap < 0) and stores with deficit (Gap > 0)
    2. For each surplus Store×Tier, find matching deficit Store×Tier
    3. Match SKUs that are:
       - In surplus store's remove candidates (low score in that store)
       - NOT already present in deficit store
       - Have sufficient stock for transfer
    4. Rank transfers by potential GMROI improvement
    
    Returns:
        DataFrame with columns:
        - FromStore, ToStore, ItemColorName, Tier, TransferQty
        - FromScore, ToExpectedScore, GMROILift
        - TransferRationale
    """
    if not getattr(cfg, "enable_transfers", False):
        return pd.DataFrame()
    
    gaps = tier_targets[["StoreID", "PriceTier", "Gap", "Eligibility"]].copy()
    gaps["StoreID"] = gaps["StoreID"].astype(str)
    gaps["PriceTier"] = gaps["PriceTier"].astype(str)
    
    df = sku_scored.copy()
    df["StoreID"] = df["StoreID"].astype(str)
    df["Assigned_Tier"] = df["Assigned_Tier"].astype(str)
    
    # Identify surplus and deficit store-tiers
    surplus = gaps[gaps["Gap"] < 0].copy()
    deficit = gaps[gaps["Gap"] > 0].copy()
    
    if surplus.empty or deficit.empty:
        return pd.DataFrame()
    
    # Presence dictionaries (computed once via shared helper)
    present_any, present_stock = _build_presence_dicts(df, articles_raw)
    present = present_any  # Backward-compatible alias used by transfer logic

    
    # Get stock by store-SKU
    stock_map = df.set_index(["StoreID", "ItemColorName"])["StoreStock"].to_dict()
    score_map = df.set_index(["StoreID", "ItemColorName"])["Score"].to_dict()
    gmroi_map = df.set_index(["StoreID", "ItemColorName"])["GMROI"].to_dict()
    
    # Pre-compute tier average GMROI once (avoids recalculating inside the loop)
    _tier_avg_gmroi = (
        df.assign(_tier_upper=df["Assigned_Tier"].str.upper())
        .groupby("_tier_upper")["GMROI"]
        .mean()
        .to_dict()
    )

    min_transfer_stock = int(getattr(cfg, "min_transfer_stock", 2))
    max_transfer_qty = int(getattr(cfg, "max_transfer_qty_per_sku", 5))

    transfers = []

    for _, surplus_row in surplus.iterrows():
        from_store = surplus_row["StoreID"]
        tier = surplus_row["PriceTier"]
        surplus_qty = abs(int(surplus_row["Gap"]))

        # Get low-scoring SKUs in this store-tier
        store_tier_skus = df[
            (df["StoreID"] == from_store) &
            (df["Assigned_Tier"].str.upper() == tier.upper())
        ].copy()

        if store_tier_skus.empty:
            continue

        # Sort by score (lowest first - these are transfer candidates)
        store_tier_skus = store_tier_skus.sort_values("Score").head(surplus_qty)

        # Find matching deficit stores for this tier
        tier_deficit = deficit[deficit["PriceTier"].str.upper() == tier.upper()]
        tier_avg_gmroi = _tier_avg_gmroi.get(tier.upper(), 0.0)

        for _, deficit_row in tier_deficit.iterrows():
            to_store = deficit_row["StoreID"]
            if to_store == from_store:
                continue

            deficit_qty = int(deficit_row["Gap"])
            to_present = present.get(to_store, set())

            # Find transferable SKUs (not already in destination)
            transferable = store_tier_skus[~store_tier_skus["ItemColorName"].isin(to_present)]

            for _, sku_row in transferable.head(deficit_qty).iterrows():
                sku = sku_row["ItemColorName"]
                from_stock = float(stock_map.get((from_store, sku), 0))

                if from_stock < min_transfer_stock:
                    continue

                transfer_qty = min(int(from_stock / 2), max_transfer_qty)
                from_score = float(score_map.get((from_store, sku), 0))
                from_gmroi = float(gmroi_map.get((from_store, sku), 0))
                gmroi_lift = tier_avg_gmroi - from_gmroi if from_gmroi > 0 else 0
                
                transfers.append({
                    "FromStore": from_store,
                    "ToStore": to_store,
                    "ItemColorName": sku,
                    "PriceTier": tier.upper(),
                    "TransferQty": transfer_qty,
                    "FromStock": from_stock,
                    "FromScore": from_score,
                    "FromGMROI": from_gmroi,
                    "TierAvgGMROI": tier_avg_gmroi,
                    "ExpectedGMROILift": gmroi_lift,
                    "TransferRationale": f"Low performer in {from_store} (Score={from_score:.2f}), needed in {to_store} (Gap=+{deficit_qty})"
                })
    
    if not transfers:
        return pd.DataFrame()
    
    result = pd.DataFrame(transfers)
    
    # Sort by expected GMROI lift
    result = result.sort_values("ExpectedGMROILift", ascending=False)
    
    # Limit to top N transfers
    max_transfers = int(getattr(cfg, "max_transfer_recommendations", 100))
    result = result.head(max_transfers)
    
    return result
