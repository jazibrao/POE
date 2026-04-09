"""
Portfolio Optimization Engine - Configuration Module
Version: 2.2.0 (Audit Fixes Applied)

New parameters added for:
- Discount penalty (Phase 2.2)
- Inter-store transfers (Phase 2.3)
- Simplified alpha blending (Phase 2.1)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class EngineConfig:
    # ------------------------------------------------------------
    # Excel-aligned defaults
    # ------------------------------------------------------------
    efficient_top_pct: float = 0.30
    # Efficiency definition: blend of GMROI and ROS percentile ranks (geometric mean)
    # These weights control how much each metric contributes to "efficient" store selection.
    # Set efficiency_ros_weight = 0.0 to revert to legacy GMROI-only definition.
    efficiency_gmroi_weight: float = 0.50
    efficiency_ros_weight: float = 0.50
    robust_p_low: float = 5.0
    robust_p_high: float = 95.0
    auto_select_gate_weight: bool = False
    fixed_gate_weight_spf: float = 0.70
    fixed_gate_weight_q: float = 0.30

    # Alpha range for tier blending
    alpha_min: float = 0.00
    alpha_max: float = 1.00
    alpha_tvi_beta: float = 0.45
    active_network_stock_threshold: float = 50.0  # Active SKU rule: NetworkStock > 50

    # ------------------------------------------------------------
    # SIMPLIFIED ALPHA BLENDING (Phase 2.1 - Audit Fix)
    # ------------------------------------------------------------
    # Instead of complex continuous alpha calculation, use a 3-tier system:
    #   - High GMROI stores (top 33%): Low alpha (0.2) - keep current mix
    #   - Mid GMROI stores (middle 33%): Medium alpha (0.5) - balanced blend
    #   - Low GMROI stores (bottom 33%): High alpha (0.8) - move toward benchmark
    use_simplified_alpha: bool = True
    simplified_alpha_high_gmroi: float = 0.20  # Alpha for top performers
    simplified_alpha_mid_gmroi: float = 0.45   # Alpha for mid performers
    simplified_alpha_low_gmroi: float = 0.60   # Alpha for low performers (was 0.80 — too aggressive)
    simplified_alpha_high_threshold: float = 0.67  # Top 33% GMROI
    simplified_alpha_low_threshold: float = 0.33   # Bottom 33% GMROI
    # Per-tier mix change cap: max % a single tier's share can shift in one run
    max_tier_mix_shift: float = 0.30  # 30% max shift per tier per run

    # Period configuration for SKU-level scoring
    period_weeks: int = 52
    score_weight_gmroi: float = 0.40  # Updated to match user preference (40/60)
    score_weight_ros: float = 0.60    # Updated to match user preference (40/60)
    score_weight_price_power: float = 0.0  # Disabled by default, use discount penalty instead
    peer_pool_top_n_per_tier: int = 500

    # ------------------------------------------------------------
    # DISCOUNT PENALTY (Phase 2.2 - Audit Fix)
    # ------------------------------------------------------------
    # Premium/Ultra tier SKUs sold at heavy discount are penalized
    # This addresses the business requirement to not reward discounted premium items
    discount_penalty_threshold: float = 0.25  # Discount > 25% triggers penalty
    discount_penalty_factor: float = 0.20     # Penalty magnitude (0-1 scale)

    # ------------------------------------------------------------
    # v2 Intelligence Layer - Peer Clustering
    # ------------------------------------------------------------
    use_peer_clusters: bool = True
    peer_cluster_k: int = 0  # 0 => auto
    peer_top_k_within_cluster: int = 40
    peer_min_cluster_size: int = 5

    # Peer clustering mode
    # 'tier_behavior' uses article-level ASP/STR/Units signals (default).
    # 'qscore_spf' clusters stores using Q_SCORE and SPF from StoresKPIs (2D clustering).
    peer_cluster_mode: str = "tier_behavior"

    # Inventory health guardrails (Insights)
    enable_inventory_prudence_filter: bool = True
    stock_cover_threshold_months: float = 6.0  # Block adds if tier has >6 months stock cover

    # SKU decision categorization thresholds (Insights)
    repeat_score_threshold: float = 0.70
    watch_score_threshold: float = 0.40
    drop_discount_penalty_threshold: float = 0.05
    peers_use_efficient_only: bool = False  # Allow all stores for better coverage

    # Lifecycle statuses to exclude from Add candidate pools (case-insensitive)
    # Populated from input data or config file; empty list = no lifecycle filter
    lifecycle_exclusions: Optional[List[str]] = None

    # Closest-tier fallback for empty candidate pools
    enable_closest_tier_fallback: bool = True
    closest_tier_max_steps: int = 2  # Increased from 1 for better coverage

    # Dynamic tiering for ADD decisions
    use_dynamic_add_tier: bool = True
    sales_tier_p_value: float = 0.45
    sales_tier_p_mid: float = 0.80
    sales_tier_p_premium: float = 0.95
    add_tier_p_value: float = 0.45
    add_tier_p_mid: float = 0.80
    add_tier_p_premium: float = 0.95

    # Store-fit ranking
    use_store_fit: bool = True
    store_fit_weight: float = 0.40
    store_fit_conf_scale: float = 6.0
    
    # Eligibility and protection
    eligibility_matrix: Optional[Dict[str, Dict[str, int]]] = None
    do_not_remove: Optional[List[str]] = None
    do_not_add: Optional[List[str]] = None  # SKUs that should never be recommended for addition
    force_add: Optional[List[str]] = None
    # DEPRECATED: min_add_network_stock removed — use active_network_stock_threshold instead.
    # Keeping the field for backward compat with old config files (ignored by engine).
    min_add_network_stock: Optional[float] = None

    # ------------------------------------------------------------
    # GOVERNANCE CONTROLS (Enhanced transparency - Audit Fix C3)
    # ------------------------------------------------------------
    # Cap how much a store's TOTAL SKU target can change in one run
    max_store_change_pct: Optional[float] = 0.15  # Default 15% cap for safety
    # Small-store stability: stores below this SKU count get a tighter change cap
    # The effective cap becomes max(max_store_change_pct, small_store_max_change / current_total)
    # Example: a 5-SKU store with small_store_max_change=1 → effective cap = max(0.15, 1/5) = 0.20
    #          but the absolute change is still just ~1 SKU
    small_store_threshold: int = 20  # Stores with fewer than this many SKUs get extra stability
    small_store_max_change: int = 2  # Maximum absolute SKU count change for small stores
    
    # Document removal logic in outputs
    document_removal_logic: bool = True

    # ------------------------------------------------------------
    # INTER-STORE TRANSFERS (Phase 2.3 - New Feature)
    # ------------------------------------------------------------
    # Enable transfer recommendations between stores
    enable_transfers: bool = True
    max_transfer_recommendations: int = 100
    min_transfer_stock: int = 2  # Minimum stock to consider transfer
    max_transfer_qty_per_sku: int = 5  # Maximum units to transfer per SKU

    # ------------------------------------------------------------
    # Demand-weighted tier benchmarking
    # ------------------------------------------------------------
    enable_demand_weighted_benchmark: bool = True
    tvi_clamp_min: Optional[float] = 0.70
    tvi_clamp_max: Optional[float] = 1.30
    min_tier_sales_qty_for_tvi: int = 5
    asp_pressure_threshold: float = 1.00
    empty_tier_requires_matrix: bool = False  # Changed: allow stores to introduce new tiers by default

    # ------------------------------------------------------------
    # Backward-compatible sub-config aliases
    # ------------------------------------------------------------
    @property
    def model(self):
        return self

    @property
    def tier_split(self):
        return self

    @property
    def articles(self):
        return self

    @property
    def stock(self):
        return self

    @property
    def scores(self):
        return self

    @property
    def decisions(self):
        return self