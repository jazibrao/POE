"""
Portfolio Optimization Engine — Production Sanity Test Suite
=============================================================

These tests validate that the engine produces commercially reasonable outputs.
They are NOT unit tests of internal functions — they are end-to-end sanity
checks that catch regressions before they reach production.

Run with:  python -m pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure engine is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.config import EngineConfig
from engine.io import load_workbook
from engine.pipeline import run_optimization_pipeline
from engine.decisions import attach_scores, build_network_stock, flag_circular_recommendations
from engine.articles import aggregate_articles_assign_tier
from engine.tiers import TIERS_ORDER, normalize_tier_labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data.xlsx")
SKIP_NO_SAMPLE = not os.path.exists(SAMPLE_DATA_PATH)


@pytest.fixture(scope="session")
def sample_data():
    """Load sample data once for all tests."""
    stores, tiers, articles = load_workbook(SAMPLE_DATA_PATH)
    return stores, tiers, articles


@pytest.fixture(scope="session")
def pipeline_result(sample_data):
    """Run full pipeline once for all tests."""
    stores, tiers, articles = sample_data
    cfg = EngineConfig()
    return run_optimization_pipeline(stores, tiers, articles, cfg)


# ---------------------------------------------------------------------------
# 1. Data Loading Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_NO_SAMPLE, reason="sample_data.xlsx not found")
class TestDataLoading:

    def test_loads_three_dataframes(self, sample_data):
        stores, tiers, articles = sample_data
        assert isinstance(stores, pd.DataFrame)
        assert isinstance(tiers, pd.DataFrame)
        assert isinstance(articles, pd.DataFrame)

    def test_stores_has_required_columns(self, sample_data):
        stores, _, _ = sample_data
        required = ["StoreID", "GMROI", "SPF", "Q_SCORE"]
        for col in required:
            assert col in stores.columns, f"Missing column: {col}"

    def test_tiers_has_required_columns(self, sample_data):
        _, tiers, _ = sample_data
        required = ["StoreID", "PriceTier"]
        for col in required:
            assert col in tiers.columns, f"Missing column: {col}"

    def test_articles_has_required_columns(self, sample_data):
        _, _, articles = sample_data
        required = ["StoreID", "ItemColorName", "PriceTier"]
        for col in required:
            assert col in articles.columns, f"Missing column: {col}"

    def test_storeid_consistent_type(self, sample_data):
        """StoreID must be string across all sheets (engine join safety)."""
        stores, tiers, articles = sample_data
        assert stores["StoreID"].dtype == object
        assert tiers["StoreID"].dtype == object
        assert articles["StoreID"].dtype == object

    def test_no_empty_dataframes(self, sample_data):
        stores, tiers, articles = sample_data
        assert len(stores) > 0, "Stores DataFrame is empty"
        assert len(tiers) > 0, "Tiers DataFrame is empty"
        assert len(articles) > 0, "Articles DataFrame is empty"


# ---------------------------------------------------------------------------
# 2. Pipeline Output Shape Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_NO_SAMPLE, reason="sample_data.xlsx not found")
class TestPipelineOutputs:

    def test_pipeline_returns_all_keys(self, pipeline_result):
        expected_keys = {
            "stores_enriched", "model_info", "tier_targets", "k_summary",
            "sku_master", "sku_scored", "network_stock",
            "remove_list", "add_list", "transfer_recommendations", "shortfall",
        }
        assert expected_keys.issubset(set(pipeline_result.keys()))

    def test_tier_targets_not_empty(self, pipeline_result):
        assert len(pipeline_result["tier_targets"]) > 0

    def test_sku_scored_has_score_column(self, pipeline_result):
        sku = pipeline_result["sku_scored"]
        assert "Score" in sku.columns
        assert "Score_AddTier" in sku.columns

    def test_score_equals_score_addtier(self, pipeline_result):
        """Score and Score_AddTier must be identical (unified computation)."""
        sku = pipeline_result["sku_scored"]
        assert (sku["Score"] == sku["Score_AddTier"]).all()


# ---------------------------------------------------------------------------
# 3. Commercial Sanity Checks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_NO_SAMPLE, reason="sample_data.xlsx not found")
class TestCommercialSanity:

    def test_no_negative_targets(self, pipeline_result):
        """Target SKU counts must never be negative."""
        tt = pipeline_result["tier_targets"]
        if "Target_SKUs" in tt.columns:
            targets = pd.to_numeric(tt["Target_SKUs"], errors="coerce").fillna(0)
            assert (targets >= 0).all(), f"Negative targets found:\n{tt[targets < 0]}"

    def test_scores_bounded_0_1(self, pipeline_result):
        """SKU scores must be in [0, 1] range."""
        sku = pipeline_result["sku_scored"]
        assert sku["Score"].min() >= -0.01, f"Score below 0: {sku['Score'].min()}"
        assert sku["Score"].max() <= 1.01, f"Score above 1: {sku['Score'].max()}"

    def test_remove_list_has_required_columns(self, pipeline_result):
        rl = pipeline_result["remove_list"]
        if len(rl) > 0:
            for col in ["StoreID", "ItemColorName", "Score"]:
                assert col in rl.columns, f"Remove list missing column: {col}"

    def test_add_list_has_required_columns(self, pipeline_result):
        al = pipeline_result["add_list"]
        if len(al) > 0:
            for col in ["ItemColorName", "StoreID_Target"]:
                assert col in al.columns, f"Add list missing column: {col}"

    def test_no_duplicate_adds_per_store(self, pipeline_result):
        """Same SKU must not appear twice as an Add for the same target store."""
        al = pipeline_result["add_list"]
        if len(al) > 0:
            dupes = al.duplicated(subset=["StoreID_Target", "ItemColorName"], keep=False)
            assert not dupes.any(), f"Duplicate adds found:\n{al[dupes]}"

    def test_removes_plus_adds_reasonable(self, pipeline_result, sample_data):
        """Net change shouldn't exceed total Store×SKU assignments (basic reasonableness)."""
        _, _, articles = sample_data
        total_assignments = len(articles)  # Store×SKU rows, not unique SKUs
        n_removes = len(pipeline_result["remove_list"])
        n_adds = len(pipeline_result["add_list"])
        assert n_removes <= total_assignments, \
            f"More removes ({n_removes}) than total Store×SKU assignments ({total_assignments})"
        assert n_adds <= total_assignments, \
            f"Unreasonable add count: {n_adds} (total assignments: {total_assignments})"

    def test_transfers_have_positive_qty(self, pipeline_result):
        """Transfer quantities must be positive."""
        tr = pipeline_result["transfer_recommendations"]
        if len(tr) > 0:
            assert (tr["TransferQty"] > 0).all(), "Transfer with zero/negative quantity found"

    def test_transfers_not_self_referencing(self, pipeline_result):
        """A store must not transfer to itself."""
        tr = pipeline_result["transfer_recommendations"]
        if len(tr) > 0:
            self_ref = tr["FromStore"] == tr["ToStore"]
            assert not self_ref.any(), f"Self-referencing transfers:\n{tr[self_ref]}"


# ---------------------------------------------------------------------------
# 4. Governance Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_NO_SAMPLE, reason="sample_data.xlsx not found")
class TestGovernance:

    def test_do_not_remove_respected(self, sample_data):
        """Protected SKUs must never appear in the remove list."""
        stores, tiers, articles = sample_data
        cfg = EngineConfig()
        # Pick a random SKU to protect
        some_sku = articles["ItemColorName"].iloc[0]
        cfg.do_not_remove = [some_sku]
        result = run_optimization_pipeline(stores, tiers, articles, cfg)
        rl = result["remove_list"]
        if len(rl) > 0:
            assert some_sku not in rl["ItemColorName"].values, \
                f"Protected SKU '{some_sku}' found in remove list"

    def test_do_not_add_respected(self, sample_data):
        """Blocked SKUs must never appear in the add list."""
        stores, tiers, articles = sample_data
        cfg = EngineConfig()
        some_sku = articles["ItemColorName"].iloc[0]
        cfg.do_not_add = [some_sku]
        result = run_optimization_pipeline(stores, tiers, articles, cfg)
        al = result["add_list"]
        if len(al) > 0:
            assert some_sku not in al["ItemColorName"].values, \
                f"Blocked SKU '{some_sku}' found in add list"


# ---------------------------------------------------------------------------
# 5. Tier Consistency Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_NO_SAMPLE, reason="sample_data.xlsx not found")
class TestTierConsistency:

    def test_tier_labels_normalized(self, pipeline_result):
        """All tier labels in targets must be from the canonical set."""
        tt = pipeline_result["tier_targets"]
        valid = {t.upper() for t in TIERS_ORDER} | {t for t in TIERS_ORDER}
        actual = set(tt["PriceTier"].astype(str).str.strip().unique())
        unexpected = actual - valid - {"UNASSIGNED", "Unassigned", "nan", ""}
        assert len(unexpected) == 0, f"Unexpected tier labels: {unexpected}"

    def test_normalize_tier_labels_round_trip(self):
        """Tier normalization should handle common variants."""
        raw = pd.Series(["value", "MID", "Premium", "ULTRA", "prem", "val"])
        normed = normalize_tier_labels(raw)
        assert list(normed) == ["Value", "Mid", "Premium", "Ultra", "Premium", "Value"]


# ---------------------------------------------------------------------------
# 6. Config Defaults Test
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_config_creates(self):
        cfg = EngineConfig()
        assert cfg.efficient_top_pct == 0.30
        assert cfg.score_weight_gmroi == 0.40
        assert cfg.score_weight_ros == 0.60

    def test_weights_sum_to_one(self):
        cfg = EngineConfig()
        total = cfg.score_weight_gmroi + cfg.score_weight_ros
        assert abs(total - 1.0) < 0.01, f"GMROI + ROS weights = {total}, expected ~1.0"

    def test_alpha_range_valid(self):
        cfg = EngineConfig()
        assert cfg.alpha_min >= 0.0
        assert cfg.alpha_max <= 1.0
        assert cfg.simplified_alpha_high_gmroi <= cfg.simplified_alpha_mid_gmroi
        assert cfg.simplified_alpha_mid_gmroi <= cfg.simplified_alpha_low_gmroi


# ---------------------------------------------------------------------------
# 7. Circular Recommendation Detection
# ---------------------------------------------------------------------------

class TestCircularDetection:

    def test_flags_circular_adds(self):
        """Circular detection should flag adds sourced from stores removing the same SKU."""
        remove_list = pd.DataFrame({
            "StoreID": ["S1", "S2"],
            "ItemColorName": ["SKU-A", "SKU-B"],
            "Score": [0.1, 0.2],
        })
        add_list = pd.DataFrame({
            "StoreID": ["S1"],  # Source store
            "StoreID_Target": ["S3"],
            "ItemColorName": ["SKU-B"],  # SKU-B is being removed from S2
            "PriceTier": ["Mid"],
            "Score": [0.5],
        })
        # SKU-B sourced from S1, but S2 is removing SKU-B — this is NOT circular
        # Circular would be: SKU-A sourced from S1, and S1 is removing SKU-A
        add_list2 = pd.DataFrame({
            "StoreID": ["S1"],
            "StoreID_Target": ["S3"],
            "ItemColorName": ["SKU-A"],  # S1 is removing SKU-A
            "PriceTier": ["Mid"],
            "Score": [0.5],
        })
        rl_out, al_out = flag_circular_recommendations(remove_list, add_list2)
        if "Circular_Flag" in al_out.columns:
            assert al_out["Circular_Flag"].any(), "Expected circular flag on SKU-A from S1"
