from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import EngineConfig
from .io import load_workbook
from .store_model import build_total_target_model
from .tier_split import split_targets_across_tiers
from .articles import aggregate_articles_assign_tier
from .decisions import (
    build_network_stock,
    attach_scores,
    create_remove_list,
    create_peer_pool_adds,
    apply_active_filter,
)
from .exports import export_excel


REQUIRED_OUTPUTS = [
    "Model_Summary",
    "Final_Targets",
    "Tier_Targets",
    "Add_List",
    "Remove_List",
    "Peer_Pool_Top",
    "Store_SKU_Grid",
]


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Excel workbook")
    ap.add_argument("--out", default="smoke_outputs.xlsx", help="Output Excel path")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        return _fail(f"Input not found: {inp}")

    stores, tiers, articles = load_workbook(str(inp))

    cfg = EngineConfig()

    sm = build_total_target_model(stores, cfg)
    stores_enriched = sm["stores_enriched"]

    ts = split_targets_across_tiers(tiers, stores_enriched, cfg, articles_raw=articles)
    tier_targets = ts["tier_targets"]

    am = aggregate_articles_assign_tier(articles)
    sku_master = am["sku_master"]

    network_stock = build_network_stock(sku_master)

    sku_master_scoring = sku_master.merge(network_stock, on="ItemColorName", how="left").fillna({"NetworkStock": 0})
    sku_scored = attach_scores(sku_master_scoring, cfg)

    remove_list = create_remove_list(sku_scored, tier_targets, cfg)
    remove_list = apply_active_filter(remove_list, network_stock, cfg)

    add_list = create_peer_pool_adds(
        sku_scored,
        stores_enriched,
        tier_targets,
        cfg,
        articles_raw=articles,
    )
    add_list = apply_active_filter(add_list, network_stock, cfg, sku_col="ItemColorName")

    # Peer pool top
    eff_ids = set(stores_enriched.loc[stores_enriched["Is_Efficient"] == 1, ("StoreID" if "StoreID" in stores_enriched.columns else "Row Labels")].astype(str))
    peer_pool = sku_scored[sku_scored["StoreID"].astype(str).isin(eff_ids)].copy()
    if "Assigned_Tier" in peer_pool.columns and "Score" in peer_pool.columns:
        peer_pool = peer_pool.sort_values(["Assigned_Tier", "Score"], ascending=[True, False])

    # Store grid is built in app layer; here we only validate core sheets.

    outputs = {
        "Model_Summary": pd.DataFrame({"Metric": ["SMOKE_TEST"], "Value": ["PASS"]}),
        "Final_Targets": ts.get("final_targets", tier_targets),
        "Tier_Targets": tier_targets,
        "Add_List": add_list,
        "Remove_List": remove_list,
        "Peer_Pool_Top": peer_pool.head(500),
    }

    # Guardrail: active filter for add/remove
    if not add_list.empty and "NetworkStock" in add_list.columns:
        if (add_list["NetworkStock"] <= cfg.active_network_stock_threshold).any():
            return _fail("Add list contains inactive SKUs")
    if not remove_list.empty and "NetworkStock" in remove_list.columns:
        if (remove_list["NetworkStock"] <= cfg.active_network_stock_threshold).any():
            return _fail("Remove list contains inactive SKUs")

    # Guardrail: counts do not exceed gaps
    if not tier_targets.empty:
        if "Gap" in tier_targets.columns:
            g = tier_targets[["StoreID", "PriceTier", "Gap"]].copy()
            # Removes
            if not remove_list.empty:
                r = remove_list.groupby(["StoreID", "PriceTier"]).size().rename("RemoveCount").reset_index()
                m = g.merge(r, on=["StoreID", "PriceTier"], how="left").fillna({"RemoveCount": 0})
                bad = m[(m["Gap"] < 0) & (m["RemoveCount"] > (-m["Gap"]))]
                if not bad.empty:
                    return _fail("Remove count exceeds negative gap for some store-tier")
            # Adds
            if not add_list.empty:
                a = add_list.groupby(["StoreID_Target", "PriceTier"]).size().rename("AddCount").reset_index()
                a = a.rename(columns={"StoreID_Target": "StoreID"})
                m2 = g.merge(a, on=["StoreID", "PriceTier"], how="left").fillna({"AddCount": 0})
                bad2 = m2[(m2["Gap"] > 0) & (m2["AddCount"] > m2["Gap"])]
                if not bad2.empty:
                    return _fail("Add count exceeds positive gap for some store-tier")

    outp = Path(args.out)
    export_excel(str(outp), outputs)
    print("PASS: Smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
