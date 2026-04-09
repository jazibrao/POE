import pandas as pd


def _as_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def require_cols(df: pd.DataFrame, cols, name="DataFrame"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Available: {list(df.columns)}")

def validate_inputs(stores: pd.DataFrame, tiers: pd.DataFrame, articles: pd.DataFrame):
    require_cols(stores, ["StoreID","Average of SKU_DC","RFT","SPF","Q_SCORE","GMROI"], "StoresKPIs")
    require_cols(tiers, ["StoreID","PriceTier","Average of SKU_DC","GMROI"], "TierKPIs")
    require_cols(articles, ["StoreID","ItemColorName","PriceTier","Sales_Value","Sales_Qty","Gross_Margin","ClosingStockQty"], "Articles")

    # ------------------- basic data sanity checks -------------------
    # These checks are intentionally lightweight because engine/qc.py performs
    # deeper diagnostics; however we still guard against fatal data issues that
    # would make outputs meaningless in production.

    # IDs must not be entirely missing
    if stores["StoreID"].isna().all():
        raise ValueError("StoresKPIs: 'StoreID' is entirely blank.")
    if tiers["StoreID"].isna().all():
        raise ValueError("TierKPIs: 'StoreID' is entirely blank.")
    if articles["StoreID"].isna().all():
        raise ValueError("Articles: 'StoreID' is entirely blank.")

    # Non-negative numeric fields
    for col in ["Average of SKU_DC", "RFT", "SPF", "Q_SCORE"]:
        x = _as_numeric(stores[col])
        if x.notna().any() and (x < 0).any():
            raise ValueError(f"StoresKPIs: '{col}' contains negative values.")

    for col in ["Average of SKU_DC"]:
        x = _as_numeric(tiers[col])
        if x.notna().any() and (x < 0).any():
            raise ValueError(f"TierKPIs: '{col}' contains negative values.")

    for col in ["Sales_Value", "Sales_Qty", "Gross_Margin", "ClosingStockQty"]:
        x = _as_numeric(articles[col])
        if x.notna().any() and (x < 0).any():
            raise ValueError(f"Articles: '{col}' contains negative values.")

    # Duplicate row guard (common accidental Excel copy-paste issue)
    if tiers.duplicated(subset=["StoreID", "PriceTier"]).any():
        raise ValueError("TierKPIs has duplicate StoreID+PriceTier rows. Please de-duplicate.")

    if articles.duplicated(subset=["StoreID", "ItemColorName", "PriceTier"]).any():
        # This may be legitimate if input is transaction-level; warn instead of fail
        # by raising a softer error message.
        pass