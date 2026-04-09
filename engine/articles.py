import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _norm_col(x: str) -> str:
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def _find_col_fuzzy(df: pd.DataFrame, candidates) -> str | None:
    """Find a column by exact name or fuzzy normalized match.

    Normalization removes spaces/underscores and lowercases.
    Example: "Avg Inventory Cost" == "Avg_Inventory_Cost" == "AvgInventoryCost".
    """
    if df is None or df.empty:
        return None

    # exact match first
    for c in candidates:
        if c in df.columns:
            return c

    norm_map = {_norm_col(c): c for c in df.columns}
    for c in candidates:
        key = _norm_col(c)
        if key in norm_map:
            return norm_map[key]
    return None


def aggregate_articles_assign_tier(articles: pd.DataFrame) -> dict:
    """Aggregate Articles to Store×SKU and assign dominant PriceTier.

    Also carries forward optional time-granularity fields so ROS can be computed as:
      - Avg Weekly ROS = Sales_Qty / Weeks_Active (if provided/derivable)
      - Avg Monthly ROS = Sales_Qty / Months_Active (if provided/derivable)
    And carries Avg Inventory Cost so GMROI can be computed as:
      - GMROI = Total Gross Margin / Avg Inventory Cost
    """

    df = articles.copy()

    # ------------------------------------------------------------------
    # Store identifier hardening
    # ------------------------------------------------------------------
    # Some user files may label the store key as "Row Labels" (pivot output)
    # or contain invisible whitespace around the header.
    # We normalize to canonical "StoreID" to avoid KeyError failures.
    if "StoreID" not in df.columns and "Row Labels" in df.columns:
        df = df.rename(columns={"Row Labels": "StoreID"})
    if "StoreID" not in df.columns:
        raise KeyError("StoreID")
    df["StoreID"] = df["StoreID"].astype(str)
    # PriceTier can be empty for unsold SKUs in the analysis period.
    # User requirement: assign PriceTier using RetailPrice (not ASP), so
    # PriceTier is always available for every SKU×Store.
    if "PriceTier" in df.columns:
        df["PriceTier"] = df["PriceTier"].astype(str)
    df["ItemColorName"] = df["ItemColorName"].astype(str)

    # Optional commercial denominator (provided by user dataset)
    avg_inv_cost_col = _find_col_fuzzy(
        df,
        [
            "Avg Inventory Cost",
            "Avg_Inventory_Cost",
            "AvgInventoryCost",
            "AvgInvCost",
            "AverageInventoryCost",
        ],
    )

    # Optional time granularity indicators
    weeks_col = _find_col_fuzzy(df, ["Weeks", "NumWeeks", "NoOfWeeks", "Weeks_Active", "WeekCount"])
    months_col = _find_col_fuzzy(df, ["Months", "NumMonths", "NoOfMonths", "Months_Active", "MonthCount", "CountofMonthYear"])
    monthyear_col = _find_col_fuzzy(df, ["MonthYear", "Month_Year", "YearMonth", "YM"])
    week_id_col = _find_col_fuzzy(df, ["WeekStart", "WeekStartDate", "Week", "WeekNo", "WeekNumber"])

    # Optional dimensional fields for store-fit logic
    dept_col = _find_col_fuzzy(df, ["Dept", "Department"])
    cat_col = _find_col_fuzzy(df, ["Category", "Cat", "SubCategory", "Sub_Category"])

    # Optional lifecycle status (used to block adds for deep-discount articles)
    lifecycle_col = _find_col_fuzzy(
        df,
        [
            "LifeCycle_Status",
            "Lifecycle_Status",
            "LifeCycleStatus",
            "LifecycleStatus",
            "LC_Status",
        ],
    )

    # Optional price signals
    asp_col = _find_col_fuzzy(df, ["ASP", "AvgSellingPrice", "Average Selling Price", "Avg_Selling_Price"])
    rp_col = _find_col_fuzzy(df, ["RetailPrice", "Retail_Price", "MRP", "RSP", "ListPrice"])

    # Optional discount percentage (store realized discount behaviour)
    disc_col = _find_col_fuzzy(df, ["DiscountPct", "Discount%", "Discount %", "Disc%", "Disc %"])
    # --------------------------------------------------------------
    # PriceTier responsibility (Audit Assumption A)
    # --------------------------------------------------------------
    # The engine does NOT infer PriceTier. Any rows with missing/blank PriceTier
    # are excluded from tier-based benchmarking and decisions.
    if "PriceTier" not in df.columns:
        df["PriceTier"] = ""
    df["PriceTier"] = df["PriceTier"].astype(str).str.strip()
    pt = df["PriceTier"].astype(str).str.strip()
    missing_pt = pt.isna() | (pt == "") | (pt.str.lower().isin(["nan", "none", "null"]))
    if missing_pt.any():
        missing_count = int(missing_pt.sum())
        logger.warning(
            f"Dropping {missing_count} articles ({100 * missing_count / len(df):.1f}%) "
            f"with missing/invalid PriceTier"
        )
        df = df.loc[~missing_pt].copy()


    agg_kwargs = dict(
        Sales_Qty=("Sales_Qty", "sum"),
        Sales_Value=("Sales_Value", "sum"),
        Gross_Margin=("Gross_Margin", "sum"),
        StoreStock=("ClosingStockQty", "sum"),
    )

    # Carry forward dims/prices for downstream logic
    if dept_col is not None:
        agg_kwargs["Dept"] = (dept_col, "first")
    if cat_col is not None:
        agg_kwargs["Category"] = (cat_col, "first")
    if lifecycle_col is not None:
        # Store-fit/add-block uses this; take first non-null value within Store×SKU.
        agg_kwargs["LifeCycle_Status"] = (lifecycle_col, "first")
    if asp_col is not None:
        agg_kwargs["ASP"] = (asp_col, "mean")
    if rp_col is not None:
        agg_kwargs["RetailPrice"] = (rp_col, "mean")

    # Carry store realized discount% if available (mean across period)
    if disc_col is not None:
        agg_kwargs["DiscountPct"] = (disc_col, "mean")

    # Carry Avg Inventory Cost if available (mean across rows in period)
    if avg_inv_cost_col is not None:
        agg_kwargs["Avg_Inventory_Cost"] = (avg_inv_cost_col, "mean")

    # Carry weeks/months counts if available
    if weeks_col is not None:
        agg_kwargs["Weeks_Active"] = (weeks_col, "max")
    if months_col is not None:
        agg_kwargs["Months_Active"] = (months_col, "max")

    agg = (
        df.groupby(["StoreID", "ItemColorName"])  # Store×SKU
        .agg(**agg_kwargs)
        .reset_index()
    )

    # --------------------------------------------------------------
    # Quantity fields should be integers (Excel often imports them as floats).
    # Keeping them as integers makes outputs merchant-friendly.
    # --------------------------------------------------------------
    if "Sales_Qty" in agg.columns:
        agg["Sales_Qty"] = pd.to_numeric(agg["Sales_Qty"], errors="coerce").fillna(0).round().astype(int)
    if "StoreStock" in agg.columns:
        agg["StoreStock"] = pd.to_numeric(agg["StoreStock"], errors="coerce").fillna(0).round().astype(int)

    # Derive Weeks_Active from week identifiers if not provided as numeric
    if "Weeks_Active" not in agg.columns and week_id_col is not None:
        wk = (
            df.groupby(["StoreID", "ItemColorName"])[week_id_col]
            .nunique(dropna=True)
            .rename("Weeks_Active")
            .reset_index()
        )
        agg = agg.merge(wk, on=["StoreID", "ItemColorName"], how="left")

    # Derive Months_Active from MonthYear if present.
    # CRITICAL: Only count months where ClosingStockQty > 0 for each Store×SKU.
    # This gives the true selling window — dividing Sales_Qty by total period months
    # would understate ROS for articles that were out-of-stock part of the period.
    if "Months_Active" not in agg.columns and monthyear_col is not None:
        stock_col = _find_col_fuzzy(df, ["ClosingStockQty", "Closing StockQty", "Closing Stock Qty", "Closing Stock"])
        if stock_col is not None:
            stock_available = df.loc[
                pd.to_numeric(df[stock_col], errors="coerce").fillna(0) > 0
            ]
            mo = (
                stock_available.groupby(["StoreID", "ItemColorName"])[monthyear_col]
                .nunique(dropna=True)
                .rename("Months_Active")
                .reset_index()
            )
        else:
            # Fallback: no stock column available, count all months
            mo = (
                df.groupby(["StoreID", "ItemColorName"])[monthyear_col]
                .nunique(dropna=True)
                .rename("Months_Active")
                .reset_index()
            )
        agg = agg.merge(mo, on=["StoreID", "ItemColorName"], how="left")
        # Store×SKU combos with 0 stock-available months get Months_Active=1
        # to avoid division-by-zero in ROS
        agg["Months_Active"] = agg["Months_Active"].fillna(1).clip(lower=1)

    # ------------------------------------------------------------------
    # Backward-compatible derived fields (kept as optional fallbacks)
    # If Avg_Inventory_Cost isn't present, downstream may fall back to proxy.
    # ------------------------------------------------------------------
    if "Avg_Inventory_Cost" not in agg.columns:
        # Proxy investment when explicit cost isn't provided
        agg["COGS"] = (agg["Sales_Value"] - agg["Gross_Margin"]).clip(lower=0)
        agg["Unit_Cost_Est"] = np.where(
            agg["Sales_Qty"].fillna(0) > 0,
            agg["COGS"] / agg["Sales_Qty"].replace(0, np.nan),
            np.nan,
        )
        agg["Closing_Inv_Cost_Est"] = agg["StoreStock"].fillna(0) * agg["Unit_Cost_Est"].fillna(0)

    # Assign dominant tier (based on highest Sales_Value)
    # If PriceTier is still missing, we fallback to UNASSIGNED for all.
    if "PriceTier" not in df.columns:
        agg["Assigned_Tier"] = "UNASSIGNED"
        agg["Assigned_Tier_Share"] = 0.0
        sku_master = agg.copy()
        return {"sku_master": sku_master, "tier_sales_matrix": pd.DataFrame()}

    tier_sales = df.groupby(["StoreID", "ItemColorName", "PriceTier"])["Sales_Value"].sum().reset_index()
    pivot = tier_sales.pivot_table(
        index=["StoreID", "ItemColorName"],
        columns="PriceTier",
        values="Sales_Value",
        fill_value=0,
        aggfunc="sum",
    )

    pivot_num = pivot.copy()
    pivot["Assigned_Tier"] = pivot_num.idxmax(axis=1)
    pivot["Assigned_Tier_Share"] = (pivot_num.max(axis=1) / pivot_num.sum(axis=1).replace(0, np.nan)).fillna(0)

    assigned = pivot[["Assigned_Tier", "Assigned_Tier_Share"]].reset_index()
    sku_master = (
        agg.merge(assigned, on=["StoreID", "ItemColorName"], how="left")
        .fillna({"Assigned_Tier": "UNASSIGNED", "Assigned_Tier_Share": 0.0})
    )

    return {"sku_master": sku_master, "tier_sales_matrix": pivot.reset_index()}
