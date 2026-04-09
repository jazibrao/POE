import pandas as pd
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Column normalization (production hardening)
# ---------------------------------------------------------------------------
# Real-world files often contain slightly different column names compared to
# the canonical engine schema. We normalize common variants here so the engine
# remains robust across templates (e.g., Men2.xlsx).


def _norm_col_name(x: str) -> str:
    # Normalize whitespace & invisible separators (Excel sometimes exports NBSP)
    s = str(x).replace("\u00a0", " ").replace("\n", " ")
    return " ".join(s.strip().split())


def _rename_with_aliases(df: pd.DataFrame, aliases: Dict[str, list[str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Normalize whitespace for existing columns
    df = df.copy()
    df.columns = [_norm_col_name(c) for c in df.columns]
    col_l = {c.lower(): c for c in df.columns}
    rename = {}
    for canonical, alts in aliases.items():
        if canonical in df.columns:
            continue
        for a in [canonical] + list(alts):
            key = _norm_col_name(a).lower()
            if key in col_l:
                rename[col_l[key]] = canonical
                break
    if rename:
        df = df.rename(columns=rename)
    return df

DEFAULT_SHEET_MAP = {
    "stores": ["Input_StoreKPIs", "StoresKPIs", "StoreKPIs"],
    "tiers":  ["INPUT_TierKPIs", "TierKPIs", "Input_TierKPIs"],
    "articles": ["INPUT_Articles", "INPUT_ArticleKPIs", "ArticleKPIs", "Articles"]
}

# Canonical column aliases for robust ingestion.
ALIASES_STORES = {
    # Canonical store identifier inside the engine is **StoreID**.
    # We accept common input aliases like "Row Labels" but normalize them to StoreID.
    "StoreID": ["Row Labels", "Store Id", "Store", "Store Code"],
    "Average of SKU_DC": ["SKU_DC", "SKUs", "SKU DC", "SKU_DC (Design-Colors)"],
    "Q_SCORE": ["Q_Score", "Qscore", "Q-Score", "Q Score"],
    "GMROI": ["Gmroi", "GM ROI", "Gross Margin ROI"],
    "RFT": ["RFT_Final", "RFT Display Capacity"],
    "SPF": ["SPF", "Store Potential", "Store Potential Factor"],
    # Optional: used as a signal for store-level price power (not required)
    "DiscountPct": ["Discount%", "Discount %", "Disc%", "Disc %"],
    # Optional store sales value (used for QC impact summaries if present)
    "Sum of Sales_Value": ["Sales_Value", "Sum of Sales Value", "Total Sales Value", "Net Sales Value"],
}

ALIASES_TIERS = {
    "StoreID": ["Store Id", "Store", "Row Labels"],
    "PriceTier": ["Tier", "Price Tier", "Price_Tier"],
    "Average of SKU_DC": ["SKU_DC", "SKUs", "SKU DC"],
    "GMROI": ["Gmroi", "GM ROI"],
    # Optional inputs used to compute GMROI if missing
    "Gross_Margin": ["Gross Margin", "Total Gross Margin", "Total_Gross_Margin"],
    "StockCost": ["Avg StockCost", "Average StockCost", "Average Stock Cost", "Avg Inventory Cost"],
    "ClosingStockQty": ["Closing StockQty", "Closing Stock Qty", "Closing Stock"],
    # Optional: tier-level discount behaviour (not required)
    "DiscountPct": ["Discount%", "Discount %", "DiscountPct", "Disc%", "Disc %"],
}

ALIASES_ARTICLES = {
    "StoreID": ["Store Id", "Store", "Row Labels"],
    "ItemColorName": ["SKU", "SKU Name", "Item Color", "ItemColor"],
    "PriceTier": ["Tier", "Price Tier", "Price_Tier"],
    "Sales_Value": ["Sale_Value", "Sales Value", "Sale Value", "Net Sales"],
    "Sales_Qty": ["Sale_Qty", "Sales Qty", "Sale Qty", "Units", "Unit Sales"],
    "Gross_Margin": ["Gross Margin", "Total Gross Margin", "GM"],
    "ClosingStockQty": ["Closing StockQty", "Closing Stock Qty", "Closing Stock"],
    "Avg_Inventory_Cost": ["Average StockCost", "Average StockCost", "Average Stock Cost", "Avg Inventory Cost", "Avg_Inventory_Cost"],
    "RetailPrice": ["Retail Price", "RSP", "MRP", "Current Retail Price"],
    "DiscountPct": ["Discount%", "Discount %", "DiscountPct", "Disc%", "Disc %"],
    "Dept": ["DEPT", "Department"],
    "Category": ["SUB-CATEGORY", "Sub Category", "SubCategory", "Sub-Category", "Category"],
    "LifeCycle_Status": ["Lifecycle_Status", "Lifecycle Status", "LifeCycleStatus", "LifecycleStatus", "LC_Status"],
}

def _find_sheet(xl: pd.ExcelFile, candidates) -> str:
    """Find a sheet name, robust to casing and minor name differences.

    This protects production runs where users rename sheets like
    "storeskpis" or "INPUT_storekpis".
    """
    sheet_map = {str(s).strip().lower(): s for s in xl.sheet_names}
    # Exact/case-insensitive match
    for c in candidates:
        key = str(c).strip().lower()
        if key in sheet_map:
            return sheet_map[key]
    # Fuzzy match: remove non-alnum
    def norm(x: str) -> str:
        return "".join(ch for ch in str(x).lower() if ch.isalnum())
    norm_map = {norm(s): s for s in xl.sheet_names}
    for c in candidates:
        k = norm(c)
        if k in norm_map:
            return norm_map[k]
    raise ValueError(f"Missing expected sheet. Tried: {candidates}. Found: {xl.sheet_names}")

def load_workbook(path_or_file, sheet_map: Optional[Dict[str, list]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sheet_map = sheet_map or DEFAULT_SHEET_MAP
    try:
        xl = pd.ExcelFile(path_or_file, engine="openpyxl")
        s_stores = _find_sheet(xl, sheet_map["stores"])
        s_tiers = _find_sheet(xl, sheet_map["tiers"])
        s_articles = _find_sheet(xl, sheet_map["articles"])
        # IMPORTANT: Reuse the already-opened ExcelFile object (xl) for all
        # sheet reads. Previously, passing path_or_file directly to read_excel
        # caused openpyxl to re-parse the entire workbook from scratch for each
        # sheet — 4 total parses for an 11 MB file could consume 800+ MB RAM
        # and crash Streamlit Cloud's 1 GB free tier.
        stores = xl.parse(s_stores)
        tiers = xl.parse(s_tiers)
        articles = xl.parse(s_articles)

        # Normalize columns to canonical engine schema
        stores = _rename_with_aliases(stores, ALIASES_STORES)
        tiers = _rename_with_aliases(tiers, ALIASES_TIERS)
        articles = _rename_with_aliases(articles, ALIASES_ARTICLES)
        # -------------------------------------------------------------------
        # PriceTier responsibility (Audit Assumption A)
        # -------------------------------------------------------------------
        # The engine DOES NOT determine PriceTier. It must be provided in input.
        # We only normalize the column type if present.
        if "PriceTier" in articles.columns:
            articles["PriceTier"] = articles["PriceTier"].astype(str).str.strip()

        # -------------------------------------------------------------------
        # StoreID normalization (must be consistent across all sheets)
        # -------------------------------------------------------------------
        # A very common real-world issue:
        # - StoresKPIs store identifier column is "Row Labels" and may load as
        #   123 (int) or 123.0 (float) depending on Excel typing.
        # - TierKPIs / Articles store identifier is "StoreID" and may load as
        #   "123" (string).
        # This mismatch makes all joins miss and leads to Target SKUs = 0.
        #
        # We normalize ALL store identifiers to a clean string form:
        # - strip whitespace
        # - remove trailing .0 (Excel float artifact)
        def _coerce_store_id(s: pd.Series) -> pd.Series:
            x = s.astype(str).str.strip()
            x = x.str.replace(r"\.0$", "", regex=True)
            return x

        if "StoreID" in stores.columns:
            stores["StoreID"] = _coerce_store_id(stores["StoreID"])
        if "StoreID" in tiers.columns:
            tiers["StoreID"] = _coerce_store_id(tiers["StoreID"])
        if "StoreID" in articles.columns:
            articles["StoreID"] = _coerce_store_id(articles["StoreID"])

        # Compute Tier GMROI if missing but Gross_Margin and StockCost are present
        if "GMROI" not in tiers.columns:
            if "Gross_Margin" in tiers.columns and "StockCost" in tiers.columns:
                gm = pd.to_numeric(tiers["Gross_Margin"], errors="coerce").fillna(0)
                inv = pd.to_numeric(tiers["StockCost"], errors="coerce").fillna(0)
                tiers["GMROI"] = (gm / inv.replace(0, pd.NA)).fillna(0)

        return stores, tiers, articles
    except Exception as e:
        # Re-raise with a friendly message (app will show this to the user)
        raise ValueError(
            f"Failed to load workbook. Ensure the file is a valid .xlsx and contains StoresKPIs, TierKPIs, and Articles sheets. Details: {e}"
        ) from e


def load_protected_skus(path_or_file, candidates: Optional[list[str]] = None) -> list[str]:
    """Load an optional Protected SKU list from the SAME input workbook.

    Why this exists:
    Merchandising often maintains a list of "Hero" / "Strategic" SKUs that must
    never be removed in automated rationalization. This function reads those
    SKUs from a dedicated sheet if present.

    Expected schema:
      Sheet name: one of candidates (default supports common names)
      Column: ItemColorName (aliases: SKU, Item, ItemName)

    Returns:
      List[str] of ItemColorName values (trimmed, unique)
    """
    candidates = candidates or [
        "Protected_SKUs",
        "Protected",
        "DoNotRemove",
        "Do_Not_Remove",
        "Protected List",
        "Hero_SKUs",
        "Strategic_SKUs",
    ]
    try:
        xl = pd.ExcelFile(path_or_file, engine="openpyxl")
        # If none of the candidates exist, return empty list.
        try:
            sheet = _find_sheet(xl, candidates)
        except Exception:
            return []
        df = xl.parse(sheet)
        df = _rename_with_aliases(df, {"ItemColorName": ["SKU", "Item", "ItemName", "Item Color", "ItemColor"]})
        if "ItemColorName" not in df.columns:
            return []
        s = df["ItemColorName"].astype(str).str.strip()
        s = s[s.notna() & (s != "")]
        # Drop obvious NaN strings
        s = s[~s.str.lower().isin(["nan", "none", "null"])]
        return sorted(set(s.tolist()))
    except Exception:
        # Never break the run due to protected list parsing.
        return []


# ---------------------------------------------------------------------------
# Optional Eligibility Matrix loader
# ---------------------------------------------------------------------------

def load_eligibility_matrix(path_or_file, candidates: Optional[list[str]] = None) -> Optional[Dict[str, Dict[str, int]]]:
    """Load an optional Eligibility Matrix from the SAME input workbook.

    Purpose:
      Control which Store×Tier combinations are eligible for tier introduction /
      expansion. If not provided, engine proceeds with default eligibility rules.

    Supported formats:
      A) Long format (recommended)
         Columns: StoreID, PriceTier, Eligible (1/0)
      B) Wide matrix
         First column: StoreID
         Other columns: tier names (e.g., VALUE, MID, PREMIUM, ULTRA) with 1/0

    Sheet name: one of candidates (default supports common names)

    Returns:
      Dict[StoreID, Dict[PriceTier, int]] or None if sheet not found/empty.
    """
    candidates = candidates or [
        "Eligibility_Matrix",
        "EligibilityMatrix",
        "Eligibility",
        "Tier_Eligibility",
    ]
    try:
        xl = pd.ExcelFile(path_or_file, engine="openpyxl")
        s = _find_sheet(xl, candidates)
        if not s:
            return None
        df = xl.parse(s)
        if df is None or df.empty:
            return None

        df = df.copy()
        df.columns = [ _norm_col_name(c) for c in df.columns ]

        # Detect long format
        colset = set(df.columns)
        store_col = "StoreID" if "StoreID" in colset else ("Row Labels" if "Row Labels" in colset else None)
        if store_col is None:
            return None

        if "PriceTier" in colset:
            eligible_col = None
            for c in ["Eligible", "Eligibility", "IsEligible", "Allow"]:
                if c in colset:
                    eligible_col = c
                    break
            if eligible_col is None:
                # Assume all rows are eligible
                df["Eligible"] = 1
                eligible_col = "Eligible"

            out: Dict[str, Dict[str, int]] = {}
            for _, r in df.iterrows():
                sid = str(r[store_col]).strip()
                tier = str(r["PriceTier"]).strip()
                val = int(float(r[eligible_col])) if pd.notna(r[eligible_col]) else 0
                out.setdefault(sid, {})[tier] = 1 if val != 0 else 0
            return out or None

        # Wide format
        out: Dict[str, Dict[str, int]] = {}
        tier_cols = [c for c in df.columns if c != store_col]
        for _, r in df.iterrows():
            sid = str(r[store_col]).strip()
            if not sid:
                continue
            out.setdefault(sid, {})
            for tc in tier_cols:
                tier = str(tc).strip()
                val = r[tc]
                try:
                    out[sid][tier] = 1 if int(float(val)) != 0 else 0
                except Exception:
                    out[sid][tier] = 0
        return out or None
    except Exception:
        return None
