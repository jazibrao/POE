"""Exports utilities.

We intentionally support **in-memory Excel export** so the app can run from
locked-down install locations (e.g., Program Files) without needing write
permissions.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def export_excel(path: str, sheets: Dict[str, Optional[pd.DataFrame]]) -> None:
    """Export multiple DataFrames to an Excel file on disk.

    This is kept for compatibility, but the Streamlit app should prefer
    :func:`export_excel_bytes`.
    """
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            for name, df in sheets.items():
                safe_name = str(name)[:31] if name else "Sheet1"
                (df if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=safe_name, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to export Excel to '{path}'. Check permissions/disk space. Details: {e}") from e


def export_excel_bytes(sheets: Dict[str, Optional[pd.DataFrame]]) -> BytesIO:
    """Export multiple DataFrames to an in-memory Excel file.

    Returns a BytesIO buffer positioned at the start.
    """
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for name, df in sheets.items():
                safe_name = str(name)[:31] if name else "Sheet1"
                (df if df is not None else pd.DataFrame()).to_excel(writer, sheet_name=safe_name, index=False)

        buffer.seek(0)
        return buffer
    except Exception as e:
        raise RuntimeError(f"Failed to export in-memory Excel bytes. Details: {e}") from e
