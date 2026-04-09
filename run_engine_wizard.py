#!/usr/bin/env python
"""Portfolio Optimization Engine - Interactive CLI Wizard

This is an easier alternative to Streamlit for non-coders.

It asks for:
  1) Input Excel path (.xlsx)
  2) Output folder
  3) Optional config JSON (file path) OR leave blank to use defaults

Then it runs the exact same pipeline as cli_engine.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
import traceback

import pandas as pd


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _prompt_path(msg: str, must_exist: bool = True) -> Path:
    while True:
        raw = input(msg).strip().strip('"')
        if not raw:
            print("Please enter a path.")
            continue
        p = Path(raw).expanduser()
        if must_exist and not p.exists():
            print(f"Path not found: {p}")
            continue
        return p.resolve()


def _prompt_optional_config() -> dict:
    raw = input(
        "Optional Config JSON (file path). Press Enter to skip: "
    ).strip().strip('"')
    if not raw:
        return {}
    p = Path(raw).expanduser()
    if p.exists() and p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    # allow inline JSON
    try:
        return json.loads(raw)
    except Exception:
        raise ValueError("Config must be a JSON file path or valid inline JSON.")


def main() -> int:
    print("==================================================")
    print(" Portfolio Optimization Engine v21 (Wizard)")
    print("==================================================")
    print("Tip: You can copy-paste Windows paths.")
    print("Example: D:\\Data\\Men2.xlsx")
    print("--------------------------------------------------\n")

    input_path = _prompt_path("Input Excel (.xlsx) full path: ", must_exist=True)

    out_raw = input(
        "Output folder (will be created if not exists). Leave blank for ./outputs/run_<timestamp>: "
    ).strip().strip('"')
    if out_raw:
        outdir = Path(out_raw).expanduser().resolve()
    else:
        outdir = Path.cwd() / "outputs" / f"run_{_ts()}"

    outdir.mkdir(parents=True, exist_ok=True)

    try:
        overrides = _prompt_optional_config()
    except Exception as e:
        print(f"ERROR reading config: {e}")
        return 2

    log_path = outdir / "run_cli.log"

    try:
        from engine.io import load_workbook
        from engine import run_qc_checks
        from app import OptimizationConfig, execute_portfolio_optimization

        cfg = OptimizationConfig()
        # Apply overrides (same rule as cli_engine.py)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

        print("\nLoading workbook...")
        stores_df, tiers_df, articles_df = load_workbook(str(input_path))

        print("Running QC checks...")
        qc = run_qc_checks(stores_df, tiers_df, articles_df)

        qc_file = outdir / "QC_Report.xlsx"
        with pd.ExcelWriter(qc_file, engine="openpyxl") as xw:
            qc.errors.to_excel(xw, sheet_name="Errors", index=False)
            qc.warnings.to_excel(xw, sheet_name="Warnings", index=False)
            qc.info.to_excel(xw, sheet_name="Info", index=False)
            qc.suggestions.to_excel(xw, sheet_name="Suggestions", index=False)
            for name, df in (qc.snapshots or {}).items():
                sheet = f"Snap_{name}"[:31]
                try:
                    df.to_excel(xw, sheet_name=sheet, index=False)
                except Exception:
                    pass

        if (not qc.errors.empty) or (not qc.warnings.empty):
            print(f"QC completed with issues. Report saved: {qc_file}")
        else:
            print(f"QC completed. Report saved: {qc_file}")

        print("\nRunning engine...")
        # Optional: load Eligibility Matrix + Protected SKUs from the same workbook
        try:
            from engine.io import load_eligibility_matrix, load_protected_skus
            elig = load_eligibility_matrix(input_path)
            if elig:
                setattr(cfg, "eligibility_matrix", elig)
                print("Eligibility Matrix detected ✅ (Store×Tier rules will be enforced)")
            else:
                print("Eligibility Matrix not provided (engine will proceed with default eligibility rules)")
            prot = load_protected_skus(input_path)
            if prot and not getattr(cfg, "do_not_remove", None):
                setattr(cfg, "do_not_remove", prot)
                print(f"Protected SKUs detected ✅ (count={len(prot)})")
        except Exception:
            pass

        results = execute_portfolio_optimization(stores_df, tiers_df, articles_df, cfg)

        # Write outputs
        print("Saving outputs...")
        final_targets = results.get("Final_Targets")
        add_list = results.get("Add_List")
        remove_list = results.get("Remove_List")
        model_summary = results.get("Model_Summary")
        peer_pool = results.get("Peer_Pool_Top500")

        if isinstance(final_targets, pd.DataFrame):
            final_targets.to_excel(outdir / "Final_Targets.xlsx", index=False)
        if isinstance(add_list, pd.DataFrame):
            add_list.to_csv(outdir / "Add_List.csv", index=False, encoding="utf-8-sig")
        if isinstance(remove_list, pd.DataFrame):
            remove_list.to_csv(outdir / "Remove_List.csv", index=False, encoding="utf-8-sig")
        if isinstance(model_summary, pd.DataFrame):
            model_summary.to_excel(outdir / "Model_Summary.xlsx", index=False)
        if isinstance(peer_pool, pd.DataFrame):
            peer_pool.to_csv(outdir / "Peer_Pool_Top500.csv", index=False, encoding="utf-8-sig")

        print("\nDONE ✅")
        print(f"Outputs saved in: {outdir}")
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Portfolio Optimization Engine - Wizard FAILED\n")
            f.write(f"Input: {input_path}\n")
            f.write(f"Outdir: {outdir}\n\n")
            f.write(str(e) + "\n\n")
            f.write(tb)

        print("\nERROR: Engine failed ❌")
        print(f"Reason: {e}")
        print(f"Log saved: {log_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())