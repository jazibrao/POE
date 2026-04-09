#!/usr/bin/env python
"""Portfolio Optimization Engine - CLI Runner (No Streamlit UI)

This script is intentionally **UI-free**. It loads the standard Excel input
workbook, runs the portfolio optimization pipeline, and writes the output files
to an output directory.

Designed for non-coders:
  python cli_engine.py --input "D:\\Men2.xlsx" --outdir "D:\\Outputs"

Outputs created:
  - Portfolio_Outputs.xlsx  (ALL outputs in one workbook ✅)
  - Final_Targets.xlsx
  - Add_List.csv
  - Remove_List.csv
  - Model_Summary.xlsx
  - Peer_Pool_Top500.csv

If any error happens, it prints a clear message and writes a detailed log.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import traceback

import pandas as pd


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_outputs(results: dict, outdir: Path, qc_result=None) -> None:
    """Write engine outputs to disk.

    We create both:
      1) Individual files (backward compatible)
      2) A single consolidated workbook: Portfolio_Outputs.xlsx
         (easy to share with Merchandising in one file)
    """
    # Always create outdir
    _safe_mkdir(outdir)

    # Expected keys from app.execute_portfolio_optimization
    final_targets = results.get("Final_Targets")

    overall_targets = results.get("Overall_Targets")
    add_list = results.get("Add_List")
    remove_list = results.get("Remove_List")
    model_summary = results.get("Model_Summary")
    store_grid = results.get("Store_SKU_Grid")
    peer_pool = results.get("Peer_Pool_Top500")
    if peer_pool is None:
        peer_pool = results.get("Peer_Pool_Top")
    if peer_pool is None:
        peer_pool = results.get("Peer_Pool_Top500.csv")

    # 1) Final targets as Excel
    try:
        from engine.output_format import format_output
    except Exception:
        format_output = lambda x: x

    if isinstance(final_targets, pd.DataFrame):
        format_output(final_targets).to_excel(outdir / "Final_Targets.xlsx", index=False)

    if isinstance(overall_targets, pd.DataFrame):
        format_output(overall_targets).to_excel(outdir / "Overall_Targets.xlsx", index=False)

    # 2) Adds / Removes
    if isinstance(add_list, pd.DataFrame):
        format_output(add_list).to_csv(outdir / "Add_List.csv", index=False, encoding="utf-8-sig")
    if isinstance(remove_list, pd.DataFrame):
        format_output(remove_list).to_csv(outdir / "Remove_List.csv", index=False, encoding="utf-8-sig")

    # 3) Model summary
    if isinstance(model_summary, pd.DataFrame):
        format_output(model_summary).to_excel(outdir / "Model_Summary.xlsx", index=False)

    # 4) Peer pool
    if isinstance(peer_pool, pd.DataFrame):
        format_output(peer_pool).to_csv(outdir / "Peer_Pool_Top500.csv", index=False, encoding="utf-8-sig")

    # 5) Consolidated workbook (all outputs in one Excel)
    try:
        from engine.exports import export_excel

        sheets = {
            "Final_Targets": format_output(final_targets) if isinstance(final_targets, pd.DataFrame) else None,
            "Overall_Targets": format_output(overall_targets) if isinstance(overall_targets, pd.DataFrame) else None,
            "Add_List": format_output(add_list) if isinstance(add_list, pd.DataFrame) else None,
            "Remove_List": format_output(remove_list) if isinstance(remove_list, pd.DataFrame) else None,
            "Model_Summary": format_output(model_summary) if isinstance(model_summary, pd.DataFrame) else None,
            "Peer_Pool_Top500": format_output(peer_pool) if isinstance(peer_pool, pd.DataFrame) else None,
            "Store_SKU_Grid": format_output(store_grid) if isinstance(store_grid, pd.DataFrame) else None,
            "SKU_Status_Master": format_output(results.get("SKU_Status_Master")) if isinstance(results.get("SKU_Status_Master"), pd.DataFrame) else None,
        }

        # Append QC tables (if available)
        if qc_result is not None:
            sheets.update(
                {
                    "QC_Errors": getattr(qc_result, "errors", None),
                    "QC_Warnings": getattr(qc_result, "warnings", None),
                    "QC_Info": getattr(qc_result, "info", None),
                    "QC_Suggestions": getattr(qc_result, "suggestions", None),
                }
            )

            # Small helpful snapshots (optional)
            for name, df in (getattr(qc_result, "snapshots", None) or {}).items():
                sheets[f"QC_Snap_{name}"] = df

        export_excel(str(outdir / "Portfolio_Outputs.xlsx"), sheets)
    except Exception:
        # Consolidated workbook is convenience only. Never fail the run because of it.
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio Optimization Engine - CLI")
    parser.add_argument("--input", required=True, help="Full path of input Excel (.xlsx)")
    parser.add_argument(
        "--outdir",
        default="",
        help="Output folder path. If omitted, ./outputs/run_<timestamp> is used.",
    )
    parser.add_argument(
        "--config",
        default="",
        help=(
            "Optional config override: either a JSON string OR a path to a JSON file. Example JSON: "
            "{\"efficient_store_pct\":30,\"gate_weight_spf\":0.70,\"gate_weight_q\":0.30,\"alpha_min\":0,\"alpha_max\":1}"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 2

    # Outdir handling
    if args.outdir.strip():
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = Path.cwd() / "outputs" / f"run_{_timestamp()}"

    _safe_mkdir(outdir)

    log_path = outdir / "run_cli.log"
    try:
        # Import inside try so we can log any import errors cleanly
        from engine.io import load_workbook
        from engine.io import load_protected_skus
        from engine import run_qc_checks
        from engine.config import EngineConfig
        from engine.pipeline import run_optimization_pipeline
        from engine.output_format import format_output

        # Build config
        cfg = EngineConfig()
        if args.config.strip():
            cfg_input = args.config.strip()
            cfg_path = Path(cfg_input).expanduser()
            if cfg_path.exists() and cfg_path.is_file():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    overrides = json.load(f)
            else:
                overrides = json.loads(cfg_input)
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    raise ValueError(f"Unknown config key: {k}")

        # Load workbook (robust sheet matching)
        stores_df, tiers_df, articles_df = load_workbook(str(input_path))

        # Optional: Protected SKU list (Hero / Strategic SKUs)
        protected = load_protected_skus(str(input_path))
        if protected:
            existing = list(cfg.do_not_remove or [])
            cfg.do_not_remove = sorted(set(existing + protected))
            print(f"Loaded Protected SKUs: {len(protected)} items")

        # Optional: Eligibility matrix from workbook
        try:
            from engine.io import load_eligibility_matrix
            elig = load_eligibility_matrix(input_path)
            if elig:
                cfg.eligibility_matrix = elig
        except Exception:
            pass

        # QC checks
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

        # Run pipeline (shared with Streamlit UI and cli_run_engine.py)
        pipeline_result = run_optimization_pipeline(stores_df, tiers_df, articles_df, cfg)

        # Map pipeline result to output format expected by _write_outputs
        results = {
            "Final_Targets": format_output(pipeline_result["tier_targets"]),
            "Overall_Targets": format_output(pipeline_result["stores_enriched"]),
            "Add_List": format_output(pipeline_result["add_list"]),
            "Remove_List": format_output(pipeline_result["remove_list"]),
            "Model_Summary": pd.DataFrame([
                {"Metric": k, "Value": v} for k, v in pipeline_result["model_info"].items()
            ]),
            "Peer_Pool_Top500": None,
            "Store_SKU_Grid": None,
            "SKU_Status_Master": None,
            "Add_Shortfall_Report": pipeline_result.get("shortfall"),
        }
        transfers = pipeline_result.get("transfer_recommendations")
        if transfers is not None and not transfers.empty:
            results["Transfer_Recommendations"] = format_output(transfers)

        # Write outputs (including consolidated workbook)
        _write_outputs(results, outdir, qc_result=qc)

        print("\nDONE ✅")
        print(f"Outputs saved in: {outdir}")
        return 0

    except Exception as e:
        # Write detailed traceback to log file
        tb = traceback.format_exc()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Portfolio Optimization Engine - CLI FAILED\n")
            f.write(f"Input: {input_path}\n")
            f.write(f"Outdir: {outdir}\n\n")
            f.write(str(e) + "\n\n")
            f.write(tb)

        print("\nERROR: Engine failed ❌")
        print(f"Reason: {e}")
        print(f"Log saved: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())