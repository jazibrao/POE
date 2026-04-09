#!/usr/bin/env python3
"""
Servis Retail | Portfolio Optimization Engine (v3) - CLI Runner

Runs the portfolio optimization pipeline WITHOUT Streamlit.
Designed for maximum reliability on Windows, even for large Excel files.

Usage:
  python cli_run_engine.py --input "D:\Men2.xlsx" --outdir "outputs"

Outputs:
  - Portfolio_Optimization_Outputs.xlsx
  - run_cli.log
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

from engine.config import EngineConfig
from engine.qc import run_qc_checks
from engine.validate import validate_inputs
from engine.io import load_workbook
from engine.pipeline import run_optimization_pipeline
from engine.exports import export_excel
from engine.output_format import format_output


def _setup_logging(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run_cli.log"

    logger = logging.getLogger("portfolio_cli")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return log_path


def run(input_path: Path, outdir: Path, config_overrides: dict | None = None) -> int:
    log_path = _setup_logging(outdir)
    logger = logging.getLogger("portfolio_cli")

    logger.info("============================================")
    logger.info("Portfolio Optimization Engine v3 | CLI Run")
    logger.info("Input : %s", input_path)
    logger.info("Outdir: %s", outdir)
    logger.info("Log   : %s", log_path)
    logger.info("============================================")

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 2

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("[1/4] Loading workbook...")
    stores_df, tiers_df, articles_df = load_workbook(str(input_path))
    logger.info("Loaded: stores=%d, tiers=%d, articles=%d",
                len(stores_df), len(tiers_df), len(articles_df))

    # ------------------------------------------------------------------
    # Build config (with optional overrides)
    # ------------------------------------------------------------------
    cfg = EngineConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                logger.warning(f"Unknown config key ignored: {k}")

    # Optional: Protected SKU list from workbook
    try:
        from engine.io import load_eligibility_matrix, load_protected_skus
        elig = load_eligibility_matrix(input_path)
        if elig:
            cfg.eligibility_matrix = elig
        prot = load_protected_skus(str(input_path))
        if prot:
            existing = list(cfg.do_not_remove or [])
            cfg.do_not_remove = sorted(set(existing + prot))
            logger.info(f"Loaded {len(prot)} protected SKUs from workbook")
    except Exception as e:
        logger.debug(f"No eligibility/protected SKU sheets found: {e}")

    # ------------------------------------------------------------------
    # Validate & QC
    # ------------------------------------------------------------------
    logger.info("[2/4] Validating inputs...")
    validate_inputs(stores_df, tiers_df, articles_df)

    logger.info("[3/4] Running QC checks...")
    qc_res = run_qc_checks(stores_df, tiers_df, articles_df, cfg)
    logger.info("QC: errors=%d warnings=%d info=%d",
                len(qc_res.errors), len(qc_res.warnings), len(qc_res.info))

    # ------------------------------------------------------------------
    # Run pipeline (single source of truth)
    # ------------------------------------------------------------------
    logger.info("[4/4] Running optimization pipeline...")

    def _progress(msg: str, pct: int) -> None:
        logger.info(f"  [{pct:3d}%%] {msg}")

    result = run_optimization_pipeline(stores_df, tiers_df, articles_df, cfg, progress_cb=_progress)

    # ------------------------------------------------------------------
    # Build model summary
    # ------------------------------------------------------------------
    model_info = result["model_info"]
    model_summary = pd.DataFrame([
        {"Metric": "Efficient_Top_Pct", "Value": cfg.efficient_top_pct},
        {"Metric": "Efficiency_Method", "Value": model_info.get("efficiency_method", "gmroi_only")},
        {"Metric": "Efficiency_GMROI_Weight", "Value": model_info.get("efficiency_gmroi_weight", "N/A")},
        {"Metric": "Efficiency_ROS_Weight", "Value": model_info.get("efficiency_ros_weight", "N/A")},
        {"Metric": "Efficient_Threshold", "Value": model_info.get("efficient_threshold", None)},
        {"Metric": "Best_Gate_Weight_SPF", "Value": model_info.get("best_weight_spf", None)},
        {"Metric": "Best_Gate_Weight_Q", "Value": model_info.get("best_weight_q", None)},
        {"Metric": "Best_Model", "Value": model_info.get("best_model", None)},
        {"Metric": "CV_R2", "Value": model_info.get("cv_r2", None)},
        {"Metric": "Train_R2", "Value": model_info.get("train_r2", None)},
    ])

    # Append k_summary
    k_summary = result.get("k_summary", pd.DataFrame())
    if k_summary is not None and not k_summary.empty:
        k_long = k_summary.copy()
        k_long.insert(0, "Metric", "k_value_by_tier")
        model_summary = pd.concat(
            [model_summary, k_long.rename(columns={"Tier": "Value"})], ignore_index=True
        )

    remove_list = result["remove_list"]
    add_list = result["add_list"]

    logger.info("Remove list rows=%d | Add list rows=%d",
                len(remove_list) if remove_list is not None else 0,
                len(add_list) if add_list is not None else 0)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    out_excel = outdir / "Portfolio_Optimization_Outputs.xlsx"

    outputs = {
        "Model_Summary": model_summary,
        "Final_Targets": format_output(result["tier_targets"]),
        "Add_List": format_output(add_list),
        "Remove_List": format_output(remove_list),
        "QC_Errors": qc_res.errors,
        "QC_Warnings": qc_res.warnings,
        "QC_Info": qc_res.info,
        **{k: v for k, v in (qc_res.snapshots or {}).items()},
        "Add_Shortfall_Report": result.get("shortfall", pd.DataFrame()),
    }

    # Add transfer recommendations if present
    transfers = result.get("transfer_recommendations")
    if transfers is not None and not transfers.empty:
        outputs["Transfer_Recommendations"] = format_output(transfers)

    export_excel(str(out_excel), outputs)
    logger.info("Excel written: %s", out_excel)

    if out_excel.exists() and out_excel.stat().st_size > 0:
        logger.info("Output workbook created successfully.")
    else:
        logger.error("Output workbook not created or is empty.")
        return 3

    if add_list is None or add_list.empty:
        logger.warning("Add_List is empty. Check peer pool / eligibility / lifecycle restrictions.")
    if remove_list is None or remove_list.empty:
        logger.warning("Remove_List is empty. Check gaps or do-not-remove rules.")

    logger.info("DONE")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Portfolio Optimization Engine - CLI")
    ap.add_argument("--input", required=True, help="Path to input Excel (.xlsx)")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument(
        "--config",
        default="",
        help="Optional config: JSON string or path to JSON file",
    )
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    # Parse config overrides
    overrides = None
    if args.config.strip():
        cfg_input = args.config.strip()
        cfg_path = Path(cfg_input).expanduser()
        if cfg_path.exists() and cfg_path.is_file():
            import json
            with open(cfg_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
        else:
            import json
            overrides = json.loads(cfg_input)

    return run(input_path, outdir, config_overrides=overrides)


if __name__ == "__main__":
    raise SystemExit(main())
