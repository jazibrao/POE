#!/usr/bin/env python3
"""Pre-release Build Checklist (AUTO-BLOCKER)

Runs:
1) python -m py_compile on all .py files
2) Minimal smoke test on sample_data.xlsx
3) Confirms output workbook created
4) Confirms non-empty Add_List and Remove_List sheets

If ANY step fails => exit code 1 and packaging should be blocked.
"""

from __future__ import annotations
import sys
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1

def ok(msg: str) -> None:
    print(f"OK  : {msg}")

def run() -> int:
    # 1) Compile all .py
    py_files = [str(p) for p in ROOT.rglob('*.py')]
    if not py_files:
        return fail("No python files found")

    cmd = [sys.executable, '-m', 'py_compile', *py_files]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        return fail("py_compile failed")
    ok(f"py_compile passed ({len(py_files)} files)")

    # 2) Smoke test on sample_data.xlsx (using CLI runner)
    sample = ROOT / 'sample_data.xlsx'
    if not sample.exists():
        return fail("sample_data.xlsx missing")

    outdir = ROOT / '_precheck_outputs'
    outdir.mkdir(parents=True, exist_ok=True)
    out_xlsx = outdir / 'Portfolio_Optimization_Outputs.xlsx'

    # Run CLI
    cmd2 = [sys.executable, str(ROOT/'cli_run_engine.py'), '--input', str(sample), '--outdir', str(outdir)]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    print(r2.stdout)
    if r2.returncode != 0:
        print(r2.stderr)
        return fail("CLI run failed on sample_data.xlsx")
    ok("CLI smoke run passed")

    # 3) Confirm output workbook created
    if not out_xlsx.exists() or out_xlsx.stat().st_size <= 0:
        return fail("Output workbook not created")

    ok("Output workbook created")

    # 4) Confirm non-empty Add/Remove
    try:
        add_df = pd.read_excel(out_xlsx, sheet_name='Add_List')
        rem_df = pd.read_excel(out_xlsx, sheet_name='Remove_List')
    except Exception as e:
        return fail(f"Failed to read Add_List/Remove_List: {e}")

    if add_df.empty:
        return fail("Add_List is empty")
    if rem_df.empty:
        return fail("Remove_List is empty")
    ok(f"Add_List rows={len(add_df)} | Remove_List rows={len(rem_df)}")

    return 0

if __name__ == '__main__':
    raise SystemExit(run())
