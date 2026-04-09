SERVIS RETAIL - Portfolio Optimization Engine (CLI Outputs)
=========================================================

Goal
----
Generate output files WITHOUT Streamlit / UI.

How to use
----------
1) Copy these 3 files into your existing folder:
   D:\Portfolio_Optimization_Engine_v21_STABILIZED_FULL_PACKAGE_v2\v21_pkg\
   - cli_engine.py
   - RUN_OUTPUTS_ONECLICK.bat
   - README_ONECLICK_CLI.txt

2) Double-click:
   RUN_OUTPUTS_ONECLICK.bat

3) When asked, paste the FULL input Excel path, example:
   D:\Men2.xlsx

4) Press ENTER to accept default output folder (recommended).

Outputs
-------
The script will create a folder like:
  .\outputs\run_20260118_073355\

And produce:
  - Final_Targets.xlsx
  - Add_List.csv
  - Remove_List.csv
  - Model_Summary.xlsx
  - PeerPool_Top500.xlsx
  - QC_Report.xlsx
  - run_cli.log

If anything fails
-----------------
Open the log file printed at the end, for example:
  D:\...\v21_pkg\outputs\run_...\run_cli.log
