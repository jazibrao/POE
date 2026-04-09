PORTFOLIO OPTIMIZATION ENGINE (v21) - EASY CMD RUN GUIDE
========================================================

If Streamlit App is not working, this package lets you run everything from CMD
with 2 double-click steps.

WHAT YOU NEED
-------------
1) Windows 10/11
2) Python 3.11.x installed (IMPORTANT)
   Download from python.org and tick: "Add python to PATH" during install.


FASTEST WAY (RECOMMENDED)
-------------------------
STEP 1 (one-time): Double-click  1_SETUP_ENV.bat
STEP 2 (every run): Double-click 2_RUN_ENGINE_WIZARD.bat


OUTPUTS CREATED
---------------
Inside your selected output folder you will get:
- QC_Report.xlsx
- Final_Targets.xlsx
- Add_List.csv
- Remove_List.csv
- Model_Summary.xlsx
- Peer_Pool_Top500.csv
- run_cli.log  (only if error happens)


CONFIGURATION (EXCEL-LIKE)
--------------------------
You can control key knobs like Excel:
- Efficient stores % (Top GMROI stores) used everywhere
- SPF/Q weights
- alpha min/max
- robust scaling percentiles

Edit this file to change settings:
  PortfolioEngine_Config_Template.json


IMPORTANT ENGINE RULES (YOUR CHOICE)
------------------------------------
Efficient Stores are STANDARD everywhere:
  Top 30% stores by TOTAL GMROI (not tier-wise)


HELP
----
If you see any error, open the log file inside output folder:
  run_cli.log
and share it with me.
