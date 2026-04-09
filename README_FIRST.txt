SERVIS RETAIL - Portfolio Optimization Engine (v2.1.0)
=====================================================

This package is the STABILIZED build for Windows.

RECOMMENDED WAY TO RUN (FAST + RELIABLE)
--------------------------------------
1) Double click: RUN_PORTFOLIO_ENGINE.bat
2) Wait until you see:
      URL: http://127.0.0.1:8501
3) Open that URL in Microsoft Edge/Chrome.

To stop the app:
- Press CTRL+C in the terminal, OR
- Double click: STOP_PORTFOLIO_ENGINE.bat

WHY THIS IS RECOMMENDED
----------------------
PyInstaller EXE builds of Streamlit are fragile on some machines
(antivirus, permissions, missing WebView2, hidden imports, etc.).
Running via the batch launcher is much more stable.

OPTIONAL: BUILD A SINGLE EXE (ADVANCED)
--------------------------------------
If you still want an EXE:
1) Double click: BUILD_EXE.bat
2) After success, run: dist\Portfolio_Optimization_Engine.exe

Important:
- Do NOT run BUILD_EXE.bat as Administrator
- If it looks stuck, open build_pip.log in Notepad to see progress

KNOWN FIXES INCLUDED
--------------------
- StoreID normalization is handled (prevents Target SKUs = 0 issues)
- Build scripts are corrected to avoid "More?" prompt / broken parsing
- Desktop shell has stronger error handling and logging

If you get any error after upload, check:
- logs\run.log
- streamlit errors in the terminal window
