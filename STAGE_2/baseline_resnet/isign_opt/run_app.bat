@echo off
rem Streamlit lives in an isolated folder so the training environment's packages are not changed.
set PYTHONPATH=D:\iSign_Experiments\GFSLT\optimized_v2_20261023\app_deps
cd /d "%~dp0.."
"C:\Users\Revetha K\anaconda3\envs\sltbaselines\python.exe" isign_opt\launch_app.py --server.headless true --server.address localhost --browser.gatherUsageStats false %*
