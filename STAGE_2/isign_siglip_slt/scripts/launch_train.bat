@echo off
rem usage: launch_train.bat <config_name>   (resumes from runs\<name>\last.pt if present)
cd /d "E:\FINAL YEAR PROJECT\GLOSS FREE SLT\isign_siglip_slt"
set PYTHONNOUSERSITE=1
envs\main\Scripts\python.exe scripts\train.py --config configs\%1.json >> logs\%1.log 2>&1
