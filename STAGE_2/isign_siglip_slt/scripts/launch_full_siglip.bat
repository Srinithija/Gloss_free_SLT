@echo off
rem Full SigLIP extraction (clean view) at the K=32 motion keyframes (resumable: rerun this file to continue).
cd /d "E:\FINAL YEAR PROJECT\GLOSS FREE SLT\isign_siglip_slt"
set PYTHONNOUSERSITE=1
envs\main\Scripts\python.exe scripts\extract_visual.py --manifest manifests\full_extract.csv --out cache\full_k32\siglip --encoder siglip --positions manifests\keyframes_K32.json >> logs\full_siglip.log 2>&1
