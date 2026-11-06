@echo off
rem Full MediaPipe extraction at the K=32 motion keyframes (resumable: rerun this file to continue).
cd /d "E:\FINAL YEAR PROJECT\GLOSS FREE SLT\isign_siglip_slt"
set PYTHONNOUSERSITE=1
set GLOG_minloglevel=2
envs\mediapipe\Scripts\python.exe scripts\extract_landmarks.py --manifest manifests\full_extract.csv --out cache\full_k32\landmarks --workers 12 --positions manifests\keyframes_K32.json --complexity 1 >> logs\full_landmarks.log 2>> logs\full_landmarks.stderr.log
