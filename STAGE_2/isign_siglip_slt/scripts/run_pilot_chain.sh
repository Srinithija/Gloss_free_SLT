#!/usr/bin/env bash
# Waits for the running visual pilot, then trains the remaining pilot variants one after another.
cd "/e/FINAL YEAR PROJECT/GLOSS FREE SLT/isign_siglip_slt"
export PYTHONNOUSERSITE=1
until grep -q '"done": true' runs/pilot_visual/train_log.jsonl 2>/dev/null; do sleep 20; done
for c in pilot_hybrid pilot_landmarks pilot_visual_noaug pilot_visual_T96; do
  envs/main/Scripts/python.exe scripts/train.py --config configs/$c.json > logs/$c.log 2>&1
  echo "finished $c exit=$?" >> logs/pilot_chain.log
done
echo "chain done" >> logs/pilot_chain.log
