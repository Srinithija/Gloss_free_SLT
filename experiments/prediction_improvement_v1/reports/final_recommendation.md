# Final recommendation

## Best configuration (chosen on signer 6 only)

**E3:** candidate-sentence scoring over the 97 training sentences (alpha 0), with log-probabilities averaged with equal
weight over three models, each given the `trim_5` temporal view:

1. `outputs/siglip_b16_stage2/best.pt`: the existing selected SigLIP model, unchanged.
2. `outputs/clip_b16_stage1/best.pt`: the existing CLIP model, unchanged.
3. `experiments/prediction_improvement_v1/runs/H1_siglip_landmarks/best.pt`: new; frozen SigLIP features + MediaPipe
   hand/pose landmarks; SHA-256 `a9736646acb0…`, epoch 65.

| | Baseline | E3 |
|---|---|---|
| Signer 6 (validation) | 64/94 = 68.1% | 74/94 = 78.7% |
| Signers 3–5 (fold models) | 170/282 = 60.3% | 211/282 = 74.8% (+14.5 pts, interval +9.2 to +19.5) |
| Signer 7 | 23/91 = 25.3% (initial test) | 59/91 = 64.8% (repeat evaluation R2) |
| Model time per clip | 0.16 s | ≈2.7 s (2.3 s of it MediaPipe on CPU) |

**Did MediaPipe improve validation?**
- **Yes, combined with SigLIP:** H1 gets 73/94 vs 62/94 for the same recipe without landmarks.
  - On signers 3–5 it gets 178 vs 154 of 282 (interval +2.8 to +14.5 points).
  - Inside the ensemble it adds 21 clips on signers 3–5 over D3 (interval +3.9 to +11.3 points).
- **Not on its own:** landmarks alone (L1) are only on par with frozen SigLIP.

## If MediaPipe at inference is not wanted

Use **D3** instead: SigLIP stage 2 + CLIP, equal weights, candidate scoring, `trim_5`.
- It uses only the two existing checkpoints and takes 0.27 s per clip.
- Results:
  - signer 6: 73/94;
  - signers 3–5: 190/282;
  - signer 7: 43/91, repeat evaluation R1.

## Not recommended
- **Multiple temporal views for gains:** the best single view on signer 6 (trim_5) did not hold up on signers 3–5,
  and trim_10 lost 19 clips there. trim_5 is kept inside D3/E3 only because the rules selected it. Its contribution is
  within noise.
- **Speed and frame-drop augmentation:** no gain on signer 6 at either strength.
- **Landmark-only model as a replacement for SigLIP.**

## Next steps, in order
1. **Integrate E3 (or D3) into the app.** The app still runs the baseline greedy decoder.
   - D3 needs only the existing checkpoints plus `pi_exp/scoring.py` and `views.py`.
   - E3 also needs MediaPipe at inference. It lives in a separate environment (`env_mediapipe`), so the app would call
     the extractor as a subprocess, or MediaPipe would have to be installed alongside after checking for conflicts:
     it pulls in opencv-contrib and protobuf 4.
   - Make the 2.3 s CPU landmark cost visible to users.
2. **Fine-tune the hybrid** (stage 2: last SigLIP blocks + landmarks). The frozen hybrid already beats the fine-tuned
   SigLIP model on signers 3–5, so fine-tuning is the obvious next model experiment. Select on signer 6 and check on
   signers 3–5.
3. **More signers.** Errors that remain on signer 7 with the correct sentence ranked 11th–50th will not be fixed by
   modelling. Recording more people remains the most reliable improvement.
4. **Keep signer 7 for final checks only.** It has now been evaluated three times. Future model choices should rely on
   signers 3–6. Ideally, a new held-out signer should be recorded for an untouched test.
