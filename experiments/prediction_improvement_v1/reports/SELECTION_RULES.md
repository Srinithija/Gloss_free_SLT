# Selection rules for prediction_improvement_v1

Written on 2026-10-24, before any experiment in this folder was run.

## Data
- Training uses signers 1-5 and validation uses signer 6, from `data/manifest.json`. The manifest is unchanged (SHA-256 is in
  `protected_files_before.json`).
- The candidate sentence list is the set of normalized training-split texts: signers 1-5 only.
- Signer 7 (test) is not loaded for any experiment, tuning or analysis until the final step below.

## Choosing settings
1. Every choice is made on signer 6 (94 clips). The choices are:
   - length normalization alpha;
   - the frame-sampling strategy and how views are combined;
   - the ensemble weight;
   - augmentation strength;
   - the training epoch and early stopping;
   - whether to use landmarks.
   Primary metric: exact match. Ties are broken by corpus BLEU-4, then by the simpler or faster setting.
2. Secondary check on signers 3, 4, 5 (282 clips): the existing leave-one-signer-out models, each trained without the
   signer it is evaluated on, are run with the same inference settings. Settings are not tuned on these signers. They show
   whether a gain seen on signer 6 holds for other unseen signers.
3. A change is adopted over its baseline only if both hold:
   - it gains at least one clip on signer 6;
   - it does not lower pooled exact match on signers 3-5, where that check is possible.
   Paired bootstrap intervals (2000 resamples over clips) are reported. No change is called significant unless its
   interval excludes zero.
4. The grids are fixed in advance:
   - alpha in {0, 0.5, 1.0};
   - the frame-sampling strategies listed in `reports/baseline_audit.md`, section 6;
   - SigLIP weight w in {0.5, 0.6, 0.7, 0.8, 0.9}, with CLIP weight 1 - w.

## Test signer (repeat evaluations)
- The initial test evaluation (23/91 = 25.3%) stays the official initial result. `outputs/TEST_EVALUATION_LEDGER.csv`
  is not modified.
- After all validation work, at most two repeat evaluations are run on signer 7. They are recorded in
  `results/REPEAT_TEST_LEDGER.csv`:
  - R1: the best validated inference-only setting, using the existing selected checkpoint;
  - R2: the best validated newly trained model, only if it beats R1 on signer 6.
- Repeat results are reported and are not used to change any choice.

## Addendum, after experiments A-C on signer 6 and before D was run
Settings selected individually on signer 6 (exact match, then BLEU-4):
- A: candidates, alpha 0.0 (66/94; alpha 0.5 also 66 but lower BLEU-4).
- B: trim_5 (68/94).
- C, greedy: w = 0.6 (69/94).
- C, candidates: w = 0.5 (69/94; ties w0.6-greedy on exact match, higher BLEU-4).

The planned combinations are D1-D3 in `configs/D_combinations.json`; nothing else is combined. The final inference
setting is the best of {A, B, C-greedy, C-candidates, D1, D2, D3} on signer 6, subject to rule 3: it must not lower
pooled exact match on signers 3-5 relative to the baseline.

## Addendum, before any landmark or augmentation training
- Training runs use `train_features.py`, which is the project's stage-1 recipe on cached inputs.
  - Control: V1_siglip_only (frozen SigLIP features only).
  - Landmark-only (L1) and hybrid (H1) runs are compared with V1, not with a different pipeline.
- Phase 5 augmentation is tested on the same frozen-SigLIP setup, training only, with two strengths fixed in advance:
  - AUG1: time_warp 0.2, frame_drop 0.1;
  - AUG2: time_warp 0.4, frame_drop 0.2.
  The spatial augmentation of the cached views is unchanged in both.
- Landmarks or augmentation are adopted if they beat V1 on signer 6 exact match (rule 3) and do not lose on the
  signers 3-5 check. That check is a fixed 60-epoch retraining per fold, scored once, compared with the same check of V1.

## Addendum, after H1 trained on signer 6 and before any hybrid inference setting was scored
The hybrid inference settings are fixed in `configs/E_hybrid.json` (E0-E4), with equal member weights and no weight
tuning. Candidate scoring uses alpha 0 and trim_5 as selected earlier. The overall best setting on signer 6 among
D3 and E0-E4 becomes R2, and is tested on signer 7 only if it uses the hybrid, beats D3 on signer 6, and the
hybrid's signers 3-5 check is not worse than V1's.

## Addendum, 2026-10-27, after all choices were final (user request)
The user asked for a test result of Model 3 (hybrid) alone, for the report. Two settings, fixed above in
`configs/R3_model3_alone.json`, are each evaluated once on signer 7:
- R3a: greedy;
- R3b: candidates with trim_5, as used inside E3.
They are recorded in `results/REPEAT_TEST_LEDGER.csv` as repeat evaluations. They are reported only and change no
choice: E3 stays the final system.

## Addendum, 2026-10-28, after all choices were final (user request)
For the report's performance table, Model 2 (CLIP) alone is evaluated once on signer 7 in two settings, fixed above in
`configs/R4_model2_alone.json`:
- R4a: greedy;
- R4b: candidates with trim_5, as used inside D3/E3.
They are recorded in `results/REPEAT_TEST_LEDGER.csv` as repeat evaluations, are reported only, and change no
choice.
