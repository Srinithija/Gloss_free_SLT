
### Best checkpoint and why
- **Selected model:** `runs/m3_hybrid_5k/best.pt` (SHA-256 in `best.pt.sha256`), epoch 5.
  - Highest dev BLEU-4 of all runs: 0.27.
  - Most varied dev output: 187 distinct outputs, 10% top share.
  - Validation loss is close to its minimum (5.03 against 4.96 at epoch 3).
- **Overfitting:** every run overfits after 2–5 epochs. Train loss keeps falling while validation loss rises.
  - Training longer would not help at this data size; early stopping ended every run well before the deadline.
- **More data mattered:** going from 2k to 4.9k training clips lowered the best validation loss from 5.42 to 4.96 for the hybrid. That is the clearest learning signal in the experiment.
- **Hybrid versus visual-only at 4.9k clips (same data, same settings):**
  - Best validation loss: 4.96 hybrid against 4.97 visual-only. Test loss: 4.99 against 4.94.
  - **In the seq2seq model, the landmarks give no measurable loss advantage at this scale.**
  - The hybrid does give much more varied outputs: 181 against 23 distinct test outputs (greedy), and greedy BLEU-1 8.76 against 6.86. These differences are within the noise of a single seed.
- **Dev BLEU-4 is a weak selector:** at values of 0.1–0.3 it is very noisy, and differences of about 0.1 are within noise.

---

## F. Evaluation results
All rows are genuinely completed runs. Each model was decoded on test once per decoding setting, after selection on dev.
- **Metric code:** `isign_siglip_slt/slt/metrics.py`, the same code used for the ResNet baseline.
  - BLEU uses sacreBLEU, corpus-level, `13a` tokenisation, on lowercased, punctuation-stripped text. ROUGE-L is the sentence-level F-measure, averaged.
- **"Content-word P / R":** precision and recall of non-stop-words in the hypothesis against the reference, in %.
- **ms/video:** batched (64) GPU decoding time per clip. It excludes feature extraction, which costs about 0.18 s (SigLIP) and about 0.4 s (MediaPipe, 14 workers) per clip of 32 frames.

{TABLES_TEST}

**Reference point on a different subset, not directly comparable:**
- The project's earlier ResNet-18 gloss-free baseline, trained on **101k** clips: test-subset (1,979 clips) BLEU-1 9.16, BLEU-4 0.76, ROUGE-L 8.16, exact match 0.10% (re-scored with the same metric code; `isign_siglip_slt/reports/02_baseline.md`).
- That model used 20× more training data, so a lower score from the one-hour Model 3 runs is expected. It says nothing yet about the architecture.

---

## G. Prediction examples (selected model `m3_hybrid_5k`, test1k, greedy)
**Exact matches:** none. 0 of 993 clips.

{EXAMPLES}

**What the outputs show:**
- **The model has learned the video domain:**
  - news-studio clips → political and police sentences ("The CPI(M) wants to …", "The police have …");
  - children's-book or educational clips → "Let's write.", "The square is a square.", "The tiger is very big.".
- The "partially correct" rows are mostly domain and language-prior overlap ("Let's …", "police", "Rs … crore"). They are not word-level recognition of the signs.

---

## H. ISL vs iSign comparison

### Historical ISL results (a different dataset; reference only)
| Model (ISL, Kaggle) | Exact Match | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|
| ResNet18 (original) | 9.9% | 12.6 | 23.2 |
| Model 1: fine-tuned SigLIP | 25.3% | 25.9 | 37.6 |
| Model 3 alone, greedy | 37.4% | 43.9 | 47.6 |
| Model 3 alone, 97 candidates | 45.1% | 47.6 | 53.9 |
| D3: Model 1 + Model 2 | 47.3% | 47.9 | 55.1 |
| E3: Model 1 + Model 2 + Model 3 | 64.8% | 67.4 | 70.3 |

### New iSign results (this experiment, test1k, greedy unless noted)
{TABLE_ISIGN_SUMMARY}

**Why the numbers are not comparable:**
- **ISL is effectively closed-set.** It has about 97 candidate sentences, each seen many times in training, which is why candidate re-ranking helped there.
- **iSign is open-vocabulary.** It has 98k distinct training sentences, and only 4.6% of test sentences appear in the full train set; just 8 of 993 test sentences appear in the 4.9k training subset.
- **Consequences:**
  - Exact match and candidate re-ranking are close to meaningless on iSign.
  - A 97-candidate restriction was deliberately **not** imposed.
  - The models were trained on 2–5% of iSign's training data for minutes.

**What transferred directly:**
- frozen SigLIP feature extraction;
- MediaPipe Holistic in static mode on the original-resolution frames;
- the landmark projection (Linear + LayerNorm + ReLU);
- 1024-d fusion;
- the Conv1d(k5) + BatchNorm + ReLU + MaxPool(2) + Linear + LayerNorm + ReLU temporal encoder;
- mBART-cc25 as the decoder with `en_XX`.

**What needed changes:**
- **Landmark dimensionality:** 204 instead of 233 (a separately designed layout; see C).
- **Frame budget:** K=32 motion-aware keyframes, because iSign clips are long (median 183 frames).
- **mBART size:** reduced to 3+3 layers with a trimmed vocabulary (6 GB GPU).
- **Visual features:** mean-centred per clip.
- **Data hygiene:** bad-crop clips excluded from training; a group split by source video.
- **Evaluation:** open-vocabulary generation metrics, plus a key-word probe instead of candidate sentences.

---

## I. Limitations and uncompleted work

1. **What was successfully implemented?**
   - The full Model 3 pipeline on iSign, end to end: keyframe selection → frozen SigLIP and MediaPipe features (cached) → fusion → temporal encoder → mBART encoder-decoder with teacher-forced cross-entropy → autoregressive greedy and beam generation → decoded English.
   - Verified shapes, masks, gradient flow and parameter updates.
   - A leakage-free train/dev/test split and test features extracted in this run.
   - Four trained models: hybrid and visual-only, each at 2k and 4.9k training clips.
   - A key-word recognition probe; saved checkpoints, logs and predictions.
2. **What actually learned during training?**
   - Train loss fell from 7.3–8.0 to 3.6–4.1, and validation loss improved to 4.96 (hybrid-5k).
   - The decoder learned **fluent English and the domain prior** (news versus children's educational content).
   - The visual and landmark encoders carry a **weak but real word-level signal**: key-word probe AUC 0.67 against 0.50 shuffled. The hybrid beats landmarks-only (0.66) and visual-only (0.64).
   - The generator does not yet convert that signal into correct content words.
3. **Sentence recognition or sign-to-text translation?**
   - The system is a genuine **direct sign-to-text generator**: there is no candidate list, and it produces 181 distinct outputs on test.
   - **In quality it does not translate yet.** Outputs are mostly generic, domain-conditioned sentences.
   - It is **not** sentence recognition, and it is **not** gloss recognition.
4. **Is the sign-to-word mapping correct?**
   - The **video-to-text mapping is correct**, verified programmatically and by inspecting consecutive story segments.
   - **No sign-to-word mapping is learned reliably:** content-word recall in generated text is under 0.5%.
   - At the probe level, some words are recognisable above chance: "sign", "deaf", "police", "government", "time" have AUC 0.73–0.88.
   - "page" (AUC 0.90–0.95) is almost certainly recognised from the visual **context** of story-book videos, not from a sign.
5. **Is mBART functioning properly?**
   - Mechanically yes. Pretrained weights load with no missing keys, the loss decreases, and generation starts from `en_XX` and stops at `</s>`.
   - Without beam search or repetition control, it sometimes loops ("registered registered …"). Beam-4 with no-repeat-3 removes the loops but makes outputs even more generic.
   - The tokenizer round-trip is exact.
6. **Can the model generate meaningful English sentences?**
   - It generates **grammatical, meaningful-looking English**, but the sentences are not faithful to the signed content.
7. **If only key signs are recognised, why?**
   - **There are no gloss labels.** The only supervision is free English sentences with about 98k distinct sentences and 33k word types, while this run had at most 4.9k clips.
   - That is far too little data for an alignment-free model to discover word-level correspondences.
   - The probe shows the features do separate some frequent words. The seq2seq decoder, trained for minutes, falls back on the language prior.
8. **Which components remain incomplete?**
   - Full-dataset training: landmark features exist for only about 4.9k training clips; the full extraction is estimated at 22–28 h of MediaPipe.
   - The full 12+12 mBART.
   - A landmarks-only seq2seq run.
   - Multi-seed runs and confidence intervals.
   - Full-test (12.6k clips) evaluation.
9. **What prevented further improvement?**
   - **Training-data volume:** 2–5% of iSign. Every run overfits within 2–5 epochs.
   - **Feature-extraction throughput:** MediaPipe runs at about 85 frames/s on the CPU, which is about 2.7 clips/s at K=32.
   - **The one-hour limit.**
10. **What caused the limitations?**
    - **Mainly the one-hour deadline,** through how many clips could be featurised, combined with **dataset limitations:** an open-vocabulary task with no gloss annotations.
    - **System specs played a role:** 6 GB VRAM forced the 3+3 mBART, and CPU-bound MediaPipe set the throughput.
    - **No implementation fault was found:** shapes, masks, the loss, gradients and the label mapping were all checked.
11. **What next?**
    1. **Run the planned full extraction** (K=32 MediaPipe for about 100k train clips, about 1 day in the background) and **train on all training clips.** The 2k → 4.9k gain in validation loss shows data is the main lever. The ResNet baseline on 101k clips also reached a higher BLEU-4 (0.76).
    2. **Add an auxiliary key-word loss.** A multi-label BCE over the top-N caption content words on the encoder output would push the encoder towards word-discriminative features. The probe shows this signal exists.
    3. **Pre-train the encoder–text alignment** contrastively, CLIP-style, against caption embeddings before seq2seq.
    4. **Use more frames for long clips** (K=48–64), plus SigLIP augmentation views.
    5. **Report on the full 12.6k test set with several seeds** and bootstrap confidence intervals.

---

## J. Final conclusion
- **Implementation:** Model 3 (frozen SigLIP ViT-B/16 + MediaPipe Holistic → 1024-d fusion → Conv1d temporal encoder → mBART-cc25) was ported to iSign and runs end to end.
  - It uses verified data mapping, leakage-free splits, motion-aware 32-keyframe sampling (about 16% of original frames kept), cached features, and a functioning teacher-forced and autoregressive mBART decoder.
- **Results:** in one hour, on 4.9k training clips, the best model (hybrid-5k) reaches test BLEU-4 {BEST_B4_GREEDY} greedy / {BEST_B4_BEAM} beam-4, BLEU-1 {BEST_B1}, ROUGE-L {BEST_RL} and 0% exact match on 993 held-out test clips.
  - Its outputs are fluent but domain-generic, so this is **not yet sign language translation**.
- **What did learn:**
  - More data clearly lowered the validation loss.
  - The fused features hold a measurable word-level signal (key-word AUC 0.67, chance 0.50), and the hybrid beats either stream alone in that probe.
  - In seq2seq loss, the hybrid and the visual-only model are tied at this scale. The ISL advantage of Model 3 is therefore only partly reproduced: in the probe and in output diversity, not in loss or BLEU.
- **Most promising next steps:** train on the full iSign training set once landmark extraction is complete, and add an auxiliary key-word or contrastive objective so the decoder is forced to use that signal.

### Artifacts
| What | Where (relative to `iSign_Model3_1h/`) |
|---|---|
| Selected checkpoint | `runs/m3_hybrid_5k/best.pt` (+ `.sha256`), with `last.pt` for resuming |
| Other checkpoints | `runs/{m3_hybrid,visual_only,visual_only_5k}/best.pt` |
| Per-epoch logs | `runs/*/train_log.jsonl`, `logs/train_*.log` |
| Test metrics and predictions | `runs/*/test_metrics.json`, `runs/*/test_predictions_{greedy,beam4_nr3}.tsv` |
| Key-word probe | `reports/keyword_probe_{hybrid,visual,landmarks}.json`, `keyword_probe.py` |
| Sanity check | `logs/sanity_check.log`, `sanity_check.py` |
| Dataset and frame statistics | `logs/dataset_stats.log`, `logs/frame_stats.log` |
| Manifests | `manifests/{train2k,train5k,dev1k,test1k}.csv` |
| Test feature caches | `cache/siglip_test`, `cache/landmarks_test` |
| Code | `run_train.py` (wrapper around the unchanged `isign_siglip_slt/scripts/train.py`), `stores.py`, `eval_test.py`, `make_tables.py` |
