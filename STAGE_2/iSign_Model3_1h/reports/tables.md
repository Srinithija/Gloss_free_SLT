
#### Model 3 hybrid (SigLIP+MediaPipe), 2k train (`runs/m3_hybrid`)

| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.943 | 6.549 | not logged | 0.74 | 0.01 | 1.44 | 53 / 10% | 116 s |
| 1 **(best)** | 6.259 | 5.635 | 0.0% | 7.61 | 0.18 | 9.31 | 41 / 22% | 27 s |
| 2 | 5.584 | 5.476 | not logged | 7.19 | 0.09 | 8.52 | 80 / 19% | 23 s |
| 3 | 5.062 | 5.417 | not logged | 9.28 | 0.15 | 8.17 | 147 / 15% | 26 s |
| 4 | 4.545 | 5.571 | not logged | 5.16 | 0.10 | 6.06 | 107 / 10% | 28 s |
| 5 | 4.143 | 5.772 | not logged | 5.24 | 0.10 | 6.65 | 245 / 12% | 25 s |

Peak VRAM 3.00 GiB. Train loss uses label smoothing 0.1; validation loss has none.

#### Visual-only SigLIP baseline, 2k train (`runs/visual_only`)

| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.999 | 6.689 | not logged | 0.69 | 0.01 | 1.34 | 21 / 29% | 40 s |
| 1 | 6.415 | 5.752 | not logged | 1.98 | 0.10 | 4.91 | 4 / 67% | 24 s |
| 2 **(best)** | 5.677 | 5.426 | 0.0% | 7.63 | 0.13 | 8.63 | 28 / 41% | 22 s |
| 3 | 5.081 | 5.393 | not logged | 8.71 | 0.07 | 8.22 | 55 / 14% | 24 s |
| 4 | 4.536 | 5.486 | not logged | 4.28 | 0.12 | 4.56 | 39 / 72% | 27 s |
| 5 | 4.107 | 5.798 | not logged | 2.05 | 0.05 | 3.34 | 18 / 43% | 22 s |
| 6 | 3.712 | 5.986 | not logged | 4.09 | 0.05 | 5.10 | 70 / 48% | 24 s |

Peak VRAM 2.98 GiB. Train loss uses label smoothing 0.1; validation loss has none.

#### Model 3 hybrid (SigLIP+MediaPipe), 4.9k train (`runs/m3_hybrid_5k`)

| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.342 | 5.657 | not logged | 0.00 | 0.00 | 7.23 | 6 / 97% | 46 s |
| 1 | 5.946 | 5.201 | not logged | 6.76 | 0.08 | 8.67 | 80 / 27% | 34 s |
| 2 | 5.470 | 5.049 | not logged | 5.68 | 0.11 | 5.96 | 108 / 53% | 38 s |
| 3 | 5.011 | 4.959 | not logged | 6.95 | 0.15 | 7.14 | 89 / 27% | 58 s |
| 4 | 4.670 | 5.009 | not logged | 6.17 | 0.08 | 7.38 | 116 / 27% | 35 s |
| 5 **(best)** | 4.356 | 5.025 | 0.0% | 8.29 | 0.27 | 8.31 | 187 / 10% | 36 s |
| 6 | 4.109 | 5.270 | not logged | 6.61 | 0.12 | 6.91 | 351 / 7% | 35 s |
| 7 | 3.892 | 5.330 | not logged | 5.38 | 0.12 | 7.44 | 291 / 16% | 35 s |
| 8 | 3.725 | 5.290 | not logged | 7.82 | 0.17 | 7.56 | 265 / 14% | 35 s |
| 9 | 3.590 | 5.346 | not logged | 7.97 | 0.10 | 7.10 | 401 / 10% | 37 s |

Peak VRAM 2.99 GiB. Train loss uses label smoothing 0.1; validation loss has none.

#### Visual-only SigLIP baseline, 4.9k train (`runs/visual_only_5k`)

| Epoch | Training Loss | Validation Loss | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Distinct outputs / top share | Time (train+eval) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.417 | 5.865 | not logged | 0.00 | 0.00 | 3.21 | 1 / 100% | 58 s |
| 1 | 5.961 | 5.129 | not logged | 5.30 | 0.06 | 9.05 | 11 / 68% | 37 s |
| 2 | 5.378 | 4.993 | not logged | 6.38 | 0.13 | 6.79 | 25 / 41% | 38 s |
| 3 **(best)** | 4.879 | 4.967 | 0.0% | 6.47 | 0.14 | 6.41 | 20 / 20% | 38 s |
| 4 | 4.483 | 5.056 | not logged | 6.17 | 0.09 | 8.12 | 21 / 44% | 38 s |
| 5 | 4.115 | 5.136 | not logged | 8.52 | 0.11 | 7.66 | 82 / 19% | 42 s |
| 6 | 3.806 | 5.384 | not logged | 8.78 | 0.11 | 7.70 | 126 / 39% | 42 s |
| 7 | 3.569 | 5.519 | not logged | 6.35 | 0.09 | 7.24 | 95 / 37% | 58 s |

Peak VRAM 2.97 GiB. Train loss uses label smoothing 0.1; validation loss has none.

Exact match per epoch was not logged by the training loop. It is known only at the selected epoch (0.0% in every run; see `dev_at_best` in `test_metrics.json`).


### Test results (held-out `test1k`, 993 scorable clips)

| Experiment | Dataset | Samples | Decoding | Exact Match | BLEU-1 | BLEU-4 | ROUGE-L | Test loss | Distinct outputs (top share) | Content-word P / R (%) | ms/video | Remarks |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Model 3 hybrid (SigLIP+MediaPipe), 2k train | iSign test1k | 993 | greedy | 0.0% | 8.11 | 0.17 | 9.51 | 5.661 | 40 (24%) | 0.6 / 0.14 | 4.1 | best epoch 1 (dev BLEU-4 0.18) |
| Model 3 hybrid (SigLIP+MediaPipe), 2k train | iSign test1k | 993 | beam4_nr3 | 0.0% | 3.61 | 0.20 | 4.46 | 5.661 | 28 (42%) | 0.3 / 0.05 | 36.2 | best epoch 1 (dev BLEU-4 0.18) |
| Visual-only SigLIP baseline, 2k train | iSign test1k | 993 | greedy | 0.0% | 8.09 | 0.23 | 8.90 | 5.407 | 33 (40%) | 0.1 / 0.04 | 2.4 | best epoch 2 (dev BLEU-4 0.13) |
| Visual-only SigLIP baseline, 2k train | iSign test1k | 993 | beam4_nr3 | 0.0% | 0.90 | 0.06 | 1.43 | 5.407 | 6 (60%) | 0.4 / 0.05 | 53.9 | best epoch 2 (dev BLEU-4 0.13) |
| Model 3 hybrid (SigLIP+MediaPipe), 4.9k train | iSign test1k | 993 | greedy | 0.0% | 8.76 | 0.13 | 8.58 | 4.993 | 181 (12%) | 0.7 / 0.42 | 4.5 | best epoch 5 (dev BLEU-4 0.27) |
| Model 3 hybrid (SigLIP+MediaPipe), 4.9k train | iSign test1k | 993 | beam4_nr3 | 0.0% | 6.97 | 0.24 | 6.73 | 4.993 | 93 (20%) | 0.8 / 0.40 | 51.2 | best epoch 5 (dev BLEU-4 0.27) |
| Visual-only SigLIP baseline, 4.9k train | iSign test1k | 993 | greedy | 0.0% | 6.86 | 0.07 | 6.89 | 4.936 | 23 (18%) | 0.8 / 0.35 | 3.2 | best epoch 3 (dev BLEU-4 0.14) |
| Visual-only SigLIP baseline, 4.9k train | iSign test1k | 993 | beam4_nr3 | 0.0% | 8.56 | 0.19 | 7.37 | 4.936 | 59 (17%) | 0.9 / 0.38 | 29.0 | best epoch 3 (dev BLEU-4 0.14) |

### Key-word recognition probe (test1k; 300 most frequent content words; chance AUC = 0.5)

| Input features | Train / dev / test clips | Mean per-word AUC (test) | Shuffled-input control AUC | P@5 model | P@5 frequency prior | Best-recognised words (AUC, test clips) |
|---|---:|---:|---:|---:|---:|---|
| hybrid | 4921 / 990 / 993 | **0.673** | 0.495 | 0.037 | 0.035 | page (0.93, 15), sign (0.88, 10), deaf (0.78, 11), day (0.76, 11), police (0.76, 19), time (0.76, 18), etc (0.75, 12), am (0.75, 10) |
| landmarks | 4921 / 990 / 993 | **0.661** | 0.505 | 0.034 | 0.035 | page (0.90, 15), am (0.82, 10), know (0.81, 11), sign (0.81, 10), found (0.79, 12), pm (0.78, 10), police (0.76, 19), years (0.76, 11) |
| visual | 4921 / 990 / 993 | **0.639** | 0.494 | 0.035 | 0.035 | page (0.95, 15), sign (0.85, 10), while (0.81, 13), government (0.77, 17), due (0.77, 14), deaf (0.77, 11), world (0.74, 14), time (0.72, 18) |