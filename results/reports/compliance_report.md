# EchoGesture deployment compliance report

Model: pure temporal 1D-CNN (`tcnn`, 866,960 parameters), checkpoint `models/checkpoints/tcnn/best.pt` (epoch 95, SHA-256 `efb862e7…`, unchanged), deployed as `models/exported/model_fp32.onnx`.
Hardware: MacBook Air M4, 16 GB, fanless, on AC power. Runtime: MediaPipe 0.10.14 (CPU), ONNX Runtime 1.32 CPU, 1 intra-op thread. All inference is local.

## Summary

| # | Requirement | Target | Measured | Status |
|---|---|---|---|---|
| 1 | Deployed model | FP32 ONNX | `model_fp32.onnx` | Pass |
| 2 | Same preprocessing as training | identical | 400/400 deployed predictions equal stored-array predictions | Pass |
| 3 | No cloud dependency | none | no network code; ONNX Runtime CPU only | Pass |
| 4 | Webcam capture + HUD | working | built; camera not accessible from this session | Partial |
| 5 | End-to-end latency (sign end -> word) | p95 <= 50 ms | p50 23.5 / p95 32.2 / p99 36.4 ms | Pass |
| 5b | Whole-clip latency from a video file | (reported) | p50 688 / p95 1,156 / p99 1,192 ms | Reported |
| 6 | Average power | <= 500 mW | 1.42 W process CPU energy while signing; 183 mW with no person | **Fail** |
| 7 | 10-minute thermal drift | <= 20% | latency drift +0.2% (p50) / +0.8% (p95); no thermal warning | Pass |
| 8 | Missing-hand robustness | drop <= 5 pp | 4.75 pp (95% CI 1.75-8.00) deployed; 1.0 pp keypoint-level | Pass (point estimate; not statistically established) |
| 9 | Checkpoint and evaluation results preserved | unchanged | hash and committed reports unchanged | Pass |
| 10 | Top-1 accuracy | >= 70% | 18.90% test (95% CI 17.25-20.60%) | **Fail** |

## Methodology and details

**Pipeline consistency (2).** `src/deployed_eval.py` ran 400 seeded test videos through the deployed path (video -> MediaPipe Holistic -> `preprocess()` -> FP32 ONNX) and compared top-1 with predictions from the stored keypoint arrays used in evaluation: 400/400 identical; top-1 20.0%, top-5 45.25% on this subset.

**Webcam and HUD (4).** `hud_demo.py` captures the camera, feeds unmirrored frames to MediaPipe (as in training; only the display is mirrored), samples landmarks at the training rate (25 fps / stride 2 = 12.5 Hz), segments signs by hand presence (start: 2 hand samples; end: 4 samples without hands or 96 samples; at least 4 hand samples), classifies each sign with the offline `preprocess()` + ONNX path, and shows captions, confidence, top-3 alternatives, FPS, landmark and sign-to-word latency, a confidence threshold (default 0.35, `[`/`]`) and repeat suppression (same word within 1.5 s). The camera could not be opened from this session because macOS had not granted camera access to the host app, so live-camera behaviour is untested. Every measurement below uses the identical `StreamingRecognizer` fed by test clips replayed at real-time 25 fps with 0.6 s blank gaps.

**Latency (5).** From the 10.5-minute run (`src/live_benchmark.py`, 213 signs): sign end -> word = MediaPipe on the closing frame + preprocessing + model = p50 23.5, p95 32.2, p99 36.4 ms (preprocess p95 0.62 ms, model p95 1.06 ms). Each word also waits a deliberate 320 ms (4 samples) to confirm the hands have dropped. Per-frame MediaPipe: p50 29.8, p95 36.1, p99 54.8 ms (7,458 samples). Processing a whole recorded clip at once (`src/benchmark.py`, 100 test videos, 3 warm-up) takes p50 688 / p95 1,156 / p99 1,192 ms, almost all MediaPipe.

**Power (6).** Measured with the macOS kernel's per-process CPU energy counter (`proc_pid_rusage` `ri_energy_nj`; `powermetrics` needs root). Scope: CPU energy of the recogniser process (video decode, MediaPipe, preprocessing, ONNX); it excludes camera, display and DRAM, so whole-system power is higher. Continuous signing: 1.42 W average, 1.39-1.45 W per minute, 50% of one CPU core. No-person stream: 183 mW. Idle process: about 0 mW. MediaPipe Holistic (about 29 ms per sample, 12.5 samples/s) dominates; the 1D-CNN costs under 1 ms per sign.

**Thermal drift (7).** 10.5 minutes of continuous real-time inference, 15,751 frames, 2 frames behind real time. Per-frame MediaPipe latency: first minute after a 30 s warm-up p50 29.75 / p95 36.11 ms, last minute p50 29.80 / p95 36.39 ms (+0.2% / +0.8%). Sign-to-word latency: -2.0% / -3.1%. `pmset -g therm` reported no thermal or performance warning in any minute. Streaming accuracy in this run: top-1 18.8%, top-5 47.4%.

**Missing hands (8).** Deployed test: in each of 400 test videos the dominant hand's pixel region (from stored landmarks, padded) was blacked out for 30% of contiguous frames before MediaPipe; that hand's detection in the window fell from 82.1% to 6.3%. Top-1 20.0% -> 15.25% (drop 4.75 pp, paired bootstrap 95% CI 1.75-8.00 pp; 31 clips lost, 12 gained); top-5 drop 8.0 pp. Keypoint-level test on all 2,000 test clips (one hand removed for 30% of frames): 1.0 pp top-1 drop.

**Accuracy (10).** Test (2,000 clips, 1 per class, official split, signer-dependent): top-1 18.90%, top-5 42.60%, macro F1 0.146. The 70% target is not met. 1,997/2,000 test clips come from signers seen in training, so accuracy for new signers is unmeasured and likely lower.

## Remaining failures and options (none applied; each needs approval)

- Power (1.42 W vs 500 mW). Options: lower MediaPipe sampling while no person is visible (measured floor 183 mW), MediaPipe pose `model_complexity=0`, or a lower landmark rate. The last two change the keypoints or temporal sampling the model was trained on and would require re-extraction and retraining.
- Accuracy (18.9% vs 70%). Limited mainly by 3 training clips per class for 2,000 classes. More data per class (the full 14,296-clip training split) is the largest lever.
- Live camera untested from this session; run `hud_demo.py` from Terminal after allowing camera access.
- The segmenter gained a minimum of 4 hand samples per sign after the 10-minute run (prevents captions from brief hand flickers). It does not change per-frame work, so latency and power figures are unaffected; no benchmark segment had 6 or fewer samples.
