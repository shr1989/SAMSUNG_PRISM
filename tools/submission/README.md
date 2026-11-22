# EchoGesture: on-device isolated ASL word recognition (WLASL-2000)

The pipeline runs fully offline: video → MediaPipe Holistic keypoints (75 landmarks × xyz, every 2nd frame) → pose-invariant preprocessing → 64 × 576 features → a pure temporal 1D-CNN (866,960 parameters) → one of 2,000 words. No cloud calls are made, and the data never leaves the device.

## Key results (measured; full table in `results.csv`)

| Metric | Target | Measured | Status |
|---|---|---|---|
| INT8 model size | ≤ 20 MB | $int8_mb MB (FP32: $fp32_mb MB) | pass |
| Model latency p95 (FP32, 1 CPU thread) | ≤ 50 ms | $model_p95 ms | pass |
| Live sign end → word latency p95 (MediaPipe + model) | ≤ 50 ms | $stream_p95 ms | pass |
| Whole video file → word p95 (offline `predict()`) | reported | $file_p95 ms | reported |
| Test top-1 / top-5, FP32 (2,000 clips) | ≥ 70% | $fp32_top1% / $fp32_top5% | **fail** |
| Test top-1 / top-5, INT8 | ≥ 70% | $int8_top1% / $int8_top5% | **fail** |
| `predict()` on test videos (`predictions.jsonl`) | ≥ 70% | $entry_top1% / $entry_top5% ($entry_clips clips, $entry_errors errors) | **fail** |
| Occlusion drop: one hand removed from keypoints | ≤ 5 pp | $occ_kp pp | pass |
| Occlusion drop: hand blacked out in video pixels | ≤ 5 pp | $occ_video pp (95% CI $occ_ci) | pass (point estimate) |
| Rotation drop | reported | $rot pp | reported |
| Average power | ≤ 500 mW | $power_mw mW | **fail** |
| Thermal drift over 10 min (p95 latency) | ≤ 20% | $drift% | pass |

### Notes on the results

- **Accuracy:** it is far below the 70% target. The task uses all 2,000 WLASL words with only 3 training clips per word; published models are also far below 70% at this vocabulary size.
- **Signer overlap:** the official test split shares signers with training (1,997 of 2,000 test clips), so accuracy on new signers is not measured.
- **Test subset:** the test set is 1 clip per class (2,000 clips) from the official WLASL-2000 test split (`data/processed/final_manifest.csv`). It was used once for evaluation and never for tuning or calibration.
- **Hardware:** all numbers come from a MacBook Air M4 (CPU), not a phone.
- **Power:** it is the macOS per-process CPU energy counter, not a phone power profiler.
- **No metrics are estimated:** see `results/reports/compliance_report.md` for methodology.

## Installation (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs offline once the packages are installed, because the MediaPipe models ship inside the `mediapipe` wheel. Only the first 7 packages are needed for `predict()`; `gradio` is for the browser demo; `torch`, `onnx`, `scikit-learn` and `matplotlib` are needed only to reproduce training, evaluation and the INT8 conversion.

## Browser demo (app.py)

```bash
python app.py        # then open http://127.0.0.1:7860
```

**Upload or record** a video of one sign, then press **Recognise sign**. The app runs the same pipeline as `predict()` (identical top-5 and scores in tests) and shows the word, top-5 scores and a timing breakdown. Webcam recording needs browser camera permission; it was verified in two real recordings on Windows in the Brave browser.

**Demo video:** `video/EchoGesture_Demo.mp4` (2 min 22 s) shows two real webcam recordings of the app, plus captioned slides.

## Running inference

```bash
python main.py path/to/sign.mp4                 # FP32 ONNX (default, deployed model)
python main.py path/to/sign.mp4 --model int8    # INT8 ONNX
python main.py clip.mp4 --start-frame 10 --end-frame 60
```

From Python, call it from this folder:

```python
from main import predict  # same function as src.main.predict

result = predict("path/to/sign.mp4")
```

Here is real output for test video `12004.mp4` (true word COMMITTEE, a miss) from a fresh Python 3.11 environment with network access blocked, with top-5 entries 3-5 shortened:

```json
{
  "prediction": "POLICEMAN",
  "confidence": 0.3296087384223938,
  "top5": [
    {"word": "POLICEMAN", "label": 1867, "confidence": 0.3296087384223938},
    {"word": "POLICE", "label": 167, "confidence": 0.12588708102703094},
    "... CHARACTER, CHRIST, SATISFY"
  ],
  "latency_ms": {"landmarks_ms": 4489.19, "preprocess_ms": 0.42, "model_ms": 0.94, "end_to_end_ms": 4490.55},
  "frames_used": 26,
  "model": "model_fp32.onnx"
}
```

The first call in a new process includes MediaPipe start-up, which took about 4.5 s here. Warm calls on whole test videos take p50 688 ms and p95 1,156 ms, almost all in MediaPipe.

**Error handling:**
- A missing file raises `FileNotFoundError`.
- An unsupported, undecodable or empty video, or an invalid frame range, raises `ValueError`.
- From the command line, the error is printed as JSON to stderr and the exit code is 1.

**Notes on the output:**
- `prediction` is the top-1 word in upper case.
- `label` is the WLASL-2000 class id.
- `landmarks_ms` covers decoding plus MediaPipe over the whole clip.

## Reproducing predictions.jsonl

This requires the WLASL-2000 videos (for example the Kaggle "WLASL-2000 Resized" copy):

```bash
python -m src.make_predictions --videos-dir /path/to/wlasl/videos --workers 4
```

It calls `src.main.predict` on every test clip of `data/processed/final_manifest.csv`, using the official NSLT clip bounds, and writes `predictions.jsonl` plus `predictions.summary.json`. Each line looks like this:

```json
{"video_id": "...", "label": 0, "true_word": "BOOK", "prediction": "BOOK", "confidence": 0.41,
 "top5": [...], "correct_top1": true, "correct_top5": true, "latency_ms": {...}, "frames_used": 31,
 "model": "model_fp32.onnx"}
```

`predictions.json` holds the same records as a single JSON array.

## Reproducing the model from scratch

**Warning:** these commands overwrite `models/` and `results/` in this folder.

```bash
export ECHOGESTURE_RAW_DIR=/path/to/wlasl-complete   # contains videos/, WLASL_v0.3.json, nslt_2000.json
python -m src.extract_keypoints --manifest data/processed/final_manifest.csv --workers 4
python -m src.validate_keypoints
python -m src.preprocess          # 64-frame tensors + train-only feature statistics
python -m src.train               # MPS/CUDA/CPU, 150 epochs, best epoch by val macro-F1
python -m src.evaluate            # test once: clean / occluded / rotated
python -m src.export_quantize     # FP32 ONNX + static INT8 (300 training calibration clips)
```

**Where the dataset split comes from:** `data/processed/final_manifest.csv` defines the 10,000 videos used (3 train / 1 val / 1 test per class), drawn from the official WLASL-2000 (NSLT) splits. `data/processed/manifest.csv` is the full WLASL-2000 index, which supplies the class-id → word mapping.

## Measurement scripts

| Script | Measures |
|---|---|
| `python -m src.benchmark` | Whole-video latency p50/p95/p99 |
| `python -m src.live_benchmark --minutes 10` | Streaming latency, power, thermal drift |
| `python -m src.deployed_eval` | Pixel-level hand occlusion on test videos |
| `python -m src.hud_demo` | Webcam caption display (HUD) |

**About `src.hud_demo`:**
- It is the webcam HUD used for the demo: captions, confidence, top-3 guesses, FPS and latency.
- It is not needed for automated evaluation.
- On macOS, grant camera access to your terminal first.

## Layout

```
main.py                    predict() entry point (wraps src/main.py)
requirements.txt
README.md
predictions.jsonl          one line per test clip, produced by predict()
predictions.json           same records as a JSON array
results.csv                every metric vs target, device, runtime, method
SHA256SUMS.txt
src/                       pipeline code (main.py, preprocess.py, model.py, export_quantize.py, ...)
models/exported/           model_fp32.onnx (deployed), model_int8.onnx
models/checkpoints/tcnn/   best.pt (epoch 95) + training history
data/processed/            preprocess_config.json, feature_stats.npz, manifests
results/                   evaluation_metrics.csv, reports/, plots/
```

## Preprocessing contract (identical in training and `predict()`)

All of this lives in `src/preprocess.py::preprocess`.

1. **Keypoints:** `(T, 75, 3)`, ordered as pose 0-32, then left hand 33-53, then right hand 54-74. A missing group is NaN.
2. **Gaps:** gaps of up to 4 samples are interpolated; trailing gaps are held at the last value.
3. **Normalisation:** keypoints are centred on the clip-median shoulder midpoint and scaled by shoulder width. Legs (pose 23-32) are dropped.
4. **Fixed length:** values are clipped to ±6, then the clip is resampled linearly to 64 frames.
5. **Missing values** become 0.

The 576 features (225 body + 126 upright wrist-relative hand + 225 velocity) and the train-only standardisation are computed inside the ONNX graph.
