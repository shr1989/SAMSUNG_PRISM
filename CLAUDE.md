# EchoGesture

Isolated ASL word recognition on WLASL-2000 (2,000 classes): MediaPipe Holistic keypoints -> CNN-GRU -> ONNX Runtime on-device.

## Pipeline and commands

Run everything from the project root with `.venv/bin/python` (Python 3.11).

| Stage | Command | Output |
|---|---|---|
| Inspect dataset | `python -m src.inspect_dataset` | `data/processed/manifest.csv`, `results/reports/dataset_report.json` |
| Select subset | `python -m src.select_subset` | `data/processed/reduced_manifest.csv`, `excluded_samples.csv` |
| Extract keypoints | `python -m src.extract_keypoints --workers 4` | `data/keypoints/stride2/*.npy`, `extraction_status.csv` |
| Preprocess | `python -m src.preprocess` | `data/processed/{train,val,test}.npz`, `preprocess_config.json` |
| Train | `python -m src.train --arch cnn_gru` (`--resume` to continue) | `models/checkpoints/<arch>/{best,last}.pt` |
| Evaluate | `python -m src.evaluate` | `results/results.csv`, `results/reports/evaluation_report.json` |
| Export | `python -m src.export_quantize` | `models/exported/model_{fp32,int8}.onnx` |
| Predict | `python main.py <video.mp4>` | JSON |

## Data contracts

- Keypoint arrays: `(T, 75, 3)` float32. Pose 0-32, left hand 33-53, right hand 54-74. A missing group is all-NaN. Hand identity is anatomical on unmirrored video.
- Extraction uses every 2nd frame (12.5 fps, `FRAME_STRIDE=2`). The full-frame Phase 2 arrays in `data/keypoints/*.npy` are kept but unused.
- Model input: `(B, 32, 75, 3)`. Shoulder-centred, shoulder-width-scaled, short gaps interpolated, legs dropped, missing = exactly 0. Features (314) and the train-fitted standardisation are inside the model and the ONNX graph.
- `src/preprocess.py::preprocess` is the single transform for training and inference.

## Rules

- Official NSLT-2000 splits only. Never move test data, never tune on test, never calibrate on test.
- Results are signer-dependent: 99.97% of test videos come from signers seen in training.
- Never fabricate metrics. Report measured values and label estimates.
- Keep operations resumable and non-destructive. Do not delete keypoint arrays or `data/raw`.
- Lint with `ruff format` and `ruff check`; run `pytest -q` before committing.
- Code style: no tutorial comments, no decorative separators, pathlib, central `src/config.py`.

## Hardware

MacBook Air M4, 16 GB, fanless. Run one heavy job at a time. MediaPipe extraction saturates at about 4 workers (~113-145 videos/min). Train on MPS in fp32 (about 10x faster than CPU for this model). Do not run extraction and training concurrently unless measured to be safe.
