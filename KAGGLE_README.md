# Running EchoGesture on Kaggle

`EchoGesture_Kaggle.ipynb` runs the full pipeline on Kaggle: dataset validation, MediaPipe keypoint extraction, preprocessing, training of the pure temporal 1D-CNN, test evaluation, ONNX export, INT8 quantization, benchmarking and artifact packaging. The final models run locally on a Mac with ONNX Runtime; nothing uses cloud inference.

The notebook has been checked for structure and syntax and its modules pass the project's unit tests on macOS. It has **not yet been executed on Kaggle**, so the first run is the real compatibility test (see "First run" below).

## Files

| File | Where it goes | Required |
|---|---|---|
| `EchoGesture_Kaggle.ipynb` | Kaggle: **File -> Import Notebook** | Yes |
| WLASL2000 Resized (`stas444/wlasl2000-resized`) | Kaggle: **Add Input -> Datasets** | Yes |
| `EchoGesture_Kaggle_Package.zip` | Kaggle: upload as a private **Dataset**, then **Add Input** | No: the notebook already contains every module. When attached, the notebook checks that the package and notebook code are identical. Keep it as the source bundle for local use. |
| Local `data/keypoints/stride2/` (zipped) | Kaggle: upload as a private **Dataset**, then **Add Input** | No: reuses keypoints already extracted on the Mac. The folder must keep the name `stride2` and contain `extraction_status.csv`. |

## Kaggle setup

1. **Create the notebook.** Go to kaggle.com -> Code -> New Notebook, then File -> Import Notebook and choose `EchoGesture_Kaggle.ipynb`.
2. **Attach the dataset.** Add Input -> search `wlasl2000-resized` -> add the dataset by stas444. The notebook finds it anywhere under `/kaggle/input` by looking for `nslt_2000.json`, `WLASL_v0.3.json`, `wlasl_class_list.txt` and `videos/`.
3. **Set the accelerator.** Settings -> Accelerator -> **GPU T4 x2** (or any NVIDIA GPU). Do not choose TPU.
   - The P100 may be unsupported by recent PyTorch builds. The notebook tests CUDA with a real operation and falls back to CPU if it fails, so prefer the T4.
4. **Enable the internet.** Settings -> Internet -> On. It is needed to install MediaPipe 0.10.14 and a Python 3.11 runtime through `uv`, and requires a phone-verified account.
5. **Optional extras.** Attach `EchoGesture_Kaggle_Package.zip` and/or your local keypoints as described above.
6. **Check the settings cell** in Section 1. Defaults:
   - `SUBSET = "full"`: all 21,095 videos. `"reduced"` uses the documented 12,069-video subset.
   - `FRAME_STRIDE = 2`: every 2nd frame. On 36 videos (16 + 20) this matched full-frame hand detection (98-101% of detections, mean coordinate difference about 0.003). `1` extracts every frame and takes about twice as long.
   - `NUM_FRAMES = 64` and `ARCH = "tcnn"`: the 64 x 576 baseline and the pure temporal 1D-CNN.
   - `TRAINING_MINUTES = 120` and `RESERVE_MINUTES = 40`: time reserved within the 12-hour session.

## First run (recommended)

Run Sections 1-5.1 interactively, which takes a few minutes. They install dependencies and print versions and GPU details. They then validate the dataset and run three checks:
- That the regenerated manifests match the development machine's splits and labels.
- A MediaPipe parity test on 3 reference videos.
- The 5-video tests: (75, 3) frames, missing hands, landmark ordering, saved arrays, resume and an interrupted run.

Fix anything that fails here before starting the long run.

## Long run and resuming

`/kaggle/working` is deleted when an interactive session ends. To keep results:

1. **Save Version -> Save & Run All (Commit).** The whole notebook runs in the background, for up to Kaggle's session limit, and everything in `/kaggle/working` is stored as that version's output.
   - Full extraction stops itself before the time limit (`--max-minutes`). It leaves time for training when possible and marks itself incomplete otherwise.
   - If your account offers **Settings -> Persistence -> Files**, it also keeps `/kaggle/working` between interactive sessions.
2. **To continue an incomplete run:** Add Input -> **Notebook Output Files** -> select the previous version of this notebook, then commit again.
   - Section 5 restores keypoint arrays (matched by the `stride2` folder name).
   - Section 10 resumes training from `last.pt`. Keep `EPOCHS` and `BATCH_SIZE` unchanged when resuming, because the learning-rate schedule depends on them.
3. When extraction is complete the notebook continues automatically: preprocessing, training, evaluation, export, INT8 and benchmarking.

MediaPipe extraction is CPU-bound; a GPU does not make it faster. Extraction speed on Kaggle has not been measured. The progress bar and the `videos/min` and ETA log lines report it during the run.

## Outputs

Everything is collected in `/kaggle/working/echogesture_output/` and `echogesture_output.zip` (Output tab -> download):

| Path | Contents |
|---|---|
| `models/checkpoints/tcnn/best.pt` | Best validation-macro-F1 PyTorch checkpoint, plus `history.csv` |
| `models/exported/model_fp32.onnx`, `model_int8.onnx` | ONNX models. The input is `keypoints` with shape (batch, 64, 75, 3) |
| `data/processed/preprocess_config.json` | Preprocessing parameters used for every split and for inference |
| `class_mapping.json` | Label index -> gloss |
| `results/results.csv` | Clean, occluded and rotated metrics for val and test |
| `results/reports/` | Dataset, selection, extraction, preprocessing, training, evaluation, export (quantization) and benchmark reports, per-class test results, confusion matrix and failures |
| `results/reports/quantization_comparison.csv` | FP32 vs INT8 accuracy |
| `results/plots/` | Training curves |
| `requirements-*-frozen.txt` | Exact package versions of both environments |
| `src/`, `kaggle/`, `main.py` | Code needed for local inference |

The quantization report lists which operators run in INT8 and which remain FP32: LayerNorm, the in-graph feature computation and pooling stay FP32. It sets `qat_recommended` when INT8 loses more than 2 percentage points of validation top-1.

## Local inference on the Mac

```bash
unzip echogesture_output.zip -d echogesture_output
cd echogesture_output
/Users/shraddhasidhan/hack/EchoGesture/.venv/bin/python main.py /path/to/sign.mp4
```

`main.py` prints JSON with the predicted uppercase word, the top 5 predictions with confidences, and latency split into MediaPipe, preprocessing and model time. Add `--model models/exported/model_int8.onnx` to use the INT8 model.

## Known limitations

- **Not yet run on Kaggle.** The Kaggle Python version, the CUDA driver and preinstalled package versions can change; the version-check cells report them.
- **Same pins, different OS.** MediaPipe on Linux x86_64 may differ numerically from macOS arm64 extraction. The Section 5 parity check quantifies this; keypoints from the Mac and from Kaggle should normally not be mixed unless the parity check passes.
- **Signer-dependent split.** The official WLASL split is signer-dependent: 99.97% of test videos come from signers seen in training, so test accuracy does not measure generalisation to new signers.
