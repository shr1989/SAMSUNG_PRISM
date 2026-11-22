# Deliverables status

- **Repository:** https://github.com/shr1989/SAMSUNG_PRISM (branch `main`)
- **Public demo URL:** not deployed yet. Files are prepared; see item 4.
- **Checklist source:** the competition checklist items 1-8 as given by the team. The original theme or "initial" slides were **not found** in the project or Downloads folder. Requirements below come from the track solution guide (`EchoGesture_Solution_Guide.pdf`, pages 2 and 25), which is the only organiser-side document available.

| # | Item | Status | Location |
|---|---|---|---|
| 1 | Submission package | Completed | `submission/EchoGesture_Submission.zip` |
| 2 | Source code | Completed | repository root, `src/`, `tests/` |
| 3 | Presentation | Completed | `presentation/EchoGesture_Presentation.pptx` and `.pdf` |
| 4 | Video / interactive demo | Video completed (see the gaps listed in section 4); demo app completed locally, not publicly deployed | `video/EchoGesture_Demo.mp4`, `app.py`, `deploy/huggingface_space/` |
| 5 | AI disclosure | Completed (team to confirm item 1 of the document) | `docs/AI_Disclosure.md` |
| 6 | README | Completed | `README.md` |
| 7 | APK / SDK | Not applicable | this document, section 7 |
| 8 | Other | Completed | results, reports, tools (section 8) |

## 1. Submission package: Completed

**File:** `submission/EchoGesture_Submission.zip` (10.99 MB, 58 files, including the demo video).

**Contents:**
- `main.py` with `predict(video_path)`;
- `app.py`;
- `src/`;
- the FP32 and INT8 ONNX models and the `best.pt` checkpoint;
- the preprocessing config, feature statistics and manifests;
- `requirements.txt` (Python 3.11) and `README.md`;
- `predictions.jsonl` / `predictions.json` (2,000 test clips) and `results.csv` (every metric vs target);
- the evaluation reports and `SHA256SUMS.txt`.

**What is excluded:** virtual environments, raw videos and datasets, caches and secrets.

**Validation:**
- `tools/verify_submission.py`:
  - the models, configs and 19 source modules are identical to the repository;
  - no absolute local paths and no dev-only requirements;
  - 2,000/2,000 predictions identical to the stored FP32 evaluation (top-1 18.90%).
- The ZIP was extracted into a fresh Python 3.11.15 environment, installed only from its `requirements.txt`, and run with outbound network blocked:
  - all checksums OK;
  - `main.py` works;
  - the app's inference gives the same top-5 as `predictions.jsonl`.

**Your action:** none required. Rename the ZIP if the organisers want a team name.

## 2. Source code: Completed

**Contents:**
- `src/`: extraction, preprocessing, features, model, training, evaluation, export/quantization, inference, streaming, benchmarks and the energy meter;
- the entry points `main.py`, `app.py` and `hud_demo.py`;
- `tests/`;
- `models/` (FP32 and INT8 ONNX, `best.pt`);
- `data/processed/`: preprocessing config, feature statistics and manifests, with no videos.

**Validation:**
- `pytest`: **65 passed** (2 warnings from third-party libraries).
- `ruff check` and `ruff format --check` are clean.
- `.gitignore` excludes `.venv`, raw data, keypoint arrays, caches, generated build folders and the Gradio stub `app.pyi`.

**Your action:** none.

## 3. Presentation: Completed

**Files:**
- `presentation/EchoGesture_Presentation.pptx` (20 slides);
- `presentation/EchoGesture_Presentation.pdf`;
- the build scripts: `build_presentation.js` and `make_charts.py`.

All figures are read from the result reports and the charts come from the real logs. Slide 13 shows a genuine screenshot of the app, including a wrong prediction.

**Validation:**
- the PPTX passes OOXML validation;
- the PDF was exported with Keynote;
- all 20 pages were rendered and inspected.

**Disclosures in the deck:**
- 18.90% top-1 and 42.60% top-5, below the 70% target;
- signer overlap;
- imperfect calibration;
- power as a per-process CPU estimate;
- the 320 ms confirmation wait is excluded from the 32.2 ms;
- webcam recording in the app is verified only by two recordings on Windows in Brave;
- isolated recognition only.

**Your action:** add team member names on slide 1 (not stored in the project).

## 4. Video / interactive demo: Video completed; demo app ready locally

**Video: completed.** `video/EchoGesture_Demo.mp4` (2 min 22 s, 1920 × 1080, H.264, 3.2 MB, silent with on-screen captions), built by `video/make_demo_video.py`. It contains:
- captioned slides from the presentation (problem, pipeline, model, architecture, results, latency, robustness, scorecard, limitations, conclusion);
- the two **unedited** webcam screen recordings of the web app (scaled only).

No footage was generated. Gaps against the guide's demo list (page 25):
- no segment of the live caption HUD with its FPS/latency overlay (the app's timing table is shown instead);
- no deliberately angled or partly hidden example;
- no voice-over.

The script in `docs/Demo_Video_Script.md` covers these if you want to re-record.

**Interactive demo app: completed and tested locally.** Run it with `python app.py` (Gradio 6.32.0, FP32 ONNX). It was tested as follows:
- upload and prediction in a real browser;
- same top-5 and scores as `main.predict()`:
  - 12/12 by direct call;
  - 4/4 over HTTP;
  - 3/3 from a clean Python 3.11 build of the Space folder with only its requirements and outbound network blocked;
- clear errors for missing, empty, wrong-type, undecodable, too-long, too-short and person-free videos;
- WebM decoding (the browser recording format);
- 25 repeated requests with no memory growth (peak RSS about 525 MB);
- about 490 MB RSS in the Space build.

**Webcam recording: verified in the app.** Two real screen recordings (Windows, Brave browser) show the app recording a sign from the webcam and returning a result: DOUBT (77% model score, 20 sampled frames in 6.3 s) and NICKEL (91%, 19 frames in 6.7 s). Saved as `video/webcam_demo_1.mp4` and `video/webcam_demo_2.mp4` (audio removed). The intended signs were not recorded alongside them, so they are not accuracy evidence. A collapsed "Error (2)" notice from an earlier attempt is visible in one recording; its content is not shown. The live HUD (`hud_demo.py`) is still untested with a camera.

**Public hosting: prepared, not deployed.**
- `deploy/huggingface_space/` holds the Space README with SDK and Python version, the requirements, `packages.txt` and `DEPLOY.md`.
- `python tools/build_space.py` assembles a 4.3 MB Space folder.
- Linux Python 3.11 binary wheels resolve for all dependencies.
- The Linux container itself and Spaces CPU speed are untested.

**Your actions:**
1. Optionally re-record the video with voice-over, the live HUD and an angled or partly hidden sign (`docs/Demo_Video_Script.md`).
2. Deploy the Space following `deploy/huggingface_space/DEPLOY.md` (needs your Hugging Face account).
3. Optionally test the live HUD (`python hud_demo.py`) with a camera.
4. Add the public URL to the README and to this file.

## 5. AI disclosure: Completed (pending one confirmation)

**File:** `docs/AI_Disclosure.md`. It covers:
- the tools used;
- how AI assisted;
- what was verified, and how;
- human decisions.

No competition-specific disclosure rules were found.

**Your action:** confirm whether any AI tool besides Claude Code was used, and adjust the document if the organisers require a specific format.

## 6. README: Completed

**File:** `README.md`. It covers:
- overview, problem, features and the architecture diagram (Mermaid);
- stack, dataset attribution and model details;
- installation, the app, webcam use, example output and tests;
- results, benchmarks, limitations, future work and structure;
- links, the submission package and team.

The README includes an honest correction: the example runs shown are misses.

**Validation:** all relative links were checked to point at existing files.

**Your action:** add team names and, after deployment, the public demo URL.

## 7. APK / SDK: Not applicable

EchoGesture is a Python application. Its entry points are:
- a Gradio web app (`app.py`);
- a command-line / Python API (`main.py`);
- a desktop webcam HUD (`hud_demo.py`).

**No Android APK, iOS app or SDK has been built, and none is provided.** The ONNX models could in future be run with ONNX Runtime Mobile on a phone, but that has not been implemented or measured. For the same reason, all latency and power figures come from a MacBook Air M4, not a phone.

**Your action:** none, unless the organisers require a mobile build.

## 8. Other: Completed

**Results and reports:**
- `results/results.csv` (evaluation table);
- `results/reports/`:
  - `compliance_report.md`;
  - the evaluation, export, benchmark, live benchmark and deployed-evaluation reports;
  - `calibration_analysis.json` (new: read-only analysis of the existing test predictions, ECE 0.173).

**Tools:**
- `tools/build_submission.py` and `tools/verify_submission.py`;
- `tools/build_space.py`;
- `tools/calibration_report.py`.

**Preserved unchanged:**
- `models/checkpoints/tcnn/best.pt` (SHA-256 `efb862e7…`) and `last.pt`;
- both ONNX models;
- all committed evaluation results.

There was no retraining.

**Your action:** none.
