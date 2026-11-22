# AI Disclosure

This document records how AI was used to build EchoGesture. The competition's own AI-disclosure rules were not available in the project files or the materials we could locate; only the track solution guide (`EchoGesture_Solution_Guide.pdf`) was found. If the organisers specify a format or extra items, this file should be updated to match.

## 1. AI tools used during development

| Tool | Used for |
|---|---|
| **Claude Code** (Anthropic's coding agent, in the Claude desktop app), running the Claude Opus 5.5 model | Writing and editing code, running commands, debugging, analysis, documentation, the presentation and the submission tooling |

- **Other tools:** we have not recorded the use of any other AI coding assistant. *To be confirmed by the team before submission* (for example ChatGPT, GitHub Copilot or Gemini, if any were used outside this workflow).
- **Commit attribution:** commits made by the agent carry a `Co-Authored-By: Claude` trailer. Some commits in the history were made under a placeholder Git identity (`Your Name`) configured on the development machine.

**AI models inside the product (not development assistance):**
- **MediaPipe Holistic** (Google): a pretrained landmark detector, used unchanged and not trained by us.
- **The EchoGesture temporal 1D-CNN:** trained by us on WLASL-2000.

## 2. How AI assisted

- **Coding:**
  - **Data:** dataset inspection, subset selection and keypoint extraction (resumable, multi-process), plus keypoint validation.
  - **Model:** preprocessing and in-graph features, augmentation, the 1D-CNN, and the training and evaluation scripts.
  - **Deployment:** ONNX export and INT8 quantization; inference (`main.py`, `src/inference.py`), the live streaming recogniser and HUD, and the Gradio web app (`app.py`).
  - **Tooling:** benchmarks, the energy meter, submission packaging and verification, the Hugging Face Space builder, and the presentation build scripts.
- **Debugging.** Problems found and fixed during development include:
  - 12 NSLT IDs missing from the master JSON;
  - resume logic that skipped orphaned arrays;
  - float64 being unsupported on Apple MPS;
  - an ONNX export failure at batch sizes other than 1;
  - an RNG-state crash when resuming on MPS;
  - a false caption from brief hand flicker in the live segmenter;
  - Gradio requiring ffmpeg for uploads.
- **Analysis:**
  - dataset statistics;
  - signer-overlap analysis;
  - bootstrap confidence intervals;
  - the confidence-calibration analysis;
  - the proposal for improving accuracy.
- **Documentation:** the READMEs, compliance report, this disclosure, the deliverables checklist, the video script and the presentation text.
- **Research:**
  - reading the track guide;
  - checking library compatibility (Gradio, MediaPipe and ONNX Runtime wheels for Linux and Python 3.11).

  Literature figures in the documents are cited from the WLASL paper as quoted in the track guide.

## 3. What was implemented and tested in the actual project

Every result reported in the README, the presentation, `results.csv` and the reports was **measured by running the project's code** on this machine (MacBook Air M4). None was estimated or written by the AI without a run.

| Component | Verification performed |
|---|---|
| Keypoint extraction | 10,000/10,000 videos extracted; validation report with 0 failed or corrupted |
| Training and evaluation | Training logged per epoch; test set evaluated once (top-1 18.90%, top-5 42.60%) |
| ONNX export | FP32 matches PyTorch (max logit diff 1.1e-5); INT8 accuracy measured |
| `main.py` / submission package | 2,000/2,000 test predictions identical to the stored evaluation; runs in a clean Python 3.11 env offline |
| Web app (`app.py`) | Upload tested in a browser; same top-5 and scores as `main.predict()` (12/12 direct, 4/4 over HTTP, 3/3 from a clean Space build); error handling tested |
| Live HUD and streaming | 10.5-minute real-time replay benchmark (latency, drift, power) |
| Tests | `pytest` suite (see `docs/Deliverables_Status.md` for the latest count) |

**Not verified:**
- webcam recording (camera access was unavailable to the agent);
- public hosting (not deployed);
- any phone deployment.

## 4. How AI-generated code was reviewed and validated

- **Automated checks:**
  - every change was linted with Ruff (an edit hook runs `ruff` automatically);
  - code was tested with `pytest` before committing.
- **Parity checks** at each deployment stage:
  - PyTorch against ONNX;
  - ONNX against stored-array predictions;
  - `predict()` against the evaluation;
  - the app against `predict()`;
  - the packaged copy against the repository.
- **Data rules enforced in code:**
  - feature statistics come only from training data, with a fingerprint check;
  - the test set is checked as untouched during training;
  - calibration uses training clips only.
- **Stage-by-stage approval:**
  - the work ran in stages, each reported to the developer with its measured results and approved before the next began;
  - architecture, input dimensions and dataset changes were proposed and approved before implementation.

## 5. Human decisions and responsibilities

The developer:

- **Defined the scope:** the project's goals, constraints and targets, and chose WLASL-2000 and the 2,000-word vocabulary.
- **Chose the model:** the pure temporal 1D-CNN with 64 × 576 input; recurrent, attention and 3D models were rejected for the final system.
- **Chose the data and compute:**
  - a 10,000-video dataset (3/1/1 per word);
  - local-only training on the MacBook, with no Kaggle or cloud.
- **Approved each step before it ran:** extraction, preprocessing, training, the one-time test evaluation, ONNX/INT8 export, deployment, packaging, the presentation and pushing to GitHub.
- **Decided:**
  - not to retrain for the submission;
  - to deploy the FP32 model rather than INT8;
  - to report failing targets (accuracy, power) plainly.
- **Remains responsible for:**
  - the final submission;
  - team information;
  - recording the demo video;
  - deploying the public demo;
  - confirming this disclosure against the competition rules.
