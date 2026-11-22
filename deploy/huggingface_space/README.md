---
title: EchoGesture Sign Language Recognition
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Isolated ASL word recognition from MediaPipe landmarks
---

# EchoGesture – Sign Language Recognition

Upload or record a short video of one isolated ASL sign. The app extracts MediaPipe Holistic landmarks,
applies the project's preprocessing and runs the EchoGesture temporal 1D-CNN (FP32 ONNX, 3.33 MB) on the
Space's CPU, then shows the top-5 of 2,000 WLASL words with a timing breakdown.

- Source code, evaluation and documentation: https://github.com/shr1989/SAMSUNG_PRISM
- Measured test accuracy (2,000 WLASL-2000 clips): top-1 18.90%, top-5 42.60%. Predictions are often wrong;
  check the top-5 list. Results were measured on signers largely seen in training.
- This is isolated sign recognition (one word per video), not continuous translation.
- Uploaded videos are processed on the Space and are not stored by the app beyond Gradio's temporary cache,
  which it clears hourly.
