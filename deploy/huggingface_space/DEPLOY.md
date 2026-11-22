# Deploying the EchoGesture demo to Hugging Face Spaces

This folder is the template for a public Gradio Space. Nothing has been deployed yet: creating the Space
needs your Hugging Face account.

## What was verified locally

| Check | Result |
|---|---|
| Dependencies resolve to Linux binary wheels (Python 3.11, manylinux x86_64) | yes: gradio 6.32.0, mediapipe 0.10.14, onnxruntime 1.32.0, numpy 1.26.4, pandas 3.0.6 |
| Built Space folder runs in a clean Python 3.11 env with only these requirements (no torch) | yes, macOS arm64 |
| Same top-5 as the repository's `main.predict()` (3 test videos through the HTTP API) | 3 / 3 |
| Works with outbound internet blocked (localhost only) | yes |
| Memory of the running app after several requests | about 490 MB RSS |
| Latency on Apple M4 CPU | about 0.5-1.5 s per 1-3 s clip, almost all MediaPipe |

**Not verified:** the Linux container itself (no Docker here), CPU speed on Spaces hardware, and browser
webcam recording (camera access was blocked in the test browser). The free "CPU basic" Space (2 vCPU,
16 GB RAM at the time of writing; check current limits) should fit the ~0.5 GB footprint, but MediaPipe
will run slower than on an M4, so expect several seconds per clip. Verify after deployment.

## Steps

1. Build the Space folder from the repository (copies `app.py`, the inference modules, the FP32 model,
   the preprocessing config and the label manifest; 4.3 MB):
   ```bash
   .venv/bin/python tools/build_space.py
   ```
2. On huggingface.co: **New Space** → SDK **Gradio** → hardware **CPU basic** → visibility **Public**.
3. Push the folder contents (replace `<user>`/`<space>`):
   ```bash
   cd build/huggingface_space
   git init && git add . && git commit -m "EchoGesture demo"
   git remote add origin https://huggingface.co/spaces/<user>/<space>
   git push origin HEAD:main
   ```
   The ONNX file is 3.3 MB, below the 10 MB limit for plain Git pushes, so Git LFS is not required.
4. Wait for the build log to show `Running on local URL`, open the Space, upload a short sign video and
   record one with the webcam.
5. Put the Space URL into `README.md` (Interactive demo) and `docs/Deliverables_Status.md`.

`packages.txt` installs `libgl1` and `libglib2.0-0`, which OpenCV (pulled in by MediaPipe) needs on the
Spaces Debian image. `README.md` pins `sdk_version: 6.32.0` and `python_version: "3.11"`.
