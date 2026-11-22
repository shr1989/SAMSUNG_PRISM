"""EchoGesture web demo: isolated ASL sign recognition in the browser (Gradio).

Runs the real pipeline: video -> MediaPipe Holistic (every 2nd frame) -> src.preprocess ->
FP32 ONNX temporal 1D-CNN -> top-5 of 2,000 WLASL words. Nothing leaves the machine that hosts it.

    python app.py            # http://127.0.0.1:7860
"""

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.extract_keypoints import create_holistic, extract_video  # noqa: E402
from src.inference import DEFAULT_MODEL, Recognizer  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MAX_SECONDS = 15.0
MAX_FRAMES = 600
MIN_SAMPLES = 6
MAX_FILE_MB = 50


class VideoError(ValueError):
    """A user-facing problem with the submitted video."""


class SignVideo(gr.Video):
    """gr.Video that hands the received file over unchanged.

    The stock component re-encodes uploads with ffmpeg to drop audio; the model ignores audio, so
    skipping that step removes the ffmpeg system dependency. Webcam recording stays video-only.
    """

    is_template = True

    def preprocess(self, payload):
        if payload is None:
            return None
        if not payload.path:
            raise ValueError("Payload path missing")
        return str(Path(payload.path))


@dataclass
class Analysis:
    top5: list[dict]
    timing_ms: dict
    frames: int
    samples: int
    fps: float
    pose_rate: float
    hand_rate: float


@lru_cache(maxsize=1)
def recognizer() -> Recognizer:
    return Recognizer(DEFAULT_MODEL)


def probe(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("This file could not be opened as a video. Try an MP4 or WebM file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames <= 0:
            # Browser recordings (WebM) often report no frame count, so count them.
            frames = 0
            while frames <= MAX_FRAMES and cap.grab():
                frames += 1
    finally:
        cap.release()
    if frames <= 0:
        raise VideoError("The video contains no readable frames.")
    return frames, fps


def validate(video_path: str | None) -> tuple[Path, int, float]:
    if not video_path:
        raise VideoError("Please upload a video or record one with the webcam first.")
    path = Path(video_path)
    if not path.is_file():
        raise VideoError("The uploaded file could not be found. Please upload it again.")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise VideoError(
            f"Unsupported file type {path.suffix or '(none)'}. Use one of: "
            + ", ".join(sorted(VIDEO_SUFFIXES))
        )
    if path.stat().st_size == 0:
        raise VideoError("The uploaded file is empty.")
    if path.stat().st_size > MAX_FILE_MB * 2**20:
        raise VideoError(
            f"The file is larger than {MAX_FILE_MB} MB. Please trim it to a single sign."
        )
    frames, fps = probe(path)
    seconds = frames / fps if fps > 0 else None
    if frames > MAX_FRAMES or (seconds is not None and seconds > MAX_SECONDS):
        raise VideoError(
            f"The video is too long. Record a single sign of at most {MAX_SECONDS:.0f} seconds."
        )
    return path, frames, fps


def analyse(video_path: str | None) -> Analysis:
    """Validate a video and run the deployed pipeline on it (same steps as main.predict)."""
    started = time.perf_counter()
    path, frames, fps = validate(video_path)
    model = recognizer()
    began = time.perf_counter()
    try:
        with create_holistic() as holistic:
            keypoints = extract_video(holistic, path, 0, frames, model.cfg.frame_stride)
    except (OSError, ValueError) as error:
        raise VideoError(f"The video frames could not be decoded ({error}).") from error
    landmarks_ms = 1000 * (time.perf_counter() - began)
    if len(keypoints) < MIN_SAMPLES:
        raise VideoError(
            "The video is too short. Record the whole sign (at least about half a second)."
        )
    present = ~np.isnan(keypoints[..., 0])
    pose_rate = float(present[:, : config.NUM_POSE].any(axis=1).mean())
    hands = present[:, config.LEFT_HAND_SLICE].any(axis=1) | present[
        :, config.RIGHT_HAND_SLICE
    ].any(axis=1)
    hand_rate = float(hands.mean())
    if pose_rate == 0:
        raise VideoError(
            "No person was detected. Make sure your upper body is in view and well lit."
        )
    if hand_rate == 0:
        raise VideoError("No hands were detected. Keep both hands inside the frame while signing.")
    top5, timing = model.classify(keypoints)
    timing = {
        "landmarks_ms": landmarks_ms,
        **timing,
        "total_ms": 1000 * (time.perf_counter() - started),
    }
    return Analysis(top5, timing, frames, len(keypoints), fps, pose_rate, hand_rate)


def result_card(analysis: Analysis) -> str:
    best = analysis.top5[0]
    return (
        '<div class="eg-result">'
        '<div class="eg-result-label">Predicted sign</div>'
        f'<div class="eg-result-word">{escape(best["word"])}</div>'
        f'<div class="eg-result-conf">model score {best["confidence"]:.1%} · rank 1 of 2,000 words</div>'
        "</div>"
    )


def details(analysis: Analysis) -> str:
    t = analysis.timing_ms
    warning = ""
    if analysis.hand_rate < 0.3:
        warning = (
            "\n\n**Note:** hands were found in few frames, so this prediction is less reliable."
        )
    return (
        "| Stage | Time |\n|---|---|\n"
        f"| Model inference (ONNX, FP32) | **{t['model_ms']:.2f} ms** |\n"
        f"| Preprocessing (normalise, resample to 64 frames) | {t['preprocess_ms']:.2f} ms |\n"
        f"| Landmark extraction (decode + MediaPipe, {analysis.samples} sampled frames) | {t['landmarks_ms']:.0f} ms |\n"
        f"| **Total processing for this video** | **{t['total_ms']:.0f} ms** |\n\n"
        f"Frames in video: {analysis.frames} · sampled every 2nd frame: {analysis.samples} · "
        f"body found in {analysis.pose_rate:.0%} and a hand in {analysis.hand_rate:.0%} of sampled frames."
        f"{warning}\n\n"
        "Scores are the model's softmax outputs. They are estimates, not calibrated probabilities."
    )


def run(video_path: str | None):
    try:
        analysis = analyse(video_path)
    except VideoError as error:
        raise gr.Error(str(error), duration=None) from error
    scores = {p["word"]: p["confidence"] for p in analysis.top5}
    status = f"Done: analysed {analysis.samples} sampled frames in {analysis.timing_ms['total_ms'] / 1000:.1f} s."
    return result_card(analysis), scores, details(analysis), status


HEADER = """
<div class="eg-header">
  <div class="eg-title">EchoGesture – Sign Language Recognition</div>
  <div class="eg-subtitle">An AI-powered isolated sign-language recognition system using temporal hand and body landmarks.</div>
  <div class="eg-chips">
    <span>2,000 ASL words (WLASL-2000)</span><span>MediaPipe landmarks</span>
    <span>Temporal 1D-CNN · 3.3 MB ONNX</span><span>Runs locally, no cloud API</span>
  </div>
</div>
"""

HOW_TO = """
**How to use**
1. Upload a short video of **one** isolated ASL sign, or switch to the webcam tab and record yourself.
2. Press **Recognise sign** and wait a few seconds while landmarks are extracted.
3. Read the predicted word, the top-5 alternatives and the timing breakdown.

**For the best landmark detection**
- Face the camera with your head, shoulders and both hands in view; leave space around your hands.
- Use even, bright lighting and a plain background; avoid strong backlight.
- Start with your hands down, sign once at a natural pace, then lower your hands.
- Keep the clip under 15 seconds (1-4 seconds is typical for one sign).
- If the preview says *Video not playable*, your browser cannot preview that codec (for example MPEG-4 Part 2); recognition still works.
"""

ABOUT_MODEL = f"""
**Pipeline.** Every 2nd frame goes through MediaPipe Holistic (33 pose + 2 × 21 hand landmarks, x/y/z).
Landmarks are centred on the shoulders, scaled by shoulder width, short gaps are filled, and the clip is
resampled to 64 frames. Inside the ONNX graph each frame becomes 576 features (225 positions, 126 upright
hand-shape features, 225 velocities), standardised with training-set statistics.

**Model.** A pure temporal 1D-CNN (866,960 parameters): Linear 576→192, a residual convolution block,
a stride-2 residual block, temporal mean pooling and a 2,000-way classifier. The demo runs
`{DEFAULT_MODEL.name}` (FP32, 3.33 MB) with ONNX Runtime on CPU; an INT8 version (0.92 MB) also exists.

**Measured results (2,000 held-out test clips, 1 per word).** Top-1 18.90 %, top-5 42.60 %, macro F1 0.1459.
Model inference p95 0.51 ms; live sign-end-to-word p95 32.2 ms on an Apple M4 CPU.
"""

DATASET = """
**WLASL-2000** (Li et al., *Word-level Deep Sign Language Recognition from Video*, WACV 2020): 21,095 videos of
2,000 ASL words by 120 signers. EchoGesture was trained on a 10,000-video subset (3 train / 1 val / 1 test per
word) from the official splits. WLASL is licensed for academic, computational use (C-UDA), so its videos are
not redistributed with this demo.
"""

LIMITATIONS = """
- **Accuracy is limited:** 18.90 % top-1 over 2,000 words, far below a 70 % target. Expect frequent mistakes;
  check whether the right word is in the top 5.
- **Signer overlap:** almost all test clips come from signers seen in training, so accuracy for new people
  (including you) has not been measured and is likely lower.
- **Isolated signs only:** one word per video. This is not continuous sign-language translation.
- **Scores are not calibrated probabilities.**
- **Camera conditions matter:** if MediaPipe misses the hands (blur, darkness, hands out of frame), the
  model has little to work with.
- **Recording direction:** the model was trained on unmirrored camera video. The webcam recorder here is
  set not to mirror; if your browser still mirrors the recording, left and right hands are swapped.
"""

CSS = """
.eg-header {background: #0B3C49; color: #fff; border-radius: 14px; padding: 28px 30px 22px;}
.eg-header * {color: #fff !important;}
.eg-title {font-size: 30px; font-weight: 700; letter-spacing: -0.01em;}
.eg-header .eg-subtitle {font-size: 16px; color: #CFE3E5 !important; margin-top: 6px;}
.eg-chips {margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px;}
.eg-chips span {background: rgba(255,255,255,0.12); border-radius: 999px; padding: 4px 12px; font-size: 13px;}
.eg-result {border: 1px solid #0B7A75; border-radius: 12px; padding: 18px 20px; text-align: center;}
.eg-result-label {font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: #6B7C88;}
.eg-result-word {font-size: 44px; font-weight: 800; color: #0B7A75; line-height: 1.15; word-break: break-word;}
.eg-result-conf {font-size: 14px; color: #6B7C88;}
.eg-empty {color: #6B7C88; text-align: center; padding: 26px 10px; border: 1px dashed #B8C7CC; border-radius: 12px;}
"""

EMPTY = '<div class="eg-empty">The predicted sign will appear here.</div>'


def build() -> gr.Blocks:
    with gr.Blocks(
        title="EchoGesture – Sign Language Recognition",
        analytics_enabled=False,
        delete_cache=(1800, 3600),
    ) as demo:
        gr.HTML(HEADER)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                video = SignVideo(
                    label="Sign video (upload or record)",
                    sources=["upload", "webcam"],
                    format=None,
                    include_audio=False,
                    webcam_options=gr.WebcamOptions(mirror=False),
                    height=360,
                )
                with gr.Row():
                    submit = gr.Button("Recognise sign", variant="primary", scale=3)
                    clear = gr.ClearButton(value="Clear", scale=1)
                status = gr.Markdown("Upload a video or record one to begin.")
                gr.Markdown(HOW_TO)
            with gr.Column(scale=4):
                card = gr.HTML(EMPTY)
                top5 = gr.Label(label="Top-5 predictions (model scores)", num_top_classes=5)
                info = gr.Markdown()
        with gr.Accordion("About the model", open=False):
            gr.Markdown(ABOUT_MODEL)
        with gr.Accordion("Dataset", open=False):
            gr.Markdown(DATASET)
        with gr.Accordion("Limitations", open=True):
            gr.Markdown(LIMITATIONS)
        submit.click(
            lambda: "Processing: extracting landmarks and running the model...",
            None,
            status,
            queue=False,
        ).then(run, video, [card, top5, info, status])
        clear.add([video, top5, info])
        clear.click(
            lambda: (EMPTY, "Upload a video or record one to begin."),
            None,
            [card, status],
            queue=False,
        )
    return demo


def main() -> None:
    recognizer()
    demo = build()
    demo.queue(default_concurrency_limit=1, max_size=16)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        max_file_size=f"{MAX_FILE_MB}mb",
        css=CSS,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        footer_links=[],
    )


if __name__ == "__main__":
    main()
