import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.inference import DEFAULT_MODEL
from src.streaming import SegmenterConfig, StreamingRecognizer

WINDOW = "EchoGesture"
GREEN, WHITE, GREY, AMBER = (80, 220, 80), (255, 255, 255), (170, 170, 170), (0, 190, 255)


def draw_landmarks(frame: np.ndarray, keypoints: np.ndarray | None) -> None:
    if keypoints is None:
        return
    height, width = frame.shape[:2]
    colours = [(config.LEFT_HAND_SLICE, (255, 140, 0)), (config.RIGHT_HAND_SLICE, (0, 140, 255))]
    for part, colour in [(slice(0, 23), GREY), *colours]:
        for x, y, _ in keypoints[part]:
            if np.isfinite(x):
                cv2.circle(frame, (int(x * width), int(y * height)), 3, colour, -1)


def put(frame, text, origin, scale, colour, thickness=2) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def draw_hud(frame: np.ndarray, stream: StreamingRecognizer, last, fps: float) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 110), (width, height), (0, 0, 0), -1)
    caption = " ".join(stream.captions) or "..."
    put(frame, caption[-40:], (16, height - 60), 1.4, WHITE, 3)
    state = "SIGNING" if stream.signing else "READY"
    put(frame, state, (16, 36), 0.8, AMBER if stream.signing else GREEN)
    stats = f"{fps:4.1f} FPS | landmarks {stream.last_landmarks_ms:4.1f} ms"
    if last is not None:
        stats += f" | sign->word {last.latency_ms['end_to_end_ms']:4.1f} ms"
    put(frame, stats, (16, height - 20), 0.55, GREY, 1)
    put(
        frame,
        f"threshold {stream.segmenter.threshold:.2f}  [ ] adjust  c clear  q quit",
        (width - 430, 36),
        0.5,
        GREY,
        1,
    )
    if last is None:
        return
    colour = GREEN if last.committed else GREY
    put(frame, f"{last.word}  {last.confidence:.0%}", (16, 80), 1.0, colour)
    for rank, alt in enumerate(last.top5[1:4], start=1):
        put(
            frame,
            f"{alt['word'].lower()} {alt['confidence']:.0%}",
            (16, 80 + 30 * rank),
            0.6,
            GREY,
            1,
        )


def open_source(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        hint = (
            ""
            if args.video
            else (
                " Allow camera access for your terminal app in "
                "System Settings > Privacy & Security > Camera."
            )
        )
        raise SystemExit(
            f"cannot open {'video ' + str(args.video) if args.video else 'camera'}.{hint}"
        )
    return cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="replay a video file instead of the camera")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=SegmenterConfig.threshold)
    args = parser.parse_args()

    stream = StreamingRecognizer(args.model, SegmenterConfig(threshold=args.threshold))
    cap = open_source(args)
    last, frame_times = None, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            # MediaPipe sees the unmirrored frame (as in training); only the display is mirrored.
            prediction = stream.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), now)
            last = prediction or last
            draw_landmarks(frame, stream.latest_keypoints)
            frame = cv2.flip(frame, 1)
            frame_times = [t for t in frame_times if now - t < 1.0] + [now]
            draw_hud(frame, stream, last, float(len(frame_times)))
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                stream.captions.clear()
            if key in (ord("["), ord("]")):
                delta = 0.05 if key == ord("]") else -0.05
                stream.segmenter.threshold = float(
                    np.clip(stream.segmenter.threshold + delta, 0, 1)
                )
    finally:
        cap.release()
        stream.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
