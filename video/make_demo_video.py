"""Assemble video/EchoGesture_Demo.mp4 from real app screen recordings and the presentation slides.

No footage is generated: the live segments are unedited screen recordings of app.py using a webcam
(only scaled to fit), and the remaining segments are pages of presentation/EchoGesture_Presentation.pdf
with a caption. Needs pymupdf, Pillow, numpy and imageio-ffmpeg.

    python video/make_demo_video.py
"""

from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PDF = ROOT / "presentation" / "EchoGesture_Presentation.pdf"
OUT = HERE / "EchoGesture_Demo.mp4"
W, H, FPS = 1920, 1080, 30
BAND = 120
FONT = "/System/Library/Fonts/Helvetica.ttc"
INK, TEAL = (20, 33, 43), (11, 60, 73)

# (kind, source, seconds, caption)
SEGMENTS = [
    ("slide", 1, 8, "EchoGesture recognises isolated American Sign Language words, on the device, from body and hand landmarks."),
    ("slide", 2, 10, "The task: one sign per clip, 2,000 possible words, no video leaving the device, answers fast enough for conversation."),
    ("slide", 4, 10, "MediaPipe turns each frame into 75 landmarks; a small temporal 1D-CNN classifies their motion."),
    ("slide", 9, 10, "The model: 866,960 parameters, 3.33 MB as FP32 ONNX, 0.92 MB as INT8."),
    ("slide", 12, 8, "The web app, the offline API and the live HUD all share the same inference code."),
    ("card", "Live demo: recording a sign with the webcam in the EchoGesture web app", 4, "Unedited screen recordings of app.py in a web browser on Windows, with signs performed live in front of a webcam."),
    ("clip", "webcam_demo_2.mp4", None, "Webcam recording -> MediaPipe -> 1D-CNN. Result: DOUBT, model score 77%, with the top-5 and timing shown."),
    ("clip", "webcam_demo_1.mp4", None, "A second recording: the model answers NICKEL (91% score). Scores are not calibrated probabilities."),
    ("slide", 14, 10, "On 2,000 held-out test clips: 18.90% top-1 and 42.60% top-5, far below the 70% target."),
    ("slide", 15, 10, "Model inference is about 0.5 ms. Live sign-end-to-word p95 is 32.2 ms, plus a deliberate 320 ms end-of-sign wait."),
    ("slide", 16, 8, "Hiding one hand costs 1.0 point (keypoints) to 4.75 points (video pixels)."),
    ("slide", 17, 10, "Five of seven targets are met; accuracy and power (1.42 W per-process CPU estimate) are not."),
    ("slide", 18, 10, "Limitations: signer overlap in the test set, imperfect calibration, isolated words only."),
    ("slide", 20, 8, "Next: more training data per word and evaluation on new signers. Code: github.com/shr1989/SAMSUNG_PRISM"),
]  # fmt: skip


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, face, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=face) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def with_caption(image: Image.Image, caption: str) -> np.ndarray:
    canvas = Image.new("RGB", (W, H), "white")
    area = (W, H - BAND)
    scale = min(area[0] / image.width, area[1] / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    canvas.paste(image, ((W - image.width) // 2, (H - BAND - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, H - BAND, W, H], fill=TEAL)
    face = font(34)
    lines = wrap(draw, caption, face, W - 160)
    y = H - BAND + (BAND - 44 * len(lines)) // 2
    for line in lines:
        draw.text((80, y), line, font=face, fill="white")
        y += 44
    return np.asarray(canvas)


def title_card(title: str) -> Image.Image:
    card = Image.new("RGB", (W, H - BAND), TEAL)
    draw = ImageDraw.Draw(card)
    face = font(64)
    lines = wrap(draw, title, face, W - 300)
    y = (H - BAND - 84 * len(lines)) // 2
    for line in lines:
        draw.text(((W - draw.textlength(line, font=face)) // 2, y), line, font=face, fill="white")
        y += 84
    return card


def clip_frames(path: Path):
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    width, height = meta["size"]
    for raw in reader:
        yield Image.frombytes("RGB", (width, height), raw)


def main() -> None:
    pdf = pymupdf.open(PDF)
    writer = imageio_ffmpeg.write_frames(
        str(OUT), (W, H), fps=FPS, codec="libx264", quality=None, macro_block_size=1,
        pix_fmt_out="yuv420p", output_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart"],
    )  # fmt: skip
    writer.send(None)
    total = 0
    for kind, source, seconds, caption in SEGMENTS:
        if kind == "clip":
            for image in clip_frames(HERE / source):
                writer.send(with_caption(image, caption).tobytes())
                total += 1
            continue
        if kind == "slide":
            pix = pdf[source - 1].get_pixmap(dpi=150)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            image = title_card(source)
        frame = with_caption(image, caption).tobytes()
        for _ in range(seconds * FPS):
            writer.send(frame)
            total += 1
    writer.close()
    print(f"{OUT} {total / FPS:.1f} s, {OUT.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
