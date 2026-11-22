import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Environment overrides let the same code run on Kaggle (read-only input, writable working dir).
WORK_DIR = Path(os.environ.get("ECHOGESTURE_WORK_DIR", ROOT))

DATA_DIR = WORK_DIR / "data"
RAW_DIR = Path(os.environ.get("ECHOGESTURE_RAW_DIR", DATA_DIR / "raw" / "wlasl-complete"))
VIDEO_DIR = RAW_DIR / "videos"
MASTER_JSON = RAW_DIR / "WLASL_v0.3.json"
NSLT_JSON = RAW_DIR / "nslt_2000.json"
CLASS_LIST = RAW_DIR / "wlasl_class_list.txt"
# Every FRAME_STRIDE-th frame is extracted; stride 2 (12.5 fps) matched full-frame detection.
FRAME_STRIDE = int(os.environ.get("ECHOGESTURE_FRAME_STRIDE", 2))
KEYPOINT_DIR = Path(
    os.environ.get("ECHOGESTURE_KEYPOINT_DIR", DATA_DIR / "keypoints" / f"stride{FRAME_STRIDE}")
)
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"
REDUCED_MANIFEST_CSV = PROCESSED_DIR / "reduced_manifest.csv"
REDUCED_EXCLUDED_CSV = PROCESSED_DIR / "excluded_samples.csv"
TRAINING_SUBSET = os.environ.get("ECHOGESTURE_SUBSET", "full")
if TRAINING_SUBSET not in ("full", "reduced"):
    raise ValueError(f"ECHOGESTURE_SUBSET must be 'full' or 'reduced', got {TRAINING_SUBSET!r}")
TRAINING_MANIFEST_CSV = MANIFEST_CSV if TRAINING_SUBSET == "full" else REDUCED_MANIFEST_CSV

MODEL_DIR = WORK_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
EXPORT_DIR = MODEL_DIR / "exported"
DEFAULT_ARCH = os.environ.get("ECHOGESTURE_ARCH", "tcnn")

RESULTS_DIR = WORK_DIR / "results"
PLOT_DIR = RESULTS_DIR / "plots"
REPORT_DIR = RESULTS_DIR / "reports"
RESULTS_CSV = RESULTS_DIR / "results.csv"

NUM_CLASSES = 2000
SPLITS = ("train", "val", "test")
SEED = 42

NUM_POSE = 33
NUM_HAND = 21
NUM_LANDMARKS = NUM_POSE + 2 * NUM_HAND
LEFT_HAND_SLICE = slice(NUM_POSE, NUM_POSE + NUM_HAND)
RIGHT_HAND_SLICE = slice(NUM_POSE + NUM_HAND, NUM_LANDMARKS)
COORDS = 3

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

# Hand identity follows the signer's anatomy on unmirrored video, as reported by MediaPipe.
LANDMARK_ORDER = ("pose", "left_hand", "right_hand")

EXTRACTION_STATUS_CSV = KEYPOINT_DIR / "extraction_status.csv"
HOLISTIC_MODEL_COMPLEXITY = 1
HOLISTIC_MIN_DETECTION_CONFIDENCE = 0.5
HOLISTIC_MIN_TRACKING_CONFIDENCE = 0.5

# Baseline 64 frames per the original specification; 32 was an unvalidated experiment.
NUM_FRAMES = int(os.environ.get("ECHOGESTURE_NUM_FRAMES", 64))
# Interpolate gaps up to 0.32 s (8 frames at 25 fps).
MAX_FILL_GAP = 8 // FRAME_STRIDE
CLIP_VALUE = 6.0
MIN_SHOULDER_WIDTH = 0.02
# Hips, legs and feet (23-32) are mostly off-frame in WLASL and carry no sign information.
DROPPED_POSE = tuple(range(23, 33))
POSE_MIRROR_PAIRS = (
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
)
PREPROCESS_CONFIG_JSON = PROCESSED_DIR / "preprocess_config.json"


def ensure_dirs() -> None:
    for path in (
        KEYPOINT_DIR,
        PROCESSED_DIR,
        CHECKPOINT_DIR,
        EXPORT_DIR,
        PLOT_DIR,
        REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
