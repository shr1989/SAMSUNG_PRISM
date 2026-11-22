from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "wlasl-complete"
VIDEO_DIR = RAW_DIR / "videos"
MASTER_JSON = RAW_DIR / "WLASL_v0.3.json"
NSLT_JSON = RAW_DIR / "nslt_2000.json"
CLASS_LIST = RAW_DIR / "wlasl_class_list.txt"
# Stride-2 arrays (12.5 fps); full-frame Phase 2 arrays remain in data/keypoints/.
KEYPOINT_DIR = DATA_DIR / "keypoints" / "stride2"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"
REDUCED_MANIFEST_CSV = PROCESSED_DIR / "reduced_manifest.csv"
REDUCED_EXCLUDED_CSV = PROCESSED_DIR / "excluded_samples.csv"

MODEL_DIR = ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
EXPORT_DIR = MODEL_DIR / "exported"

RESULTS_DIR = ROOT / "results"
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
FRAME_STRIDE = 2
HOLISTIC_MODEL_COMPLEXITY = 1
HOLISTIC_MIN_DETECTION_CONFIDENCE = 0.5
HOLISTIC_MIN_TRACKING_CONFIDENCE = 0.5

# Stride-2 clips have median 30 frames (p95 46); 32 avoids fabricating motion.
NUM_FRAMES = 32
MAX_FILL_GAP = 4
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
