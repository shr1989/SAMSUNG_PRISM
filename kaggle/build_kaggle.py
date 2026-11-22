import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "EchoGesture_Kaggle.ipynb"
PACKAGE = ROOT / "EchoGesture_Kaggle_Package.zip"

PACKAGE_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/inspect_dataset.py",
    "src/select_subset.py",
    "src/extract_keypoints.py",
    "src/preprocess.py",
    "src/augment.py",
    "src/dataset.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/export_quantize.py",
    "src/inference.py",
    "src/benchmark.py",
    "main.py",
    "kaggle/kaggle_checks.py",
    "kaggle/reference.json",
    "kaggle/requirements-extraction.txt",
    "kaggle/requirements-training.txt",
    "requirements.txt",
    "pyproject.toml",
    "KAGGLE_README.md",
]
EXCLUDED_FROM_NOTEBOOK = {"requirements.txt", "pyproject.toml", "KAGGLE_README.md"}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def module(relative: str) -> dict:
    body = (ROOT / relative).read_text()
    return code(f"%%writefile {{PROJECT}}/{relative}\n{body}")


SETTINGS = '''
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

NOTEBOOK_START = time.time()
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJECT = WORK / "EchoGesture"
OUTPUT = WORK / "echogesture_output"
MP_VENV = WORK / "venv_mediapipe"
MP_PY = MP_VENV / "bin" / "python"
SYS_PY = Path(sys.executable)

SUBSET = "full"  # "full": all 21,095 videos; "reduced": documented 12,069-video subset
FRAME_STRIDE = 2  # 2: every 2nd frame (12.5 fps); 1: every frame (about 2x extraction time)
NUM_FRAMES = 64
ARCH = "tcnn"
EXTRACT_WORKERS = os.cpu_count() or 2
EPOCHS = 150
BATCH_SIZE = 64
PATIENCE = 20
SESSION_LIMIT_HOURS = 12
TRAINING_MINUTES = 120  # reserved for training when extraction runs in the same session
RESERVE_MINUTES = 40  # reserved for evaluation, export, benchmarking and saving
RUN_FULL_EXTRACTION = True
RUN_TRAINING = True
BENCHMARK_VIDEOS = 30

os.environ.update(
    {
        "ECHOGESTURE_WORK_DIR": str(PROJECT),
        "ECHOGESTURE_SUBSET": SUBSET,
        "ECHOGESTURE_FRAME_STRIDE": str(FRAME_STRIDE),
        "ECHOGESTURE_NUM_FRAMES": str(NUM_FRAMES),
        "ECHOGESTURE_ARCH": ARCH,
        "PYTHONPATH": str(PROJECT),
        "PYTHONUNBUFFERED": "1",
        "GLOG_minloglevel": "2",
    }
)
for directory in (PROJECT / "src", PROJECT / "kaggle", OUTPUT):
    directory.mkdir(parents=True, exist_ok=True)
KEYPOINT_DIR = PROJECT / "data" / "keypoints" / f"stride{FRAME_STRIDE}"
CHECKPOINT_DIR = PROJECT / "models" / "checkpoints" / ARCH
REPORT_DIR = PROJECT / "results" / "reports"
NOISE = (
    "W0000 ",
    "I0000 ",
    "All log messages before absl::InitializeLog",
    "Created TensorFlow Lite XNNPACK",
    "SymbolDatabase.GetPrototype",
    "warnings.warn(",
)


def run(*command, check: bool = True) -> int:
    """Run one pipeline stage in a subprocess and stream its progress output."""
    process = subprocess.Popen(
        [str(part) for part in command], cwd=PROJECT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    buffer = b""

    def emit(raw: bytes) -> None:
        line = raw.decode(errors="replace").rstrip()
        if line and not any(token in line for token in NOISE):
            print(line, flush=True)

    while chunk := process.stdout.read1(4096):
        *lines, buffer = re.split(rb"[\\r\\n]", buffer + chunk)
        for line in lines:
            emit(line)
    emit(buffer)
    code = process.wait()
    if check and code:
        raise RuntimeError(f"stage failed (exit {code}): {' '.join(map(str, command))}")
    return code


def minutes_left() -> float:
    return SESSION_LIMIT_HOURS * 60 - (time.time() - NOTEBOOK_START) / 60


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def resources() -> None:
    import psutil

    disk = shutil.disk_usage(WORK)
    memory = psutil.virtual_memory()
    print(
        f"disk free {disk.free / 2**30:.1f} GB | RAM available {memory.available / 2**30:.1f}"
        f" of {memory.total / 2**30:.1f} GB | session time left about {minutes_left():.0f} min"
    )
'''

HARDWARE = """
import platform

import torch

print(f"Python {platform.python_version()} | PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()} | CUDA runtime: {torch.version.cuda}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(
        f"GPU {index}: {props.name} | {props.total_memory / 2**30:.1f} GB"
        f" | compute capability {props.major}.{props.minor}"
    )
if not torch.cuda.is_available():
    print("No CUDA GPU visible: training falls back to PyTorch CPU.")
print(f"CPU cores: {os.cpu_count()} (MediaPipe extraction is CPU-bound regardless of GPU)")
resources()
"""

INSTALL = """
run(SYS_PY, "-m", "pip", "install", "--quiet", "uv", "-r", "kaggle/requirements-training.txt")
if not MP_PY.exists():
    run(SYS_PY, "-m", "uv", "venv", "--python", "3.11", MP_VENV)
run(
    SYS_PY, "-m", "uv", "pip", "install", "--quiet", "--python", MP_PY,
    "-r", "kaggle/requirements-extraction.txt",
)
"""

VERSION_CHECK = """
run(
    MP_PY, "-c",
    "import mediapipe as mp, numpy, cv2, pandas, onnxruntime, sys;"
    "mp.solutions.holistic.Holistic;"
    "print('MediaPipe env: Python', sys.version.split()[0], '| mediapipe', mp.__version__,"
    "'| numpy', numpy.__version__, '| opencv', cv2.__version__, '| pandas', pandas.__version__,"
    "'| onnxruntime', onnxruntime.__version__, '| legacy Holistic API available')",
)
run(
    SYS_PY, "-c",
    "import torch, onnx, onnxruntime, sklearn, numpy;"
    "print('Training env: torch', torch.__version__, '| onnx', onnx.__version__,"
    "'| onnxruntime', onnxruntime.__version__, '| scikit-learn', sklearn.__version__,"
    "'| numpy', numpy.__version__)",
)
"""

VERIFY_PACKAGE = """
EXPECTED_HASHES = __HASHES__
packages = [p.parent for p in INPUT_ROOT.rglob("package_hashes.json")]
for package in packages:
    stored = read_json(package / "package_hashes.json")
    differing = [name for name, digest in stored.items() if digest != EXPECTED_HASHES.get(name)]
    status = "identical to this notebook" if not differing else f"DIFFERS in {differing}"
    print(f"Attached package {package}: {status}")
if not packages:
    print("No EchoGesture package attached (optional); using the modules written by this notebook.")
"""

DISCOVER = """
def find_dirs(marker: str) -> list[Path]:
    found = []
    for root, dirs, files in os.walk(INPUT_ROOT, followlinks=True):
        dirs[:] = [d for d in dirs if d != "videos"]
        if marker in files:
            found.append(Path(root))
    return found


required = ("WLASL_v0.3.json", "nslt_2000.json", "wlasl_class_list.txt")
candidates = [
    d for d in find_dirs("nslt_2000.json")
    if (d / "videos").is_dir() and all((d / name).exists() for name in required)
]
if len(candidates) != 1:
    raise RuntimeError(
        f"Expected one WLASL-2000 directory under {INPUT_ROOT}, found {candidates}. "
        "Attach the WLASL2000 Resized dataset (see KAGGLE_README.md)."
    )
RAW_DIR = candidates[0]
os.environ["ECHOGESTURE_RAW_DIR"] = str(RAW_DIR)
video_count = sum(1 for _ in (RAW_DIR / "videos").glob("*.mp4"))
print(f"Dataset directory: {RAW_DIR}")
print(f"Annotation files: {', '.join(required)} | videos: {video_count}")
"""

INSPECT = """
run(MP_PY, "-m", "src.inspect_dataset", "--workers", EXTRACT_WORKERS)
report = read_json(REPORT_DIR / "dataset_report.json")
for key in (
    "videos_on_disk", "valid_videos", "excluded", "num_classes_master",
    "classes_with_valid_samples",
    "classes_without_test_samples", "split_counts", "samples_per_class", "duplicate_video_ids",
    "nslt_ids_missing_from_master", "class_list_mismatches",
):
    print(f"{key}: {report[key]}")
run(MP_PY, "kaggle/kaggle_checks.py", "manifests", "manifest")
"""

REDUCTION = """
import pandas as pd

run(MP_PY, "-m", "src.select_subset")
run(MP_PY, "kaggle/kaggle_checks.py", "manifests", "reduced_manifest")
processed = PROJECT / "data" / "processed"
full = pd.read_csv(processed / "manifest.csv", dtype={"video_id": str})
reduced = pd.read_csv(processed / "reduced_manifest.csv", dtype={"video_id": str})
excluded = pd.read_csv(processed / "excluded_samples.csv", dtype={"video_id": str})
summary = pd.DataFrame(
    {
        "full": full.groupby("split").size(),
        "reduced": reduced.groupby("split").size(),
        "excluded": excluded.groupby("split").size(),
        "classes_full": full.groupby("split")["label"].nunique(),
        "classes_reduced": reduced.groupby("split")["label"].nunique(),
    }
).fillna(0).astype(int)
display(summary)
print("Exclusion reasons:", excluded["reason"].value_counts().to_dict())
print("Official test clips removed by the reduction:", int(summary.loc["test", "excluded"]))
print("Reduced train clips per class:", reduced[reduced.split == "train"].groupby("label").size()
      .describe().round(2).to_dict())
print(f"SUBSET = {SUBSET!r}: training uses "
      f"{len(full) if SUBSET == 'full' else len(reduced)} videos")
"""

RESTORE = """
def restore_keypoints() -> None:
    sources = [d for d in find_dirs("extraction_status.csv") if d.name == KEYPOINT_DIR.name]
    KEYPOINT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = KEYPOINT_DIR / "extraction_status.csv"
    for source in sources:
        with (source / "extraction_status.csv").open() as handle:
            rows = [r for r in csv.DictReader(handle) if r["status"] == "ok"]
        copied = []
        for row in rows:
            target = KEYPOINT_DIR / f"{row['video_id']}.npy"
            if not target.exists() and (source / target.name).exists():
                shutil.copy2(source / target.name, target)
                copied.append(row)
        if copied:
            new_file = not status_path.exists()
            with status_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(copied[0]))
                if new_file:
                    writer.writeheader()
                writer.writerows(copied)
        print(f"Restored {len(copied)} keypoint arrays from {source}")
    if not sources:
        print(f"No previous {KEYPOINT_DIR.name} keypoints attached; extracting from scratch.")


def restore_checkpoints() -> None:
    for source in find_dirs("last.pt"):
        if source.name == ARCH and not (CHECKPOINT_DIR / "last.pt").exists():
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                shutil.copy2(item, CHECKPOINT_DIR / item.name)
            print(f"Restored training checkpoint from {source}")


restore_keypoints()
restore_checkpoints()
resources()
"""

PARITY = """
reference_stride = read_json(PROJECT / "kaggle" / "reference.json")["frame_stride"]
if reference_stride == FRAME_STRIDE:
    run(MP_PY, "kaggle/kaggle_checks.py", "parity")
else:
    print("Reference keypoints were extracted at a different stride; parity check skipped.")
"""

SAMPLE_TEST = """
SAMPLE_DIR = WORK / "sample_tests"


def sample_extract(limit: int, max_minutes: float | None = None) -> dict:
    command = [
        MP_PY, "-m", "src.extract_keypoints", "--limit", limit,
        "--workers", min(2, EXTRACT_WORKERS),
        "--out-dir", SAMPLE_DIR, "--report", SAMPLE_DIR / "report.json",
    ]
    if max_minutes:
        command += ["--max-minutes", max_minutes]
    run(*command)
    return read_json(SAMPLE_DIR / "report.json")


first = sample_extract(5)
assert first["succeeded"] == 5, first
run(MP_PY, "kaggle/kaggle_checks.py", "samples", SAMPLE_DIR)
assert sample_extract(5)["pending_at_start"] == 0, "resume did not skip finished videos"
assert sample_extract(8)["pending_at_start"] == 3, "resume did not extend the finished sample"
interrupted = sample_extract(14, max_minutes=0.05)
resumed = sample_extract(14)
assert resumed["succeeded"] == 14 and not list(SAMPLE_DIR.glob(".*tmp*")), resumed
print(f"Interruption test: stopped after {interrupted['videos_this_run']} new videos, "
      f"resumed {resumed['pending_at_start']} remaining, 14/14 complete, no temp files left.")
print("Pre-extraction tests passed.")
"""

FULL_EXTRACTION = """
extraction_minutes = minutes_left() - TRAINING_MINUTES - RESERVE_MINUTES
if RUN_FULL_EXTRACTION and extraction_minutes > 5:
    run(
        MP_PY, "-m", "src.extract_keypoints", "--workers", EXTRACT_WORKERS,
        "--max-minutes", round(extraction_minutes),
    )
else:
    print(f"Full extraction skipped (enabled={RUN_FULL_EXTRACTION}, "
          f"minutes available={extraction_minutes:.0f}).")
resources()
"""

EXTRACTION_SUMMARY = """
manifest_name = "manifest.csv" if SUBSET == "full" else "reduced_manifest.csv"
manifest = pd.read_csv(PROJECT / "data" / "processed" / manifest_name, dtype={"video_id": str})
status = pd.read_csv(KEYPOINT_DIR / "extraction_status.csv", dtype={"video_id": str})
status = status.drop_duplicates("video_id", keep="last")
merged = manifest.merge(status[["video_id", "status"]], on="video_id", how="left")
merged["status"] = merged["status"].fillna("pending")
display(merged.groupby(["split", "status"]).size().unstack(fill_value=0))
failures = status[status["status"] == "failed"][["video_id", "error"]]
failures.to_csv(REPORT_DIR / "extraction_failures.csv", index=False)
print(f"Failures: {len(failures)} (listed in results/reports/extraction_failures.csv)")
EXTRACTION_COMPLETE = bool((merged["status"] != "pending").all())
if (REPORT_DIR / "extraction_report.json").exists():
    report = read_json(REPORT_DIR / "extraction_report.json")
    print({k: v for k, v in report.items() if k != "failures"})
print("Extraction complete:" , EXTRACTION_COMPLETE)
if not EXTRACTION_COMPLETE:
    print("Save Version now, attach this version's output as input, and re-run to continue.")
"""

PREPROCESS = """
if EXTRACTION_COMPLETE:
    run(SYS_PY, "-m", "src.preprocess")
    print(json.dumps(read_json(REPORT_DIR / "preprocess_report.json"), indent=2))
    print(read_json(PROJECT / "data" / "processed" / "preprocess_config.json"))
else:
    print("Preprocessing waits for complete extraction.")
"""

AUGMENT_CHECK = '''
run(SYS_PY, "-c", """
import torch
from src import config
from src.augment import MIRROR_INDEX, augment, mirror, presence_mask
x = torch.randn(8, config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS)
x[:, :20, config.LEFT_HAND_SLICE] = 0
out = augment(x, torch.Generator().manual_seed(0))
assert out.shape == x.shape and torch.isfinite(out).all()
assert torch.equal(mirror(mirror(x)), x) and sorted(MIRROR_INDEX) == list(range(75))
print('augment output', tuple(out.shape), '| mirror swaps hands and is an involution')
print('fraction of landmarks present before/after:',
      round(presence_mask(x).mean().item(), 3), round(presence_mask(out).mean().item(), 3))
print('augmentation is applied only inside the training loop (src/train.py)')
""")
'''

DATASET_CHECK = '''
if EXTRACTION_COMPLETE:
    run(SYS_PY, "-c", """
from src.dataset import check_disjoint, load_split
splits = {name: load_split(name) for name in ('train', 'val', 'test')}
check_disjoint(*splits.values())
for name, data in splits.items():
    print(name, tuple(data.features.shape), 'classes', data.labels.unique().numel())
print('splits are disjoint by video ID')
""")
'''

MODEL_CHECK = '''
run(SYS_PY, "-c", """
import torch
from src import config
from src.model import build_model, count_parameters, size_mb
model = build_model(config.DEFAULT_ARCH).eval()
x = torch.randn(64, config.NUM_FRAMES, config.NUM_LANDMARKS, config.COORDS)
features = model.features(x)
logits = model(x)
print(model)
print('input', tuple(x.shape), '-> features', tuple(features.shape),
      '-> logits', tuple(logits.shape))
print(f'parameters {count_parameters(model):,} | size {size_mb(model):.2f} MB (FP32)')
assert features.shape[-1] == 576 and logits.shape == (64, config.NUM_CLASSES)
""")
'''

TRAIN = """
run(SYS_PY, "-c", "from src.train import select_device; print('Training device:', select_device())")
if EXTRACTION_COMPLETE and RUN_TRAINING:
    budget = max(5, minutes_left() - RESERVE_MINUTES)
    command = [
        SYS_PY, "-m", "src.train", "--arch", ARCH, "--epochs", EPOCHS, "--batch-size", BATCH_SIZE,
        "--patience", PATIENCE, "--time-budget-min", round(budget),
    ]
    if (CHECKPOINT_DIR / "last.pt").exists():
        command.append("--resume")
    run(*command)
else:
    print("Training skipped until extraction is complete.")
resources()
"""

VALIDATION = """
from IPython.display import Image

if (REPORT_DIR / f"training_report_{ARCH}.json").exists():
    training = read_json(REPORT_DIR / f"training_report_{ARCH}.json")
    print(json.dumps({k: training[k] for k in (
        "device", "mixed_precision", "parameters", "model_size_mb", "train_samples",
        "val_samples", "epochs_completed", "best_epoch", "best_val", "training_minutes",
        "peak_memory_mb",
    )}, indent=2))
    display(Image(str(PROJECT / "results" / "plots" / f"training_{ARCH}.png")))
"""

EVALUATE = """
if (CHECKPOINT_DIR / "best.pt").exists():
    run(SYS_PY, "-m", "src.evaluate", "--checkpoint", CHECKPOINT_DIR / "best.pt")
    display(pd.read_csv(PROJECT / "results" / "results.csv"))
    evaluation = read_json(REPORT_DIR / "evaluation_report.json")
    print(json.dumps({k: evaluation[k] for k in ("test_per_class_summary", "note")}, indent=2))
"""

EXPORT = """
if (CHECKPOINT_DIR / "best.pt").exists():
    run(SYS_PY, "-m", "src.export_quantize", "--checkpoint", CHECKPOINT_DIR / "best.pt")
    export = read_json(REPORT_DIR / "export_report.json")
    for split in ("val", "test"):
        print(split, "ONNX FP32 vs PyTorch:", export[split]["onnx_fp32_vs_torch"])
    print("FP32 size MB:", round(export["size_mb"]["onnx_fp32"], 2))
"""

QUANT = """
if (REPORT_DIR / "export_report.json").exists():
    export = read_json(REPORT_DIR / "export_report.json")
    print("Operator precision:", json.dumps(export["int8_quantization"], indent=2))
    rows = []
    for split in ("val", "test"):
        for variant in ("torch_fp32", "onnx_fp32", "onnx_int8"):
            rows.append({"split": split, "model": variant} | export[split][variant])
    display(pd.DataFrame(rows))
    print("INT8 vs FP32 agreement:", export["test"]["onnx_int8_vs_fp32"])
    print(f"Validation top-1 drop from INT8: {export['int8_val_top1_drop_pp']:.2f} pp "
          f"| QAT recommended: {export['qat_recommended']}")
    print("Sizes MB:", {k: round(v, 2) for k, v in export["size_mb"].items()})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "quantization_comparison.csv", index=False)
"""

BENCHMARK = """
if (PROJECT / "models" / "exported" / "model_fp32.onnx").exists():
    export = read_json(REPORT_DIR / "export_report.json")
    print("Model-only latency, batch 1 (ms):", json.dumps(export["latency_batch1_ms"], indent=2))
    run(MP_PY, "-m", "src.benchmark", "--videos", BENCHMARK_VIDEOS)
    test_video = RAW_DIR / "videos" / f"{manifest[manifest.split == 'test'].video_id.iloc[0]}.mp4"
    run(MP_PY, "main.py", test_video)
"""

SAVE = """
def copy_into_output(relative: str) -> None:
    source = PROJECT / relative
    if not source.exists():
        print(f"missing, not saved: {relative}")
        return
    target = OUTPUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    else:
        shutil.copy2(source, target)


for relative in (
    f"models/checkpoints/{ARCH}/best.pt",
    f"models/checkpoints/{ARCH}/history.csv",
    "models/exported/model_fp32.onnx",
    "models/exported/model_int8.onnx",
    "data/processed/preprocess_config.json",
    "data/processed/manifest.csv",
    "data/processed/reduced_manifest.csv",
    "data/processed/excluded_samples.csv",
    f"data/keypoints/{KEYPOINT_DIR.name}/extraction_status.csv",
    "results",
    "src",
    "kaggle",
    "main.py",
):
    copy_into_output(relative)

full = pd.read_csv(PROJECT / "data" / "processed" / "manifest.csv")
class_map = full.drop_duplicates("label").sort_values("label")
(OUTPUT / "class_mapping.json").write_text(
    json.dumps({int(r.label): r.gloss for r in class_map.itertuples()}, indent=2)
)
for name, python in (("training", SYS_PY), ("extraction", MP_PY)):
    frozen = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True)
    if frozen.returncode:
        frozen = subprocess.run(
            [str(SYS_PY), "-m", "uv", "pip", "freeze", "--python", str(python)],
            capture_output=True, text=True,
        )
    (OUTPUT / f"requirements-{name}-frozen.txt").write_text(frozen.stdout)
archive = shutil.make_archive(str(WORK / "echogesture_output"), "zip", OUTPUT)
print(f"Saved {archive} ({Path(archive).stat().st_size / 2**20:.1f} MB)")
for path in sorted(OUTPUT.rglob("*")):
    if path.is_file() and "src" not in path.parts and "kaggle" not in path.parts:
        print(f"  {path.relative_to(OUTPUT)}  {path.stat().st_size / 2**20:.2f} MB")
resources()
"""


def build_cells(hashes: dict) -> list[dict]:
    verify = VERIFY_PACKAGE.replace("__HASHES__", json.dumps(hashes, indent=4))
    return [
        md("""
# EchoGesture: WLASL-2000 sign recognition on Kaggle

MediaPipe Holistic keypoints -> 64 x 576 features -> pure temporal 1D-CNN -> 2,000 classes ->
ONNX -> INT8 ONNX. Every stage reuses the project modules written into
`/kaggle/working/EchoGesture`. Stages run as subprocesses so the MediaPipe environment
(numpy 1.26) and the training environment (Kaggle PyTorch) never conflict.
Results only appear after the notebook is executed; nothing below is pre-filled.
"""),
        md("## 1. Environment and hardware setup"),
        code(SETTINGS),
        code(HARDWARE),
        md("""
## 2. Dependency installation

MediaPipe 0.10.14 (the version used locally; legacy Holistic API) ships Linux wheels for
Python 3.9-3.12 and needs numpy < 2 and protobuf < 5, so it gets its own Python 3.11 venv
created with `uv`. Training uses Kaggle's preinstalled PyTorch. No TPU libraries are used.
"""),
        module("kaggle/requirements-extraction.txt"),
        module("kaggle/requirements-training.txt"),
        code(INSTALL),
        code(VERSION_CHECK),
        md("""
## 3. Dataset discovery and validation

Writes the configuration and inspection modules, locates the mounted WLASL-2000 dataset under
`/kaggle/input`, validates it, and checks that the official splits and labels match the
manifest generated on the development machine.
"""),
        module("src/__init__.py"),
        module("src/config.py"),
        module("src/inspect_dataset.py"),
        module("kaggle/reference.json"),
        module("kaggle/kaggle_checks.py"),
        code(verify),
        code(DISCOVER),
        code(INSPECT),
        md("""
## 4. Manifest inspection

The reduced manifest (12,069 videos) was created only to fit a 60-minute local run: it halves
training clips per class (minimum 3), keeps one validation clip per class and keeps every
official test clip. `SUBSET = "full"` (default) trains on all 21,095 videos.
"""),
        module("src/select_subset.py"),
        code(REDUCTION),
        md("""
## 5. MediaPipe extraction

33 pose + 21 left-hand + 21 right-hand landmarks, (x, y, z) each, saved per video as
`(T, 75, 3)` float32 with missing groups as NaN. Extraction is resumable: arrays are written
atomically, every video gets a status row, and finished videos are skipped. Previous progress
attached as a Kaggle input (a `stride2` keypoint folder or a checkpoint folder) is restored
first. MediaPipe runs on CPU; a GPU does not speed it up.
"""),
        module("src/extract_keypoints.py"),
        code(RESTORE),
        code(PARITY),
        md("### 5.1 Small-sample tests (5 videos, resume, interruption) before full extraction"),
        code(SAMPLE_TEST),
        md("### 5.2 Full extraction (time-limited; re-run the notebook to continue)"),
        code(FULL_EXTRACTION),
        code(EXTRACTION_SUMMARY),
        md("""
## 6. Preprocessing

Per clip: short-gap interpolation, shoulder-centred and shoulder-width-scaled coordinates,
hips and legs removed, values clipped, uniform resampling to 64 frames, missing landmarks set
to 0. The same function is used by training, evaluation and inference; its configuration is
saved to `data/processed/preprocess_config.json`.
"""),
        module("src/preprocess.py"),
        code(PREPROCESS),
        md("""
## 7. Augmentation (training only)

3D rotation (yaw +/-20 deg, roll +/-10 deg), scale and shift, mirroring with left/right
landmark swap, temporal crop and speed change, Gaussian jitter, one-hand occlusion for
10-40% of frames, random landmark dropout. Validation and test data are never augmented.
"""),
        module("src/augment.py"),
        code(AUGMENT_CHECK),
        md("""
## 8. Dataset and data loading

Each preprocessed split is held as one tensor on the training device and batched by index,
which avoids DataLoader worker processes; videos are never loaded into RAM.
"""),
        module("src/dataset.py"),
        code(DATASET_CHECK),
        md("""
## 9. Pure temporal 1D-CNN

Features per frame (computed in-graph after augmentation): 225 body coordinates +
126 wrist-relative upright handshape values + 225 velocities = 576. Network: Linear
576 -> 192, LayerNorm, ReLU, residual Conv1D (k=5), residual stride-2 Conv1D (k=5), each with
BatchNorm and ReLU, temporal average pooling, dropout, Linear -> 2,000. Feature
standardisation statistics are fitted on the training split only and stored in the model.
"""),
        module("src/model.py"),
        code(MODEL_CHECK),
        md("""
## 10. Training

Cross-entropy with label smoothing 0.1, AdamW (lr 0.002, weight decay 0.05), OneCycleLR,
batch 64, up to 150 epochs, early stopping on validation macro F1. CUDA mixed precision
(bfloat16 where native, else float16 with loss scaling). `last.pt` is saved every epoch and
training resumes from it automatically; `best.pt` holds the best validation macro F1.
"""),
        module("src/train.py"),
        code(TRAIN),
        md("## 11. Validation"),
        code(VALIDATION),
        md("""
## 12. Test evaluation

Run once on the untouched official test split after model selection: top-1, top-5,
precision, recall, macro F1, per-class table, confusion matrix, occlusion (one hand hidden in
30% of frames) and rotation robustness. The official split is signer-dependent.
"""),
        module("src/evaluate.py"),
        code(EVALUATE),
        md("## 13. ONNX export"),
        module("src/export_quantize.py"),
        code(EXPORT),
        md("""
## 14. INT8 quantization

Static post-training QDQ quantization calibrated on 300 training clips (never test data).
The report lists exactly which operators run in INT8 and which remain FP32.
"""),
        code(QUANT),
        md("""
## 15. Benchmarking and local inference

Model-only latency (20 warm-up + 200 runs, batch 1) comes from the export report. The
end-to-end benchmark separates MediaPipe landmark time, preprocessing and model time on test
videos. `main.py` shows the JSON returned by `predict(video_path)`.
"""),
        module("src/inference.py"),
        module("src/benchmark.py"),
        module("main.py"),
        code(BENCHMARK),
        md("""
## 16. Save downloadable artifacts

Collects checkpoints, ONNX models, preprocessing config, class mapping, reports, plots and
frozen requirements in `/kaggle/working/echogesture_output` and `echogesture_output.zip`.
`/kaggle/working` does not survive the session: use **Save Version -> Save & Run All** to keep
it as notebook output (see KAGGLE_README.md).
"""),
        code(SAVE),
    ]


def file_hashes() -> dict:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()[:16]
        for name in PACKAGE_FILES
        if name not in EXCLUDED_FROM_NOTEBOOK
    }


def build_notebook(hashes: dict) -> None:
    notebook = {
        "cells": build_cells(hashes),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{index:02d}"
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n")


def build_package(hashes: dict) -> None:
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in PACKAGE_FILES:
            archive.write(ROOT / name, f"EchoGesture/{name}")
        archive.write(NOTEBOOK, "EchoGesture/EchoGesture_Kaggle.ipynb")
        archive.writestr("EchoGesture/package_hashes.json", json.dumps(hashes, indent=2))


def format_notebook() -> None:
    ruff = Path(sys.executable).with_name("ruff")
    subprocess.run([ruff, "format", "--quiet", NOTEBOOK], check=True)
    subprocess.run([ruff, "check", "--quiet", NOTEBOOK], check=True)


def verify_embedded_modules() -> None:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    embedded = {}
    for cell in cells:
        source = "".join(cell["source"])
        if source.startswith("%%writefile {PROJECT}/"):
            header, _, body = source.partition("\n")
            embedded[header.removeprefix("%%writefile {PROJECT}/")] = body
    for name in PACKAGE_FILES:
        if name in EXCLUDED_FROM_NOTEBOOK:
            continue
        if embedded.get(name, "").rstrip("\n") != (ROOT / name).read_text().rstrip("\n"):
            raise ValueError(f"notebook copy of {name} differs from the source file")


def main() -> None:
    hashes = file_hashes()
    build_notebook(hashes)
    format_notebook()
    verify_embedded_modules()
    build_package(hashes)
    print(f"wrote {NOTEBOOK.name} and {PACKAGE.name} ({len(PACKAGE_FILES)} files)")


if __name__ == "__main__":
    main()
