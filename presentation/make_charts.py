"""Render the deck's charts from measured reports (images, so Keynote and PDF export show them)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "assets"
TEAL, BLUE, INK, MUTED, GRID = "#0B7A75", "#3A6EA5", "#14212B", "#6B7C88", "#D9E2E5"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": "#0B3C49",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def report(name: str) -> dict:
    return json.loads((ROOT / "results" / "reports" / name).read_text())


def training_curve() -> None:
    history = pd.read_csv(ROOT / "models/checkpoints/tcnn/history.csv")
    best = report("training_report_tcnn.json")["best_epoch"]
    fig, ax = plt.subplots(figsize=(7.4, 4.9), dpi=200)
    ax.plot(history["epoch"], 100 * history["val_top5"], color=BLUE, lw=2, label="Validation top-5")
    ax.plot(history["epoch"], 100 * history["val_top1"], color=TEAL, lw=2, label="Validation top-1")
    ax.axvline(best, color=MUTED, ls="--", lw=1)
    ax.text(best + 2, 46, f"selected epoch {best}", color=MUTED, fontsize=11)
    ax.set(xlabel="Epoch", ylabel="%", ylim=(0, 50), xlim=(0, len(history) - 1))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("Validation accuracy by epoch (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "training_curve.png", transparent=False, facecolor="white")
    plt.close(fig)


def test_accuracy() -> None:
    test = report("evaluation_report.json")["test"]
    names = ["Clean", "One hand occluded", "Rotated"]
    keys = ["clean", "occluded", "rotated"]
    fig, ax = plt.subplots(figsize=(7.2, 3.55), dpi=200)
    width = 0.36
    for offset, metric, colour, label in (
        (-width / 2, "top1", TEAL, "Top-1"),
        (width / 2, "top5", BLUE, "Top-5"),
    ):
        values = [100 * test[k][metric] for k in keys]
        bars = ax.bar([i + offset for i in range(3)], values, width, color=colour, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=3, color=INK, fontsize=12)
    ax.set_xticks(range(3), names, color=INK)
    ax.set(ylim=(0, 52))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Test accuracy, 2,000 clips (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "test_accuracy.png", facecolor="white")
    plt.close(fig)


def latency() -> None:
    export = report("export_report.json")
    live = report("live_benchmark.json")
    stream = live["prediction_latency_ms"]
    labels = ["Model only (FP32)", "Preprocess", "MediaPipe per frame", "Sign end to word"]
    values = [
        export["latency_batch1_ms"]["onnx_fp32"]["p95"],
        stream["preprocess_ms"]["p95"],
        live["per_sample_landmarks_ms"]["p95"],
        stream["end_to_end_ms"]["p95"],
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200)
    bars = ax.barh(labels[::-1], values[::-1], color=TEAL, height=0.55)
    ax.bar_label(bars, fmt="%.1f ms", padding=4, color=INK, fontsize=12)
    ax.axvline(50, color="#C8463D", ls="--", lw=1.2)
    ax.text(50.5, 3.35, "50 ms target", color="#C8463D", fontsize=11)
    ax.set(xlim=(0, 62), xlabel="p95 latency (ms)")
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Measured p95 latency, Apple M4 CPU")
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", facecolor="white")
    plt.close(fig)


def calibration() -> None:
    data = report("calibration_analysis.json")
    bins = data["bins"]
    centres = [sum(b["range"]) / 2 for b in bins]
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=200)
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="Perfect calibration")
    ax.bar(
        centres,
        [b["accuracy"] for b in bins],
        width=0.085,
        color=TEAL,
        label="Actual top-1 accuracy",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Model score of the top prediction", ylabel="Accuracy")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ece = data["expected_calibration_error_10_bins"]
    ax.set_title(f"Scores overstate accuracy (ECE {ece:.3f})")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    training_curve()
    test_accuracy()
    latency()
    calibration()
    print(sorted(p.name for p in OUT.glob("*.png")))
