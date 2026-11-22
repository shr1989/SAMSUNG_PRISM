import argparse
import json
import logging
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn

from src import config
from src.augment import augment
from src.dataset import SplitData, batches, load_split
from src.model import build_model, count_parameters, size_mb

log = logging.getLogger("train")


@dataclass
class TrainConfig:
    arch: str = config.DEFAULT_ARCH
    epochs: int = 150
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    patience: int = 20
    seed: int = config.SEED
    time_budget_min: float = 600.0
    amp: str = "auto"


def _usable(device: str) -> bool:
    # Some CUDA builds load but lack kernels for older GPUs; run a real op to confirm.
    try:
        (torch.ones(2, device=device) * 2).sum().item()
        return True
    except (RuntimeError, AssertionError):
        return False


def select_device() -> torch.device:
    requested = os.environ.get("ECHOGESTURE_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available() and _usable("cuda"):
        return torch.device("cuda")
    if torch.backends.mps.is_available() and _usable("mps"):
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device: torch.device, mode: str) -> torch.dtype | None:
    # Mixed precision only on CUDA: bfloat16 where supported, otherwise float16 with loss scaling.
    if mode == "off" or device.type != "cuda":
        return None
    try:
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        native_bf16 = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if native_bf16 else torch.float16


def restore_rng(state: dict, generator: torch.Generator) -> None:
    # map_location moves every tensor to the training device; RNG states must stay on CPU.
    generator.set_state(state["generator"].cpu())
    torch.set_rng_state(state["torch_rng"].cpu())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def feature_stats(
    model: nn.Module, data: SplitData, batch_size: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fitted on clean (unaugmented) training data only.
    total, total_sq, count = 0.0, 0.0, 0
    for x, _ in batches(data, batch_size, shuffle=False):
        f = model.features(x).flatten(0, 1).cpu().double()
        total = total + f.sum(0)
        total_sq = total_sq + (f * f).sum(0)
        count += f.shape[0]
    mean = total / count
    std = (total_sq / count - mean * mean).clamp_min(0).sqrt()
    return mean.float(), std.float()


def training_feature_stats(model: nn.Module, train: SplitData) -> tuple[torch.Tensor, torch.Tensor]:
    from src.preprocess import ids_fingerprint

    path = config.FEATURE_STATS_NPZ
    if path.exists():
        with np.load(path) as saved:
            if str(saved["train_ids_sha256"]) != ids_fingerprint(train.video_ids):
                raise ValueError(
                    f"{path.name} was fitted on a different training set; rerun preprocess"
                )
            log.info("feature statistics loaded from %s", path.name)
            return torch.from_numpy(saved["mean"]), torch.from_numpy(saved["std"])
    log.info("feature statistics computed from the training split")
    return feature_stats(model, train)


@torch.no_grad()
def predict_logits(model: nn.Module, data: SplitData, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x).float().cpu() for x, _ in batches(data, batch_size, False)])


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.cpu()
    loss = nn.functional.cross_entropy(logits, labels).item()
    top5 = logits.topk(5, dim=1).indices
    pred = top5[:, 0]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(),
        pred.numpy(),
        labels=np.unique(labels.numpy()),
        average="macro",
        zero_division=0,
    )
    return {
        "loss": loss,
        "top1": (pred == labels).float().mean().item(),
        "top5": (top5 == labels[:, None]).any(1).float().mean().item(),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def check_test_untouched(train: SplitData, val: SplitData) -> None:
    # Test IDs come from the manifest; the test features are never loaded during training.
    manifest = pd.read_csv(config.TRAINING_MANIFEST_CSV, dtype={"video_id": str})
    test_ids = set(manifest.loc[manifest["split"] == "test", "video_id"])
    train_ids, val_ids = set(train.video_ids.tolist()), set(val.video_ids.tolist())
    if train_ids & val_ids or (train_ids | val_ids) & test_ids:
        raise ValueError("train/val/test video IDs overlap")


def thermal_state() -> str:
    try:
        output = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    warnings = [
        line.strip() for line in output.splitlines() if "level" in line and "No " not in line
    ]
    return "; ".join(warnings) or "no thermal warning"


def system_snapshot(device: torch.device) -> dict:
    snapshot = {
        "load_1min": os.getloadavg()[0],
        "ram_available_gb": psutil.virtual_memory().available / 2**30,
        "process_rss_mb": psutil.Process().memory_info().rss / 2**20,
    }
    if device.type == "mps":
        snapshot["mps_allocated_mb"] = torch.mps.current_allocated_memory() / 2**20
        snapshot["mps_driver_mb"] = torch.mps.driver_allocated_memory() / 2**20
    return snapshot


def save_atomic(state: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def memory_mb(device: torch.device) -> float:
    rss = psutil.Process().memory_info().rss / 2**20
    if device.type == "mps":
        rss = max(rss, torch.mps.driver_allocated_memory() / 2**20)
    if device.type == "cuda":
        rss = max(rss, torch.cuda.max_memory_allocated() / 2**20)
    return rss


def train(cfg: TrainConfig, run_dir: Path, resume: bool) -> dict:
    seed_everything(cfg.seed)
    device = select_device()
    train_data, val_data = load_split("train"), load_split("val")
    check_test_untouched(train_data, val_data)
    train_data, val_data = train_data.to(device), val_data.to(device)

    model = build_model(cfg.arch).to(device)
    model.set_feature_stats(*[t.to(device) for t in training_feature_stats(model, train_data)])
    dtype = amp_dtype(device, cfg.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = -(-len(train_data) // cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    generator = torch.Generator().manual_seed(cfg.seed)
    history, best_f1, best_epoch, start_epoch, elapsed = [], -1.0, -1, 0, 0.0

    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        saved = state["train_config"]
        if (saved["epochs"], saved["batch_size"], saved["arch"]) != (
            cfg.epochs,
            cfg.batch_size,
            cfg.arch,
        ):
            raise ValueError(f"resume needs the same epochs/batch_size/arch as {saved}")
        model.load_state_dict(state["model"])
        scaler.load_state_dict(state["scaler"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state, generator)
        history, best_f1, best_epoch = state["history"], state["best_f1"], state["best_epoch"]
        start_epoch, elapsed = state["epoch"] + 1, state["elapsed"]
        log.info("resumed at epoch %d", start_epoch)

    log.info(
        "device=%s amp=%s params=%d size=%.2fMB train=%d val=%d",
        device,
        dtype,
        count_parameters(model),
        size_mb(model),
        len(train_data),
        len(val_data),
    )
    peak_mb = memory_mb(device)
    began = time.perf_counter() - elapsed
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_began, running, seen = time.perf_counter(), 0.0, 0
        for x, y in batches(train_data, cfg.batch_size, True, generator):
            with torch.autocast(
                device.type, dtype=dtype or torch.float32, enabled=dtype is not None
            ):
                logits = model(augment(x, generator))
            loss = criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)
        val = classification_metrics(predict_logits(model, val_data), val_data.labels)
        peak_mb = max(peak_mb, memory_mb(device))
        row = {
            "epoch": epoch,
            "train_loss": running / seen,
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": time.perf_counter() - epoch_began,
        }
        row |= {f"val_{k}": v for k, v in val.items()}
        row |= system_snapshot(device)
        history.append(row)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            save_atomic(
                {
                    "model": model.state_dict(),
                    "arch": cfg.arch,
                    "epoch": epoch,
                    "val": val,
                    "train_config": asdict(cfg),
                },
                best_path,
            )
        log.info(
            "epoch %3d loss %.3f val_loss %.3f top1 %.4f top5 %.4f P %.4f R %.4f F1 %.4f "
            "%.1fs | mps %.0f MB | load %.1f",
            epoch,
            row["train_loss"],
            val["loss"],
            val["top1"],
            val["top5"],
            val["macro_precision"],
            val["macro_recall"],
            val["macro_f1"],
            row["epoch_seconds"],
            row.get("mps_driver_mb", 0.0),
            row["load_1min"],
        )
        if epoch % 10 == 0:
            log.info("thermal: %s", thermal_state())
        save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "train_config": asdict(cfg),
                "generator": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "history": history,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "elapsed": time.perf_counter() - began,
            },
            last_path,
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if epoch - best_epoch >= cfg.patience:
            log.info("early stop: no val macro F1 gain for %d epochs", cfg.patience)
            break
        if (time.perf_counter() - began) / 60 > cfg.time_budget_min:
            log.info("time budget of %.0f min reached", cfg.time_budget_min)
            break

    best = history[best_epoch]
    return {
        "arch": cfg.arch,
        "device": str(device),
        "mixed_precision": str(dtype) if dtype else "off",
        "parameters": count_parameters(model),
        "model_size_mb": size_mb(model),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val": {k.removeprefix("val_"): v for k, v in best.items() if k.startswith("val_")},
        "final_train_loss": history[-1]["train_loss"],
        "train_loss_at_best": best["train_loss"],
        "mean_epoch_seconds": float(np.mean([h["epoch_seconds"] for h in history])),
        "training_minutes": (time.perf_counter() - began) / 60,
        "peak_memory_mb": peak_mb,
        "config": asdict(cfg),
    }


def plot_history(history_csv: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train (augmented)")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    for key in ("val_top1", "val_top5", "val_macro_precision", "val_macro_recall", "val_macro_f1"):
        axes[1].plot(history["epoch"], history[key], label=key.removeprefix("val_"))
    axes[1].set_title("Validation")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    cfg = TrainConfig(**args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_dirs()

    run_dir = config.CHECKPOINT_DIR / cfg.arch
    run_dir.mkdir(parents=True, exist_ok=True)
    report = train(cfg, run_dir, resume)
    plot_history(run_dir / "history.csv", config.PLOT_DIR / f"training_{cfg.arch}.png")
    (config.REPORT_DIR / f"training_report_{cfg.arch}.json").write_text(
        json.dumps(report, indent=2)
    )
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
