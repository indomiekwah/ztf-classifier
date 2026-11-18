"""Training and evaluation helpers for the padded light-curve tensors."""

from __future__ import annotations

from pathlib import Path
import tracemalloc
import time
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ("Periodic", "Stochastic", "Transient")


def load_split(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = np.load(path)
    return (
        torch.from_numpy(arrays["x"]).float(),
        torch.from_numpy(arrays["mask"]).float(),
        torch.from_numpy(arrays["y"]).long(),
    )


def make_loader(
    x: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(TensorDataset(x, mask, y), batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, mask), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_items += len(y)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> dict:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    total_loss = 0.0
    total_items = 0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        if loss_fn is not None:
            total_loss += loss_fn(logits, y).item() * len(y)
            total_items += len(y)
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        actual.extend(y.tolist())
    if not actual:
        raise ValueError("Loader evaluasi tidak berisi objek")
    report = classification_report(
        actual,
        predicted,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=range(len(CLASS_NAMES))).tolist(),
    }
    if loss_fn is not None:
        metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_size_bytes(path: Path) -> int:
    """Return a checkpoint size and fail explicitly when it is unavailable."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    return path.stat().st_size


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 1,
) -> dict[str, float]:
    """Measure inference time and peak device memory for a fixed loader."""
    if warmup_batches < 0:
        raise ValueError("warmup_batches tidak boleh negatif")
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()
    iterator = iter(loader)
    for _ in range(warmup_batches):
        try:
            x, mask, _ = next(iterator)
        except StopIteration:
            break
        model(x.to(device), mask.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    items = 0
    for x, mask, _ in loader:
        model(x.to(device), mask.to(device))
        items += len(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device))
    else:
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {
        "inference_seconds": elapsed,
        "inference_seconds_per_object": elapsed / max(items, 1),
        "peak_memory_bytes": peak_memory,
        "object_count": float(items),
    }


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_weights: np.ndarray,
    device: torch.device,
    epochs: int = 10,
    patience: int | None = None,
    learning_rate: float = 1e-3,
    checkpoint_path: Path | None = None,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    training_started = time.perf_counter()
    for epoch in range(epochs):
        started = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        validation = evaluate(model, validation_loader, device, loss_fn)
        record = {"epoch": epoch + 1, "train_loss": train_loss, **validation}
        history.append(record)
        improved = validation["macro_f1"] > best_f1
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if verbose:
            marker = " *best*" if improved else ""
            elapsed = time.perf_counter() - started
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={validation['loss']:.4f} | "
                f"val_acc={validation['accuracy']:.4f} | "
                f"val_macro_f1={validation['macro_f1']:.4f} | "
                f"{elapsed:.1f}s{marker}",
                flush=True,
            )
        if patience is not None and patience > 0 and epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"Early stopping setelah epoch {epoch + 1}: "
                    f"validation macro-F1 tidak meningkat selama {patience} epoch.",
                    flush=True,
                )
            break
    if best_state is None:
        raise RuntimeError("Training tidak menghasilkan checkpoint")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - training_started
    if checkpoint_path is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_f1,
                "training_seconds": training_seconds,
            },
            checkpoint_path,
        )
    return history, {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "stopped_epoch": history[-1]["epoch"],
        "training_seconds": training_seconds,
    }


def evaluate_prefixes(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    fractions: Iterable[float],
    device: torch.device,
) -> dict[str, dict]:
    results = {}
    for fraction in fractions:
        prefix_x = x.clone()
        prefix_mask = mask.clone()
        lengths = mask.sum(dim=1).to(torch.long)
        prefix_lengths = torch.ceil(lengths.float() * fraction).to(torch.long).clamp_min(1)
        positions = torch.arange(x.shape[1]).unsqueeze(0)
        prefix_mask = (positions < prefix_lengths.unsqueeze(1)).float() * mask
        prefix_x = prefix_x * prefix_mask.unsqueeze(-1)
        results[str(fraction)] = evaluate(
            model, make_loader(prefix_x, prefix_mask, y, batch_size=64), device
        )
    return results


def evaluate_detection_prefixes(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    detection_counts: Iterable[int | None],
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, dict]:
    """Evaluate chronological prefixes using an absolute detection count.

    ``None`` means all available valid detections. Objects with fewer than a
    requested count are excluded and reported through ``valid_object_count``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size harus positif")
    lengths = mask.sum(dim=1).to(torch.long)
    total_objects = int(len(y))
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    results: dict[str, dict] = {}
    for detection_count in detection_counts:
        if detection_count is not None and detection_count <= 0:
            raise ValueError("detection_count harus positif atau None")
        key = "all" if detection_count is None else str(detection_count)
        eligible = (
            torch.ones_like(lengths, dtype=torch.bool)
            if detection_count is None
            else lengths >= detection_count
        )
        selected_x = x[eligible].clone()
        selected_mask = mask[eligible].clone()
        selected_y = y[eligible]
        if detection_count is not None:
            selected_mask = (
                (positions < detection_count).to(selected_mask.dtype) * selected_mask
            )
            selected_x = selected_x * selected_mask.unsqueeze(-1)
        metrics = evaluate(
            model,
            make_loader(
                selected_x,
                selected_mask,
                selected_y,
                batch_size=batch_size,
            ),
            device,
        )
        results[key] = {
            "detection_count": detection_count,
            "valid_object_count": int(eligible.sum().item()),
            "total_object_count": total_objects,
            "coverage": float(eligible.float().mean().item()),
            **metrics,
        }
    return results
