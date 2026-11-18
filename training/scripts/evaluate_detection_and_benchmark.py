"""Evaluate detection-count prefixes and benchmark trained model artifacts.

Run this script after the common 70/15/15 preprocessing and model training
notebook has produced the three seed-42 checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alerce_classifier.models import CNNClassifier, GRUClassifier, HybridCNNGRU
from alerce_classifier.training.loop import (
    benchmark_inference,
    checkpoint_size_bytes,
    count_parameters,
    evaluate,
    evaluate_detection_prefixes,
    load_split,
    make_loader,
)


MODEL_FACTORIES = {
    "hybrid_cnn_gru": HybridCNNGRU,
    "cnn": CNNClassifier,
    "gru": GRUClassifier,
}
DETECTION_COUNTS = (6, 10, 15, 20, 30, None)


def checkpoint_path(artifact_dir: Path, model_name: str, seed: int) -> Path:
    model_dir = (
        artifact_dir
        if model_name == "hybrid_cnn_gru"
        else artifact_dir / "models" / model_name
    )
    return model_dir / f"derived_{model_name}_seed{seed}_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-count", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device; defaults to CUDA when available, otherwise CPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.feature_count <= 0 or args.batch_size <= 0:
        raise ValueError("feature-count dan batch-size harus positif")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    x, mask, y = load_split(args.split)
    full_loader = make_loader(x, mask, y, batch_size=args.batch_size)
    output = {
        "protocol": {
            "seed": args.seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "detection_counts": ["6", "10", "15", "20", "30", "all"],
            "test_object_count": int(len(y)),
        },
        "models": {},
    }

    for model_name, factory in MODEL_FACTORIES.items():
        checkpoint_file = checkpoint_path(args.artifact_dir, model_name, args.seed)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_file}")
        model = factory(feature_count=args.feature_count).to(device)
        checkpoint = torch.load(
            checkpoint_file, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        inference = benchmark_inference(model, full_loader, device)
        test_metrics = evaluate(model, full_loader, device)
        output["models"][model_name] = {
            "test": test_metrics,
            "early_classification": evaluate_detection_prefixes(
                model,
                x,
                mask,
                y,
                DETECTION_COUNTS,
                device,
                batch_size=args.batch_size,
            ),
            "benchmark": {
                "trainable_parameters": count_parameters(model),
                "checkpoint_size_bytes": checkpoint_size_bytes(checkpoint_file),
                **inference,
            },
            "selection": {
                "best_epoch": checkpoint.get("best_epoch"),
                "best_validation_macro_f1": checkpoint.get(
                    "best_validation_macro_f1"
                ),
                "training_seconds": checkpoint.get("training_seconds"),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Hasil evaluasi ditulis ke {args.output}")


if __name__ == "__main__":
    main()
