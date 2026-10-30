"""Inference helpers shared by the local Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..preprocessing.pipeline import (
    DERIVED_FEATURE_COLUMNS,
    add_derived_sequence_features,
    make_padded_sequences,
    prepare_detection_frame,
)
from ..models.hybrid_cnn_gru import HybridCNNGRU
from ..training.loop import CLASS_NAMES


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint_path: Path
    normalization_path: Path
    feature_config_path: Path
    device: str | None = None


class LightCurveInference:
    """Load one trained model and apply the exact training preprocessing."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.normalization = self._load_json(config.normalization_path)
        feature_config = self._load_json(config.feature_config_path)
        self.features = tuple(feature_config["features"])
        if self.features != tuple(DERIVED_FEATURE_COLUMNS):
            raise ValueError(
                "Feature order konfigurasi tidak sesuai dengan derived feature pipeline"
            )
        self.max_length = int(feature_config["max_length"])
        self.model = HybridCNNGRU(feature_count=len(self.features))
        checkpoint = torch.load(
            config.checkpoint_path, map_location=self.device, weights_only=False
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _prepare(self, detections: pd.DataFrame) -> pd.DataFrame:
        required = {
            "oid", "candid", "mjd", "fid", "magpsf", "sigmapsf",
            "magpsf_corr", "sigmapsf_corr", "corrected",
        }
        missing = required.difference(detections.columns)
        if missing:
            raise ValueError(f"Kolom detection tidak lengkap: {sorted(missing)}")
        prepared = prepare_detection_frame(detections.copy())
        prepared = add_derived_sequence_features(prepared)
        if prepared.empty:
            raise ValueError("Tidak ada detection valid setelah preprocessing")
        for feature in self.features:
            stats = self.normalization.get(feature)
            if not stats:
                raise ValueError(f"Normalisasi tidak tersedia untuk fitur {feature}")
            std = float(stats["std"])
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"Standard deviation tidak valid untuk fitur {feature}")
            prepared[feature] = (
                prepared[feature].astype("float32") - float(stats["mean"])
            ) / std
        return prepared.sort_values(["mjd", "candid"], kind="mergesort").reset_index(
            drop=True
        )

    def predict(
        self, detections: pd.DataFrame, mode: str = "full"
    ) -> dict[str, Any]:
        prepared = self._prepare(detections)
        if mode not in {"full", "early"}:
            raise ValueError("mode harus 'full' atau 'early'")
        if mode == "early":
            if len(prepared) < 6:
                raise ValueError("Early classification membutuhkan minimal 6 detection")
            prepared = prepared.iloc[:6].copy()
        sequence = prepared.loc[:, self.features].to_numpy(dtype="float32")
        _, x, mask, truncated = make_padded_sequences(
            {"selected": sequence}, ["selected"], self.max_length,
            feature_count=len(self.features),
        )
        with torch.no_grad():
            tensor_x = torch.from_numpy(x).to(self.device)
            tensor_mask = torch.from_numpy(mask).to(self.device)
            probabilities = torch.softmax(self.model(tensor_x, tensor_mask), dim=1)
        values = probabilities[0].cpu().numpy()
        class_index = int(values.argmax())
        return {
            "class_name": CLASS_NAMES[class_index],
            "class_index": class_index,
            "probabilities": dict(zip(CLASS_NAMES, values.astype(float))),
            "confidence": float(values[class_index]),
            "mode": mode,
            "detection_count": int(len(prepared)),
            "truncated": bool(truncated),
            "prepared": prepared,
        }
