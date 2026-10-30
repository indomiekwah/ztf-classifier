"""Reusable, reviewable helpers for the ALeRCE preprocessing notebook.

The functions in this module operate on OID-level samples and keep the
dataset split separate from detection rows. The notebook is the orchestration
and review layer; this module contains the reusable transformations.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.model_selection import train_test_split

DETECTION_COLUMNS = [
    "oid",
    "candid",
    "mjd",
    "fid",
    "magpsf",
    "sigmapsf",
    "magpsf_corr",
    "sigmapsf_corr",
    "corrected",
]
FEATURE_COLUMNS = [
    "time_rel",
    "delta_time",
    "magnitude",
    "magnitude_error",
    "filter_g",
    "filter_r",
]
DERIVED_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS[:4],
    "flux_log",
    "flux_error",
    "delta_magnitude",
    "local_slope",
    "log_snr",
    *FEATURE_COLUMNS[4:],
]


def add_derived_sequence_features(
    prepared: pd.DataFrame,
    slope_epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Add per-detection photometric features without using future observations."""
    if slope_epsilon <= 0:
        raise ValueError("slope_epsilon harus positif")
    result = prepared.copy()
    result["flux_log"] = -0.4 * result["magnitude"]
    flux = np.power(10.0, result["flux_log"])
    result["flux_error"] = 0.4 * np.log(10.0) * flux * result["magnitude_error"]
    result["delta_magnitude"] = (
        result.groupby("oid")["magnitude"].diff().fillna(0.0)
    )
    result["local_slope"] = result["delta_magnitude"] / result["delta_time"].clip(
        lower=slope_epsilon
    )
    snr = 1.0 / (0.4 * np.log(10.0) * result["magnitude_error"].clip(lower=slope_epsilon))
    result["log_snr"] = np.log1p(snr)
    return result


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        path,
        usecols=["oid", "classALeRCE", "classMacro"],
        dtype={"oid": "string", "classALeRCE": "string", "classMacro": "string"},
    ).dropna(subset=["oid", "classMacro"])
    labels = labels.drop_duplicates("oid").reset_index(drop=True)
    if not labels["oid"].is_unique:
        raise ValueError("Manifest label memiliki OID duplikat")
    return labels


def split_oids(
    labels: pd.DataFrame,
    random_state: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.DataFrame:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction dan test_fraction harus positif")
    temporary_fraction = validation_fraction + test_fraction
    train_ids, temporary_ids = train_test_split(
        labels["oid"],
        test_size=temporary_fraction,
        stratify=labels["classMacro"],
        random_state=random_state,
    )
    temporary = labels.set_index("oid").loc[temporary_ids]
    validation_ids, test_ids = train_test_split(
        temporary.index,
        test_size=test_fraction / temporary_fraction,
        stratify=temporary["classMacro"],
        random_state=random_state,
    )
    split_by_oid = {oid: "train" for oid in train_ids}
    split_by_oid.update({oid: "validation" for oid in validation_ids})
    split_by_oid.update({oid: "test" for oid in test_ids})
    result = labels.copy()
    result["split"] = result["oid"].map(split_by_oid)
    if result["split"].isna().any():
        raise ValueError("Sebagian OID tidak memiliki split")
    return result


def iter_detection_frames(
    detections_dir: Path,
    labels: pd.DataFrame,
    batch_size: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """Yield only g/r detections, joined to the OID manifest, per Arrow batch."""
    if batch_size <= 0:
        raise ValueError("batch_size harus positif")
    dataset = ds.dataset(detections_dir, format="parquet")
    scanner = dataset.scanner(columns=DETECTION_COLUMNS, batch_size=batch_size)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        frame = frame[frame["fid"].isin([1, 2])].copy()
        if frame.empty:
            continue
        frame["oid"] = frame["oid"].astype("string")
        yield frame.merge(labels, on="oid", how="inner", validate="many_to_one")


def prepare_detection_frame(
    frame: pd.DataFrame,
    magnitude_min: float = 0.0,
    magnitude_max: float = 40.0,
    error_max: float = 10.0,
) -> pd.DataFrame:
    """Apply the documented photometry policy and create chronological features."""
    prepared = frame.sort_values(["oid", "mjd", "candid"], kind="mergesort").copy()
    corrected_ok = (
        prepared["corrected"].eq(True)
        & prepared["magpsf_corr"].between(magnitude_min, magnitude_max)
        & prepared["sigmapsf_corr"].between(0.0, error_max)
    )
    prepared["magnitude"] = prepared["magpsf"].where(
        ~corrected_ok, prepared["magpsf_corr"]
    )
    prepared["magnitude_error"] = prepared["sigmapsf"].where(
        ~corrected_ok, prepared["sigmapsf_corr"]
    )
    valid = (
        prepared["mjd"].between(0.0, np.inf)
        & prepared["magnitude"].between(magnitude_min, magnitude_max)
        & prepared["magnitude_error"].between(0.0, error_max)
    )
    prepared = prepared.loc[valid].copy()
    prepared["time_rel"] = prepared["mjd"] - prepared.groupby("oid")["mjd"].transform(
        "first"
    )
    prepared["delta_time"] = (
        prepared.groupby("oid")["mjd"].diff().fillna(0.0)
    )
    prepared["filter_g"] = prepared["fid"].eq(1).astype("float32")
    prepared["filter_r"] = prepared["fid"].eq(2).astype("float32")
    return prepared


def make_padded_sequences(
    sequences: dict[str, np.ndarray],
    oids: Sequence[str],
    max_length: int,
    feature_count: int | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    if max_length <= 0:
        raise ValueError("max_length harus positif")
    if feature_count is None:
        feature_count = len(FEATURE_COLUMNS)
    if feature_count <= 0:
        raise ValueError("feature_count harus positif")
    selected = [oid for oid in oids if oid in sequences]
    x = np.zeros((len(selected), max_length, feature_count), dtype="float32")
    mask = np.zeros((len(selected), max_length), dtype="float32")
    truncated = 0
    for index, oid in enumerate(selected):
        sequence = sequences[oid]
        if sequence.ndim != 2 or sequence.shape[1] != feature_count:
            raise ValueError(
                f"Sequence {oid} memiliki {sequence.shape[1]} fitur; "
                f"diharapkan {feature_count}"
            )
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
            truncated += 1
        length = len(sequence)
        x[index, :length] = sequence
        mask[index, :length] = 1.0
    return selected, x, mask, truncated
