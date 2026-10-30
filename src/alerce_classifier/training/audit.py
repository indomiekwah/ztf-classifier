"""Audit and preprocessing utilities for the acquired ALeRCE light curves.

The audit command is intentionally class-scoped and streaming. It is the first
step before making a train/validation/test split or fitting normalization
statistics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

MACRO_CLASSES = ("Periodic", "Stochastic", "Transient")
DETECTION_COLUMNS = (
    "oid",
    "candid",
    "mjd",
    "fid",
    "magpsf",
    "sigmapsf",
    "magpsf_corr",
    "sigmapsf_corr",
    "corrected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-class",
        help="Audit one macro class without loading the full dataset into RAM.",
    )
    audit.add_argument(
        "--input",
        type=Path,
        default=Path("data/alerce_ztf_lc_sanchezsaez_full/data/full"),
        help="Acquisition output directory.",
    )
    audit.add_argument("--class-macro", choices=MACRO_CLASSES, required=True)
    audit.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path; defaults to <input>/preprocessing/audit-<class>.json.",
    )
    audit.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Rows per Arrow batch while scanning detections.",
    )
    return parser.parse_args()


def require_acquisition(input_dir: Path) -> None:
    required = (
        input_dir / "_SUCCESS",
        input_dir / "labels_selected.csv",
        input_dir / "detections",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Artefak akuisisi tidak lengkap; file/folder tidak ditemukan: "
            + ", ".join(missing)
        )


def audit_class(input_dir: Path, class_macro: str, batch_size: int) -> dict:
    require_acquisition(input_dir)
    labels = pd.read_csv(
        input_dir / "labels_selected.csv",
        usecols=["oid", "classALeRCE", "classMacro"],
        dtype={"oid": "string", "classALeRCE": "string", "classMacro": "string"},
    )
    labels = labels[labels["classMacro"] == class_macro].copy()
    if labels.empty:
        raise ValueError(f"Tidak ada OID untuk classMacro={class_macro!r}")

    class_oids = set(labels["oid"].dropna().astype(str))
    sequence_counts: Counter[str] = Counter()
    filter_counts: Counter[str] = Counter()
    invalid_mjd = 0
    missing_values: Counter[str] = Counter()
    corrected_counts: Counter[str] = Counter()
    magnitude_source_counts: Counter[str] = Counter()
    magnitude_values: defaultdict[str, list[float]] = defaultdict(list)
    error_values: defaultdict[str, list[float]] = defaultdict(list)

    dataset = ds.dataset(input_dir / "detections", format="parquet")
    scanner = dataset.scanner(
        columns=list(DETECTION_COLUMNS),
        filter=ds.field("oid").isin(class_oids),
        batch_size=batch_size,
    )
    detection_rows = 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        detection_rows += len(frame)
        frame["oid"] = frame["oid"].astype(str)

        for column in DETECTION_COLUMNS:
            missing_values[column] += int(frame[column].isna().sum())
        invalid_mjd += int((~pd.to_numeric(frame["mjd"], errors="coerce").notna()).sum())

        for value, count in frame["fid"].value_counts(dropna=False).items():
            filter_counts[str(value)] += int(count)
        for value, count in frame["corrected"].value_counts(dropna=False).items():
            corrected_counts[str(value)] += int(count)

        for row in frame.itertuples(index=False):
            oid = str(row.oid)
            if pd.notna(row.mjd):
                sequence_counts[oid] += 1
            if pd.notna(row.magpsf_corr) and row.corrected == 1:
                source = "magpsf_corr"
                magnitude = float(row.magpsf_corr)
                error = row.sigmapsf_corr
            else:
                source = "magpsf"
                magnitude = row.magpsf
                error = row.sigmapsf
            if pd.notna(magnitude):
                magnitude_source_counts[source] += 1
                if len(magnitude_values[source]) < 100_000:
                    magnitude_values[source].append(float(magnitude))
            if pd.notna(error):
                if len(error_values[source]) < 100_000:
                    error_values[source].append(float(error))

    lengths = pd.Series(
        [sequence_counts.get(oid, 0) for oid in sorted(class_oids)],
        dtype="int64",
    )

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "median": None, "max": None}
        series = pd.Series(values)
        return {
            "min": float(series.min()),
            "median": float(series.median()),
            "max": float(series.max()),
        }

    return {
        "classMacro": class_macro,
        "source": str(input_dir),
        "selected_oids": len(class_oids),
        "oids_with_detections": int((lengths > 0).sum()),
        "oids_without_valid_mjd": int((lengths == 0).sum()),
        "detection_rows": detection_rows,
        "sequence_length": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "p95": float(lengths.quantile(0.95)),
            "max": int(lengths.max()),
        },
        "filter_counts": dict(filter_counts),
        "corrected_counts": dict(corrected_counts),
        "missing_values": dict(missing_values),
        "invalid_mjd_values": invalid_mjd,
        "duplicate_check": "validated_by_acquisition_pipeline",
        "magnitude_source_counts": dict(magnitude_source_counts),
        "magnitude_summary": {
            source: summary(values) for source, values in magnitude_values.items()
        },
        "error_summary": {
            source: summary(values) for source, values in error_values.items()
        },
        "label_counts": labels["classALeRCE"].value_counts().to_dict(),
    }


def main() -> None:
    args = parse_args()
    if args.command != "audit-class":
        raise ValueError(f"Perintah tidak didukung: {args.command}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size harus lebih besar dari nol")

    result = audit_class(args.input, args.class_macro, args.batch_size)
    output = args.output or (
        args.input / "preprocessing" / f"audit-{args.class_macro.lower()}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Audit disimpan ke {output}")


if __name__ == "__main__":
    main()
