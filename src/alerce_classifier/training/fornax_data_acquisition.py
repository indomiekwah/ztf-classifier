"""Akuisisi labeled light curves ZTF dari ALeRCE TAP untuk Fornax.

Pipeline menggabungkan katalog label Sánchez-Sáez et al. (2021), ztf.object,
dan ztf.detection menggunakan oid. Hasil utama berupa partisi Parquet per batch
agar hemat ruang, dapat dilanjutkan, dan tidak memerlukan seluruh data di RAM.

Contoh:
    python fornax_data_acquisition.py --per-subclass 2 --output data/smoke
    python fornax_data_acquisition.py --per-subclass 20 --output data/pilot20
    python fornax_data_acquisition.py --per-subclass 0 --output data/full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyvo

TAP_URL = "https://tap.alerce.online/tap"
LABEL_FILE = "labeled_set_lc_classifier_SanchezSaez_2020.csv"
ID_PATTERN = re.compile(r"^ZTF[0-9]{2}[a-z]+$")

SUBCLASS_TO_MACRO = {
    "SNIa": "Transient",
    "SNIbc": "Transient",
    "SNII": "Transient",
    "SLSN": "Transient",
    "AGN": "Stochastic",
    "QSO": "Stochastic",
    "Blazar": "Stochastic",
    "YSO": "Stochastic",
    "CV/Nova": "Stochastic",
    "LPV": "Periodic",
    "E": "Periodic",
    "DSCT": "Periodic",
    "RRL": "Periodic",
    "CEP": "Periodic",
    "Periodic-Other": "Periodic",
}

OBJECT_COLUMNS = ["oid", "ndet", "meanra", "meandec", "firstmjd", "lastmjd"]
DETECTION_COLUMNS = [
    "oid",
    "candid",
    "mjd",
    "fid",
    "magpsf",
    "sigmapsf",
    "magpsf_corr",
    "sigmapsf_corr",
    "isdiffpos",
    "rb",
    "drb",
    "dubious",
    "corrected",
]
OUTPUT_COLUMNS = [
    "oid",
    "classALeRCE",
    "classMacro",
    *[column for column in OBJECT_COLUMNS if column != "oid"],
    *[column for column in DETECTION_COLUMNS if column != "oid"],
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_dataframe(data: pd.DataFrame, path: Path, output_format: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if output_format == "parquet":
        data.to_parquet(temporary, index=False, compression="zstd")
    else:
        data.to_csv(temporary, index=False)
    temporary.replace(path)


def combine_unique(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values.dropna() if str(value).strip()}))


def read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        path,
        dtype={
            "oid": "string",
            "classALeRCE": "string",
            "source": "string",
            "id_source": "string",
        },
    )
    required = ["oid", "classALeRCE", "ra", "dec", "source", "id_source"]
    missing = set(required) - set(labels.columns)
    if missing:
        raise ValueError(f"Kolom katalog tidak ada: {sorted(missing)}")

    labels = labels[required].copy()
    labels["oid"] = labels["oid"].str.strip()
    labels["classALeRCE"] = labels["classALeRCE"].str.strip()
    labels = labels.dropna(subset=["oid", "classALeRCE"])
    labels = labels[(labels["oid"] != "") & (labels["classALeRCE"] != "")]

    invalid = labels.loc[~labels["oid"].str.fullmatch(ID_PATTERN.pattern), "oid"]
    if not invalid.empty:
        raise ValueError(f"OID katalog tidak valid, misalnya {invalid.iloc[0]!r}")

    unknown = sorted(set(labels["classALeRCE"]) - set(SUBCLASS_TO_MACRO))
    if unknown:
        raise ValueError(f"Subkelas belum memiliki pemetaan kelas makro: {unknown}")

    conflicts = labels.groupby("oid")["classALeRCE"].nunique()
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index[:5].tolist()
        raise ValueError(f"Satu oid memiliki label bertentangan, misalnya {examples}")

    # Satu OID dapat tercatat dari lebih dari satu katalog dengan label yang sama.
    labels = (
        labels.groupby(["oid", "classALeRCE"], as_index=False)
        .agg(
            ra=("ra", "first"),
            dec=("dec", "first"),
            source=("source", combine_unique),
            id_source=("id_source", combine_unique),
        )
        .sort_values("oid")
        .reset_index(drop=True)
    )
    labels["classMacro"] = labels["classALeRCE"].map(SUBCLASS_TO_MACRO)
    return labels


def choose_sample(labels: pd.DataFrame, per_subclass: int, seed: int) -> pd.DataFrame:
    if per_subclass == 0:
        return labels.sort_values("oid").reset_index(drop=True)
    pieces = [
        group.sample(n=min(per_subclass, len(group)), random_state=seed)
        for _, group in labels.groupby("classALeRCE", sort=True)
    ]
    return pd.concat(pieces, ignore_index=True).sort_values("oid").reset_index(drop=True)


def selection_hash(labels: pd.DataFrame) -> str:
    payload = labels[["oid", "classALeRCE"]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sql_values(oids: list[str]) -> str:
    # Semua OID telah lolos regex alfanumerik yang ketat.
    return ", ".join(f"'{oid}'" for oid in oids)


def run_query(tap: pyvo.dal.TAPService, query: str, maxrec: int) -> pd.DataFrame:
    for attempt in range(3):
        try:
            result = tap.run_sync(query, maxrec=maxrec).to_table().to_pandas()
            if len(result) >= maxrec:
                raise ValueError(
                    f"Kueri mencapai MAXREC={maxrec}; naikkan --maxrec atau kecilkan batch"
                )
            if "oid" in result.columns:
                result["oid"] = result["oid"].astype(str)
            return result
        except Exception:
            if attempt == 2:
                raise
            wait = 2**attempt
            print(f"  Kueri gagal; mencoba lagi dalam {wait} detik...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Kueri TAP gagal tanpa exception")


def fetch_objects(tap: pyvo.dal.TAPService, oids: list[str]) -> pd.DataFrame:
    query = f"""
SELECT oid, ndet, meanra, meandec, firstmjd, lastmjd
FROM ztf.object
WHERE oid IN ({sql_values(oids)})
"""
    objects = run_query(tap, query, maxrec=max(len(oids) + 1, 100))
    if objects["oid"].duplicated().any():
        raise ValueError("ztf.object mengembalikan OID ganda")
    return objects


def detection_groups(objects: pd.DataFrame, maxrec: int) -> list[list[str]]:
    target_rows = max(1, int(maxrec * 0.8))
    groups: list[list[str]] = []
    current: list[str] = []
    current_rows = 0

    for oid, ndet in objects.sort_values("oid")[["oid", "ndet"]].itertuples(
        index=False, name=None
    ):
        expected = max(1, int(ndet)) if pd.notna(ndet) else 1
        if expected >= maxrec:
            raise ValueError(
                f"OID {oid} memiliki ndet={expected}, melebihi --maxrec={maxrec}"
            )
        if current and current_rows + expected > target_rows:
            groups.append(current)
            current = []
            current_rows = 0
        current.append(str(oid))
        current_rows += expected
    if current:
        groups.append(current)
    return groups


def fetch_detections(
    tap: pyvo.dal.TAPService, objects: pd.DataFrame, maxrec: int
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    groups = detection_groups(objects, maxrec)
    for index, oids in enumerate(groups, start=1):
        print(
            f"  Mengambil ztf.detection bagian {index}/{len(groups)} "
            f"untuk {len(oids)} oid...",
            flush=True,
        )
        query = f"""
SELECT oid, candid, mjd, fid, magpsf, sigmapsf,
       magpsf_corr, sigmapsf_corr, isdiffpos,
       rb, drb, dubious, corrected
FROM ztf.detection
WHERE oid IN ({sql_values(oids)})
"""
        part = run_query(tap, query, maxrec=maxrec)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    detections = pd.concat(parts, ignore_index=True)
    if detections.duplicated(["oid", "candid"]).any():
        raise ValueError("Ada deteksi ganda (oid, candid) dalam hasil TAP")
    return detections.sort_values(["oid", "mjd", "candid"]).reset_index(drop=True)


def audit_detection_counts(
    objects: pd.DataFrame, detections: pd.DataFrame
) -> list[dict]:
    observed = detections.groupby("oid").size().reindex(objects["oid"], fill_value=0)
    expected = objects.set_index("oid")["ndet"].astype(float).reindex(objects["oid"])
    audit = []

    for oid in objects["oid"]:
        received = int(observed.loc[oid])
        reported = expected.loc[oid]
        if pd.isna(reported):
            raise ValueError(f"ztf.object.ndet kosong untuk {oid}")

        reported_int = int(reported)
        difference = received - reported_int

        if difference != 0:
            audit.append(
                {
                    "oid": oid,
                    "detections_received": received,
                    "object_ndet": reported_int,
                    "difference": difference,
                    "reason": (
                        "detection_fewer_than_object_snapshot"
                        if difference < 0
                        else "detection_newer_than_object_snapshot"
                    ),
                }
            )

    return audit


def prepare_directories(output_dir: Path) -> dict[str, Path]:
    paths = {
        "detections": output_dir / "detections",
        "object_parts": output_dir / "object_parts",
        "missing_parts": output_dir / "missing_parts",
        "audit_parts": output_dir / "audit_parts",
        "status": output_dir / "status",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(exist_ok=True)
    return paths


def part_path(directory: Path, batch_index: int, output_format: str) -> Path:
    extension = "parquet" if output_format == "parquet" else "csv"
    return directory / f"part-{batch_index:06d}.{extension}"


def load_status(status_dir: Path, batch_index: int) -> dict | None:
    path = status_dir / f"batch-{batch_index:06d}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_empty_or_data(
    data: pd.DataFrame, path: Path, output_format: str, columns: list[str]
) -> None:
    atomic_dataframe(data.reindex(columns=columns), path, output_format)


def finalize(
    output_dir: Path,
    paths: dict[str, Path],
    labels_all: pd.DataFrame,
    labels_selected: pd.DataFrame,
    total_batches: int,
    output_format: str,
    started_at: float,
) -> None:
    statuses = []
    for batch_index in range(total_batches):
        status = load_status(paths["status"], batch_index)
        if status is None:
            raise RuntimeError(f"Status batch {batch_index} tidak ditemukan")
        statuses.append(status)

    def read_parts(directory: Path) -> list[pd.DataFrame]:
        extension = "parquet" if output_format == "parquet" else "csv"
        frames = []
        for path in sorted(directory.glob(f"*.{extension}")):
            frames.append(
                pd.read_parquet(path) if output_format == "parquet" else pd.read_csv(path)
            )
        return frames

    object_frames = read_parts(paths["object_parts"])
    objects = (
        pd.concat(object_frames, ignore_index=True).drop_duplicates("oid")
        if object_frames
        else pd.DataFrame(columns=OBJECT_COLUMNS)
    )
    atomic_dataframe(objects, output_dir / f"objects.{output_format}", output_format)

    missing_frames = read_parts(paths["missing_parts"])
    missing = (
        pd.concat(missing_frames, ignore_index=True).drop_duplicates("oid")
        if missing_frames
        else pd.DataFrame(columns=["oid", "reason"])
    )
    missing.to_csv(output_dir / "missing_oids.csv", index=False)

    audit_frames = read_parts(paths["audit_parts"])
    audit_columns = [
        "oid",
        "detections_received",
        "object_ndet",
        "difference",
        "reason",
    ]
    audit = (
        pd.concat(audit_frames, ignore_index=True)
        if audit_frames
        else pd.DataFrame(columns=audit_columns)
    )
    audit.reindex(columns=audit_columns).to_csv(
        output_dir / "count_mismatches.csv", index=False
    )

    rows_by_subclass: defaultdict[str, int] = defaultdict(int)
    oids_by_subclass: defaultdict[str, int] = defaultdict(int)
    detection_bytes = 0
    for status in statuses:
        detection_bytes += int(status["detection_file_bytes"])
        for name, count in status["detection_rows_by_subclass"].items():
            rows_by_subclass[name] += int(count)
        for name, count in status["selected_oids_by_subclass"].items():
            oids_by_subclass[name] += int(count)

    full_counts = labels_all["classALeRCE"].value_counts().to_dict()
    estimated_full_rows = 0.0
    for subclass, full_count in full_counts.items():
        sampled_oids = oids_by_subclass.get(subclass, 0)
        if sampled_oids:
            estimated_full_rows += (
                rows_by_subclass.get(subclass, 0) / sampled_oids
            ) * full_count

    total_rows = sum(rows_by_subclass.values())
    bytes_per_row = detection_bytes / total_rows if total_rows else 0.0
    estimated_full_bytes = estimated_full_rows * bytes_per_row
    summary = {
        "completed_at_utc": utc_now(),
        "elapsed_seconds_this_run": round(time.monotonic() - started_at, 3),
        "output_format": output_format,
        "selected_oids": int(len(labels_selected)),
        "found_objects": int(len(objects)),
        "missing_oids": int(len(missing)),
        "detection_rows": int(total_rows),
        "detection_file_bytes": int(detection_bytes),
        "estimated_full_detection_rows": round(estimated_full_rows),
        "estimated_full_detection_bytes": round(estimated_full_bytes),
        "estimated_full_detection_gib": round(estimated_full_bytes / (1024**3), 3),
        "note": (
            "Estimasi berasal dari rerata detection per subkelas pada sampel dan ukuran "
            "partisi saat ini; ruang preprocessing/checkpoint model belum termasuk."
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "_SUCCESS", summary)

    print(
        f"Selesai: {len(objects)}/{len(labels_selected)} objek, "
        f"{total_rows:,} detection. Estimasi partisi detection penuh: "
        f"{summary['estimated_full_detection_gib']} GiB.",
        flush=True,
    )


def run(
    label_path: Path,
    output_dir: Path,
    per_subclass: int,
    batch_size: int,
    maxrec: int,
    seed: int,
    resume: bool,
    output_format: str,
) -> None:
    started_at = time.monotonic()
    labels_all = read_labels(label_path)
    labels = choose_sample(labels_all, per_subclass, seed)
    if labels.empty:
        raise ValueError("Katalog tidak memiliki label yang dapat digunakan")

    success_path = output_dir / "_SUCCESS"
    if success_path.exists():
        raise FileExistsError(f"Akuisisi sudah selesai: {success_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Folder output tidak kosong: {output_dir}; gunakan folder baru atau --resume"
        )

    paths = prepare_directories(output_dir)
    config_path = output_dir / "config.json"
    selected_path = output_dir / "labels_selected.csv"
    config = {
        "tap_url": TAP_URL,
        "label_file": str(label_path),
        "selection_sha256": selection_hash(labels),
        "per_subclass": per_subclass,
        "batch_size": batch_size,
        "maxrec": maxrec,
        "seed": seed,
        "output_format": output_format,
        "selected_oids": int(len(labels)),
        "created_at_utc": utc_now(),
    }

    if resume:
        if not config_path.exists() or not selected_path.exists():
            raise FileNotFoundError("--resume memerlukan config.json dan labels_selected.csv")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        keys = [
            "selection_sha256",
            "per_subclass",
            "batch_size",
            "maxrec",
            "seed",
            "output_format",
        ]
        changed = [key for key in keys if previous.get(key) != config.get(key)]
        if changed:
            raise ValueError(
                f"Konfigurasi resume berbeda pada {changed}; gunakan argumen awal yang sama"
            )
    else:
        labels.to_csv(selected_path, index=False)
        atomic_json(config_path, config)

    tap = pyvo.dal.TAPService(TAP_URL)
    oid_list = labels["oid"].tolist()
    batches = [oid_list[i : i + batch_size] for i in range(0, len(oid_list), batch_size)]

    for batch_index, batch in enumerate(batches):
        if load_status(paths["status"], batch_index) is not None:
            print(f"Batch {batch_index + 1}/{len(batches)} sudah selesai; dilewati.", flush=True)
            continue

        batch_started = time.monotonic()
        batch_labels = labels[labels["oid"].isin(batch)].copy()
        print(
            f"Batch {batch_index + 1}/{len(batches)}: mengambil ztf.object "
            f"untuk {len(batch)} oid...",
            flush=True,
        )
        objects = fetch_objects(tap, batch)
        found_objects = set(objects["oid"])
        missing_object = sorted(set(batch) - found_objects)

        detections = (
            fetch_detections(tap, objects, maxrec)
            if not objects.empty
            else pd.DataFrame(columns=DETECTION_COLUMNS)
        )
        count_audit = audit_detection_counts(objects, detections) if not objects.empty else []
        found_detections = set(detections["oid"])
        no_detection = sorted(found_objects - found_detections)

        data = detections.merge(objects, on="oid", how="left", validate="many_to_one")
        data = data.merge(
            batch_labels[["oid", "classALeRCE", "classMacro"]],
            on="oid",
            how="left",
            validate="many_to_one",
        )
        if not data.empty and data[["ndet", "classALeRCE", "classMacro"]].isna().any().any():
            raise ValueError("Hasil join memiliki metadata atau label kosong")
        data = data.reindex(columns=OUTPUT_COLUMNS)

        detection_path = part_path(paths["detections"], batch_index, output_format)
        object_path = part_path(paths["object_parts"], batch_index, output_format)
        missing_path = part_path(paths["missing_parts"], batch_index, output_format)
        audit_path = part_path(paths["audit_parts"], batch_index, output_format)

        missing_rows = [
            {"oid": oid, "reason": "not_found_in_ztf_object"} for oid in missing_object
        ] + [{"oid": oid, "reason": "object_has_no_detection"} for oid in no_detection]
        write_empty_or_data(data, detection_path, output_format, OUTPUT_COLUMNS)
        write_empty_or_data(objects, object_path, output_format, OBJECT_COLUMNS)
        write_empty_or_data(
            pd.DataFrame(missing_rows), missing_path, output_format, ["oid", "reason"]
        )
        write_empty_or_data(
            pd.DataFrame(count_audit),
            audit_path,
            output_format,
            ["oid", "detections_received", "object_ndet", "difference", "reason"],
        )

        status = {
            "batch_index": batch_index,
            "completed_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - batch_started, 3),
            "requested_oids": len(batch),
            "found_objects": len(found_objects),
            "missing_oids": len(missing_rows),
            "detection_rows": len(data),
            "detection_file_bytes": detection_path.stat().st_size,
            "selected_oids_by_subclass": {
                str(key): int(value)
                for key, value in batch_labels["classALeRCE"].value_counts().items()
            },
            "detection_rows_by_subclass": {
                str(key): int(value)
                for key, value in data["classALeRCE"].value_counts().items()
            },
        }
        atomic_json(paths["status"] / f"batch-{batch_index:06d}.json", status)
        print(
            f"  Tersimpan: {len(found_objects)}/{len(batch)} objek, "
            f"{len(data):,} detection.",
            flush=True,
        )

    finalize(
        output_dir,
        paths,
        labels_all,
        labels,
        len(batches),
        output_format,
        started_at,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path(LABEL_FILE))
    parser.add_argument("--output", type=Path, default=Path("data/smoke"))
    parser.add_argument(
        "--per-subclass",
        "--per-class",
        dest="per_subclass",
        type=int,
        default=2,
        help=(
            "Maksimum OID yang dipilih dari masing-masing 15 subkelas; "
            "0 = seluruh labeled set"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Jumlah OID pada satu checkpoint batch (default: 50)",
    )
    parser.add_argument("--maxrec", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Format partisi detection; Parquet direkomendasikan",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Lanjutkan batch yang belum selesai"
    )
    args = parser.parse_args()

    if args.per_subclass < 0 or args.batch_size < 1 or args.maxrec < 2:
        parser.error("--per-subclass >= 0, --batch-size >= 1, --maxrec >= 2")

    run(
        args.labels,
        args.output,
        args.per_subclass,
        args.batch_size,
        args.maxrec,
        args.seed,
        args.resume,
        args.output_format,
    )


if __name__ == "__main__":
    main()
