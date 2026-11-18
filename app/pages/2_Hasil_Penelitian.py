from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "evaluation" / "benchmark" / "derived_detection_benchmark_results.json"
if not RESULTS.is_file():
    RESULTS = ROOT / "artifacts" / "evaluation" / "training" / "derived_experiment_results.json"

st.set_page_config(page_title="Result of the research", page_icon="Z", layout="wide")
st.title("Hasil Penelitian")
st.caption("Ringkasan eksperimen feature engineering dan early classification seed 42")

if not RESULTS.is_file():
    st.error(f"Artifact hasil tidak ditemukan: {RESULTS}")
    st.stop()

results = json.loads(RESULTS.read_text(encoding="utf-8"))
model_results = results.get("models", results)
rows = []
for model_name, payload in model_results.items():
    if not isinstance(payload, dict) or "test" not in payload:
        continue
    test = payload["test"]
    rows.append({
        "Model": model_name.replace("_seed42", "").replace("_", " "),
        "Accuracy": test["accuracy"],
        "Balanced accuracy": test["balanced_accuracy"],
        "Macro-precision": test.get(
            "macro_precision", test["classification_report"]["macro avg"]["precision"]
        ),
        "Macro-recall": test.get(
            "macro_recall", test["classification_report"]["macro avg"]["recall"]
        ),
        "Macro-F1": test["macro_f1"],
    })
st.subheader("Perbandingan model derived features")
st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.2%}"), use_container_width=True)
st.info(
    "Hybrid CNN-GRU digunakan sebagai model inference aplikasi karena menggabungkan "
    "ekstraksi pola lokal dan pemodelan temporal. GRU tetap ditampilkan sebagai pembanding."
)

early_results = results.get("early_classification")
if early_results is None and "models" in results:
    early_results = {
        model_name: payload["early_classification"]
        for model_name, payload in model_results.items()
        if "early_classification" in payload
    }
if early_results:
    st.subheader("Evaluasi prefix temporal berbasis jumlah detection")
    early_rows = []
    for model_name, prefixes in early_results.items():
        for detection_count, metrics in prefixes.items():
            early_rows.append({
                "Model": model_name.replace("_", " "),
                "Detection": detection_count,
                "Objek valid": metrics["valid_object_count"],
                "Coverage": metrics["coverage"],
                "Macro-precision": metrics["macro_precision"],
                "Macro-recall": metrics["macro_recall"],
                "Macro-F1": metrics["macro_f1"],
                "Balanced accuracy": metrics["balanced_accuracy"],
                "Accuracy": metrics["accuracy"],
            })
    st.dataframe(
        pd.DataFrame(early_rows).set_index(["Model", "Detection"]).style.format({
            "Coverage": "{:.2%}",
            "Macro-precision": "{:.2%}",
            "Macro-recall": "{:.2%}",
            "Macro-F1": "{:.2%}",
            "Balanced accuracy": "{:.2%}",
            "Accuracy": "{:.2%}",
        }),
        use_container_width=True,
    )

benchmarks = results.get("benchmarks")
if benchmarks is None and "models" in results:
    benchmarks = {
        model_name: payload["benchmark"]
        for model_name, payload in model_results.items()
        if "benchmark" in payload
    }
if benchmarks:
    st.subheader("Benchmark efisiensi komputasi")
    st.dataframe(
        pd.DataFrame.from_dict(benchmarks, orient="index"),
        use_container_width=True,
    )
