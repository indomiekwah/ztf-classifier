from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "derived_experiment_results.json"

st.set_page_config(page_title="Result of the research", page_icon="Z", layout="wide")
st.title("Hasil Penelitian")
st.caption("Ringkasan eksperimen feature engineering dan early classification seed 42")

if not RESULTS.is_file():
    st.error(f"Artifact hasil tidak ditemukan: {RESULTS}")
    st.stop()

results = json.loads(RESULTS.read_text(encoding="utf-8"))
rows = []
for model_name, payload in results.items():
    test = payload["test"]
    rows.append({
        "Model": model_name.replace("_seed42", "").replace("_", " "),
        "Accuracy": test["accuracy"],
        "Balanced accuracy": test["balanced_accuracy"],
        "Macro-F1": test["macro_f1"],
    })
st.subheader("Perbandingan model derived features")
st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.2%}"), use_container_width=True)
st.info(
    "Hybrid CNN-GRU digunakan sebagai model inference aplikasi karena menggabungkan "
    "ekstraksi pola lokal dan pemodelan temporal. GRU tetap ditampilkan sebagai pembanding."
)
