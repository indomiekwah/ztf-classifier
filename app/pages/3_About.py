from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
LANDING_IMAGE = ROOT / "app" / "assets" / "landing_page.png"

st.set_page_config(page_title="About | ZTF Classifier", page_icon="Z", layout="wide")
st.title("About")

st.subheader("Author")
st.write("Clara Angelin Pijoh")

st.subheader("Latar belakang penelitian")
st.write(
    "Penelitian ini mengembangkan sistem klasifikasi objek astronomi berdasarkan "
    "kurva cahaya ZTF dari ALeRCE. Sistem menggunakan Hybrid 1D-CNN dan GRU "
    "untuk mempelajari pola lokal serta informasi temporal pada sequence "
    "detection."
)
st.write(
    "Model menghasilkan tiga kelas makro: Transient, Stochastic, dan Periodic. "
    "Halaman klasifikasi menggunakan preprocessing yang sama dengan proses "
    "training agar hasil inference tetap konsisten."
)

st.subheader("Repository")
st.markdown(
    "[GitHub repository](https://github.com/indomiekwah/ztf-classifier)"
)
st.caption("Placeholder URL — replace this link with the actual repository URL.")

st.subheader("Acknowledgement")
st.write(
    "Penelitian ini menggunakan data dan layanan ALeRCE serta Fornax Science "
    "Console. Tambahkan credit gambar landing page di bagian ini setelah gambar "
    "dimasukkan ke repository."
)
if LANDING_IMAGE.is_file():
    st.caption("Landing page image credit: add the source and author here.")
