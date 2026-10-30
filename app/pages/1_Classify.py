from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alerce_classifier.inference import InferenceConfig, LightCurveInference
from alerce_classifier.preprocessing.pipeline import prepare_detection_frame
from alerce_classifier.clients.tap import AlerceTapClient


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts"
LABELS_PATH = ROOT / "labeled_set_lc_classifier_SanchezSaez_2020.csv"
LANDING_IMAGE = ROOT / "app" / "assets" / "landing_page.png"

@st.cache_resource
def load_inference() -> LightCurveInference:
    return LightCurveInference(
        InferenceConfig(
            checkpoint_path=ARTIFACT_DIR / "derived_hybrid_cnn_gru_seed42_best.pt",
            normalization_path=ARTIFACT_DIR / "derived_normalization.json",
            feature_config_path=ARTIFACT_DIR / "feature_config.json",
        )
    )


@st.cache_resource
def load_tap_client() -> AlerceTapClient:
    return AlerceTapClient()


@st.cache_data(ttl=900)
def load_object_data(oid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = load_tap_client()
    return client.fetch_object(oid), client.fetch_detections(oid)


@st.cache_data
def load_labeled_oids() -> tuple[str, ...]:
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Dataset label tidak ditemukan: {LABELS_PATH}")
    labels = pd.read_csv(LABELS_PATH, usecols=["oid"], dtype={"oid": "string"})
    oids = tuple(labels["oid"].dropna().astype(str).drop_duplicates())
    if not oids:
        raise ValueError("Dataset label tidak berisi OID yang valid.")
    return oids


def random_labeled_oid() -> str:
    oids = load_labeled_oids()
    return str(oids[np.random.randint(0, len(oids))])


def plot_light_curve(detections: pd.DataFrame, early_count: int | None) -> None:
    valid = detections.sort_values(["mjd", "candid"], kind="mergesort").copy()
    fig = go.Figure()
    colors = {1: "#1f77b4", 2: "#d62728"}
    labels = {1: "g", 2: "r"}
    for fid in (1, 2):
        subset = valid[valid["fid"] == fid]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["mjd"],
                y=subset["magnitude"],
                mode="markers",
                name=f"filter {labels[fid]}",
                marker={"color": colors[fid], "size": 7},
                error_y={
                    "type": "data",
                    "array": subset["magnitude_error"],
                    "visible": True,
                },
                customdata=subset[["candid", "oid", "magnitude_error"]].to_numpy(),
                hovertemplate=(
                    "MJD: %{x:.5f}<br>Magnitude: %{y:.4f}"
                    "<br>Error: %{customdata[2]:.4f}"
                    "<br>candid: %{customdata[0]}<br>OID: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )
    if early_count is not None and len(valid) >= early_count:
        boundary = valid.iloc[early_count - 1]["mjd"]
        early = valid.iloc[:early_count]
        fig.add_vline(
            x=boundary,
            line_dash="dash",
            line_color="#9467bd",
            annotation_text="batas 6 detection",
        )
        fig.add_trace(
            go.Scatter(
                x=early["mjd"],
                y=early["magnitude"],
                mode="markers",
                name="6 detection awal",
                marker={
                    "color": "#9467bd",
                    "size": 13,
                    "symbol": "circle-open",
                    "line": {"width": 2},
                },
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        title="Kurva cahaya ZTF",
        xaxis_title="MJD",
        yaxis_title="Magnitude",
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


def prepare_stamp_image(image: object) -> np.ndarray:
    """Convert an ALeRCE stamp array to a Streamlit-compatible uint8 image."""
    array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim not in (2, 3):
        raise ValueError(f"Dimensi stamp tidak didukung: {array.shape}")
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        raise ValueError(f"Format channel stamp tidak didukung: {array.shape}")

    values = np.asarray(array, dtype="float32")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Stamp hanya berisi nilai non-finite.")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    finite_values = values[finite]
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if minimum >= 0.0 and maximum <= 1.0:
        values *= 255.0
    elif maximum > minimum:
        values = (values - minimum) / (maximum - minimum) * 255.0
    else:
        values.fill(0.0)
    return np.clip(values, 0.0, 255.0).astype("uint8")


def render_stamp(image: object, caption: str) -> None:
    try:
        st.image(prepare_stamp_image(image), caption=caption)
    except (TypeError, ValueError) as exc:
        st.warning(f"{caption} tidak dapat ditampilkan: {exc}")


def show_stamps(oid: str, detections: pd.DataFrame) -> None:
    st.subheader("Visual stamp ALeRCE")
    candidates = detections.dropna(subset=["candid"]).sort_values("mjd")
    if candidates.empty:
        st.info("Tidak ada candid yang dapat digunakan untuk mengambil stamp.")
        return
    options = candidates["candid"].astype("int64").tolist()
    selected = st.selectbox("Detection untuk stamp", options, format_func=str)
    if not st.button("Tampilkan science/reference/difference stamp"):
        return
    try:
        from alerce_classifier.clients.stamp import AlerceStampClient

        stamps = AlerceStampClient().get_stamps(oid, int(selected))
    except RuntimeError as exc:
        st.warning(str(exc))
        st.caption("Kurva cahaya dan inference tetap dapat digunakan.")
        return
    if isinstance(stamps, dict):
        names = ("science", "reference", "difference")
        columns = st.columns(3)
        for column, name in zip(columns, names):
            with column:
                image = stamps.get(name)
                if image is None:
                    st.info(f"{name.capitalize()} stamp tidak tersedia.")
                else:
                    render_stamp(image, name.capitalize())
    elif isinstance(stamps, (list, tuple)) and len(stamps) >= 3:
        columns = st.columns(3)
        for column, name, image in zip(
            columns, ("Science", "Reference", "Difference"), stamps[:3]
        ):
            with column:
                render_stamp(image, name)
    else:
        st.info("Stamp berhasil diambil, tetapi format respons belum dapat dirender.")


def render_result(result: dict) -> None:
    mode_label = (
        "KLASIFIKASI FINAL — SELURUH SEQUENCE"
        if result["mode"] == "full"
        else "KLASIFIKASI AWAL — 6 DETECTION PERTAMA"
    )
    st.markdown(f"### {mode_label}")
    st.success(f"Prediksi: **{result['class_name']}**")
    st.metric("Confidence model", f"{result['confidence']:.2%}")
    probabilities = pd.DataFrame(
        {"Kelas": list(result["probabilities"]), "Probabilitas": list(result["probabilities"].values())}
    ).set_index("Kelas")
    st.bar_chart(probabilities)
    if result["truncated"]:
        st.warning("Sequence melebihi panjang maksimum training dan dipotong kronologis.")
    if result["mode"] == "early":
        st.caption("Prediksi berdasarkan enam detection awal; hasil dapat berubah setelah detection tambahan.")


def main() -> None:
    st.title("ZTF Classifier with Deep Learning Model")
    if LANDING_IMAGE.is_file():
        st.image(LANDING_IMAGE, use_container_width=True)
    st.write(
        "Choose a labeled ZTF object at random or enter an OID from the "
        "ALeRCE catalogue."
    )

    randomize_col, _ = st.columns([1, 3])
    with randomize_col:
        if st.button("Randomize", use_container_width=True):
            try:
                st.session_state["oid_input"] = random_labeled_oid()
                st.session_state.pop("object_data", None)
            except (FileNotFoundError, ValueError) as exc:
                st.error(f"Unable to randomize an object: {exc}")
    st.caption("You can randomize an object or enter an OID manually below.")
    oid_input = st.text_input(
        "Object ID (OID)",
        placeholder="For example: ZTF18abkifng",
        key="oid_input",
    )
    mode = st.radio(
        "Classification mode",
        options=("full", "early"),
        format_func=lambda value: (
            "Full sequence classification" if value == "full"
            else "Early classification — first 6 detections"
        ),
    )
    load_clicked = st.button("Classify", type="primary", use_container_width=True)

    if not load_clicked and "object_data" not in st.session_state:
        st.info("Enter an OID or randomize an object to begin.")
        return
    if load_clicked:
        oid = oid_input.strip()
        if not oid:
            st.error("Please enter an OID or randomize an object first.")
            return
        st.session_state["oid"] = oid
        try:
            st.session_state["object_data"] = load_object_data(oid)
        except Exception as exc:
            st.error(f"Unable to load object data: {exc}")
            return
    oid = st.session_state["oid"]
    object_frame, detections = st.session_state["object_data"]
    if object_frame.empty or detections.empty:
        st.error("OID tidak ditemukan atau tidak memiliki detection.")
        return

    valid = detections[detections["fid"].isin([1, 2])].copy()
    display_detections = prepare_detection_frame(valid)
    if display_detections.empty:
        st.error("Objek tidak memiliki detection g/r yang valid.")
        return
    early_count = 6 if len(display_detections) >= 6 else None
    try:
        inference = load_inference()
        result = inference.predict(valid, mode=mode)
    except ValueError as exc:
        if mode == "early" and len(valid) < 6:
            st.warning("Early classification tidak tersedia: objek memiliki kurang dari 6 detection valid.")
            result = None
        else:
            st.error(f"Inference tidak dapat dijalankan: {exc}")
            return
    except Exception as exc:
        st.error(f"Model atau preprocessing gagal: {exc}")
        return

    if result is not None:
        render_result(result)
    plot_light_curve(display_detections, early_count)
    with st.expander("Detail objek", expanded=False):
        row = object_frame.iloc[0]
        duration = float(row["lastmjd"] - row["firstmjd"])
        cols = st.columns(4)
        cols[0].metric("OID", oid)
        cols[1].metric("RA / DEC", f"{row['meanra']:.6f}, {row['meandec']:.6f}")
        cols[2].metric("Durasi", f"{duration:.2f} hari")
        cols[3].metric("Detection", len(display_detections))
        st.write(
            "Filter tersedia: "
            f"{', '.join(sorted(display_detections['fid'].astype(str).unique()))}"
        )
        st.write("Early classification:", "tersedia" if early_count else "tidak tersedia")
    with st.expander("Lihat visual gambar objek", expanded=False):
        show_stamps(oid, valid)


if __name__ == "__main__":
    main()
