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
SELECTED_LABELS_PATH = (
    ROOT
    / "data"
    / "alerce_ztf_lc_sanchezsaez_full"
    / "data"
    / "full"
    / "labels_selected.csv"
)
LANDING_IMAGE = ROOT / "app" / "assets" / "landing_page.png"
FALLBACK_SAMPLE_OIDS = {
    "Periodic": "ZTF18aaxxonc",
    "Stochastic": "ZTF18aabtxvd",
    "Transient": "ZTF18aaqeasu",
}
MACRO_CLASS_MAP = {
    "LPV": "Periodic",
    "E": "Periodic",
    "DSCT": "Periodic",
    "RRL": "Periodic",
    "CEP": "Periodic",
    "Periodic-Other": "Periodic",
    "AGN": "Stochastic",
    "QSO": "Stochastic",
    "Blazar": "Stochastic",
    "YSO": "Stochastic",
    "CV/Nova": "Stochastic",
    "SNIa": "Transient",
    "SNIbc": "Transient",
    "SNII": "Transient",
    "SLSN": "Transient",
}

@st.cache_resource
def load_inference() -> LightCurveInference:
    return LightCurveInference(
        InferenceConfig(
            checkpoint_path=ARTIFACT_DIR / "hybrid" / "hybrid_cnn_gru_seed42_best.pt",
            normalization_path=ARTIFACT_DIR / "normalization.json",
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


@st.cache_data
def get_sample_oids() -> dict[str, str]:
    candidates = (SELECTED_LABELS_PATH, LABELS_PATH)

    for path in candidates:
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        if "oid" not in frame.columns:
            continue

        if "classMacro" in frame.columns:
            frame["sampleMacro"] = frame["classMacro"].astype("string")
        elif "classALeRCE" in frame.columns:
            frame["sampleMacro"] = (
                frame["classALeRCE"].astype("string").map(MACRO_CLASS_MAP)
            )
        else:
            continue

        frame = frame.dropna(subset=["oid", "sampleMacro"]).copy()
        frame["oid"] = frame["oid"].astype("string")
        if "ndet" in frame.columns:
            frame = frame[
                pd.to_numeric(frame["ndet"], errors="coerce").fillna(0) >= 30
            ]

        result = {}
        for macro_name in ("Periodic", "Stochastic", "Transient"):
            match = frame.loc[frame["sampleMacro"].eq(macro_name), "oid"]
            if not match.empty:
                result[macro_name] = str(match.iloc[0])
        if len(result) == 3:
            return result

    return dict(FALLBACK_SAMPLE_OIDS)


def handle_quick_sample(oid: str) -> None:
    st.session_state["oid_input"] = oid
    st.session_state["auto_classify"] = True
    st.session_state.pop("object_data", None)


def handle_randomize() -> None:
    try:
        st.session_state["oid_input"] = random_labeled_oid()
        st.session_state.pop("object_data", None)
    except (FileNotFoundError, ValueError) as exc:
        st.session_state["randomize_error"] = str(exc)


def plot_light_curve(detections: pd.DataFrame, early_count: int | None) -> None:
    valid = detections.sort_values(["mjd", "candid"], kind="mergesort").copy()
    fig = go.Figure()
    colors = {1: "#2ECC71", 2: "#d62728"}
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
        height=265,
        margin={"l": 45, "r": 15, "t": 35, "b": 35},
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
        st.image(
            prepare_stamp_image(image),
            caption=caption,
            width=100,
        )
    except (TypeError, ValueError) as exc:
        st.warning(f"{caption} tidak dapat ditampilkan: {exc}")


def show_stamps(oid: str, detections: pd.DataFrame) -> None:
    candidates = detections.dropna(subset=["candid"]).sort_values("mjd")
    if candidates.empty:
        st.info("Tidak ada candid yang dapat digunakan untuk mengambil stamp.")
        return

    options = candidates["candid"].astype("int64").tolist()
    select_key = f"stamp_candid_{oid}"
    if select_key not in st.session_state:
        st.session_state[select_key] = int(options[-1])

    selected = st.selectbox(
        "Pilih Deteksi Stamp",
        options,
        index=max(0, options.index(int(st.session_state[select_key])))
        if int(st.session_state[select_key]) in options
        else len(options) - 1,
        format_func=lambda candid: f"Candid {candid}",
        key=select_key,
    )

    try:
        from alerce_classifier.clients.stamp import AlerceStampClient

        stamps = AlerceStampClient().get_stamps(oid, int(selected))
    except RuntimeError as exc:
        st.warning(str(exc))
        st.caption("Kurva cahaya dan inference tetap dapat digunakan.")
        return

    if isinstance(stamps, dict):
        reference = stamps.get("reference")
        science = stamps.get("science")
        difference = stamps.get("difference")
    elif isinstance(stamps, (list, tuple)) and len(stamps) >= 3:
        science, reference, difference = stamps[:3]
    else:
        st.info("Stamp berhasil diambil, tetapi format respons belum dapat dirender.")
        return

    s1, s2, s3 = st.columns(3, gap="small")
    with s1:
        if reference is None:
            st.info("Reference stamp tidak tersedia.")
        else:
            render_stamp(reference, "Reference")
    with s2:
        if science is None:
            st.info("Science stamp tidak tersedia.")
        else:
            render_stamp(science, "Science")
    with s3:
        if difference is None:
            st.info("Difference stamp tidak tersedia.")
        else:
            render_stamp(difference, "Difference")


def render_result(result: dict) -> None:
    mode_label = {
        "full": "Full Sequence",
        "early": "Early Classification dengan 6 detection",
        "early_fallback": "Fallback Full Sequence",
    }.get(str(result["mode"]), "Full Sequence")
    probability_rows = ""
    for class_name, probability in result["probabilities"].items():
        percentage = float(probability) * 100
        probability_rows += (
            '<div class="ztf-probability">'
            f'<div><span>{class_name}</span><strong>{percentage:.2f}%</strong></div>'
            f'<div class="ztf-probability-track">'
            f'<div class="ztf-probability-fill" style="width:{percentage:.2f}%"></div>'
            "</div></div>"
        )

    st.markdown(
        '<div class="ztf-result-card">'
        '<h3>HASIL KLASIFIKASI</h3>'
        f'<div class="ztf-mode-subtitle">{mode_label}</div>'
        '<div class="ztf-prediction">'
        f'<span>Prediksi</span><strong>{result["class_name"]}</strong>'
        "</div>"
        '<div class="ztf-confidence">'
        f'<span>Confidence model</span><strong>{result["confidence"]:.2%}</strong>'
        "</div>"
        f'<div class="ztf-probabilities">{probability_rows}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if result["truncated"]:
        st.warning("Sequence melebihi panjang maksimum training dan dipotong kronologis.")


def main() -> None:
    if "oid_input" not in st.session_state:
        st.session_state["oid_input"] = ""
    if "auto_classify" not in st.session_state:
        st.session_state["auto_classify"] = False

    st.markdown(
        """
        <style>
        .block-container {max-width: 1450px; padding-top: 2.5rem; padding-bottom: 1.5rem;}
        .ztf-header-image img {
            width: 100%;
            max-height: 140px;
            object-fit: cover;
            border-radius: 10px;
            border: 1px solid rgba(49, 51, 63, 0.12);
        }
        .ztf-header-image {margin-bottom: 1.25rem; margin-top: 0.25rem;}
        .ztf-title {margin-bottom: 0.15rem; font-size: 2rem; font-weight: 700;}
        .ztf-subtitle {color: #64748b; margin-bottom: 0.65rem;}
        .ztf-empty {
            min-height: 365px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px dashed rgba(49, 51, 63, 0.22);
            border-radius: 12px;
            background: rgba(248, 250, 252, 0.72);
            color: #64748b;
        }
        .ztf-empty-icon {font-size: 2.2rem; margin-bottom: 0.35rem;}
        [data-testid="stMetric"] {padding: 0.15rem 0;}
        [data-testid="stHorizontalBlock"] {align-items: center;}
        .ztf-result-card {
            height: 280px;
            overflow: hidden;
            # padding: 0.05rem 0.15rem;
        }
        .ztf-result-card h3 {
            # margin: 1rem 0 1rem 0;
            font-size: 1.15rem;
            letter-spacing: 0.02em;
        }
        .ztf-mode-subtitle {
            color: #64748b;
            font-size: 0.82rem;
            # margin: 0.15rem 0 0.45rem 0;
        }
        .ztf-prediction, .ztf-confidence {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            border-bottom: 1px solid rgba(49, 51, 63, 0.10);
            padding: 0.35rem 0;
        }
        .ztf-prediction strong {
            color: #2563eb;
            font-size: 1.15rem;
        }
        .ztf-confidence strong {font-size: 1.05rem;}
        .ztf-probabilities {margin-top: 0.45rem;}
        .ztf-probability {margin: 0.38rem 0;}
        .ztf-probability > div:first-child {
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .ztf-probability-track {
            height: 7px;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
        }
        .ztf-probability-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2563eb, #38bdf8);
        }
        .ztf-metadata-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            align-items: stretch;
            gap: 0.65rem;
            margin-top: 0.35rem;
            margin-bottom: 1rem;
        }
        .ztf-meta-card {
            min-height: 88px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow-wrap: anywhere;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 9px;
            padding: 0.75rem 0.85rem;
        }
        .ztf-meta-card span {
            color: #9da7b3;
            font-size: 0.72rem;
            margin-bottom: 0.35rem;
        }
        .ztf-meta-card strong {
            color: #f0f6fc;
            font-size: 0.92rem;
            line-height: 1.25;
        }
        .ztf-meta-card small {
            color: #8b949e;
            font-size: 0.68rem;
            margin-top: 0.28rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_left, header_right = st.columns([1, 3], gap="large")
    with header_left:
        if LANDING_IMAGE.is_file():
            import base64

            image_data = base64.b64encode(LANDING_IMAGE.read_bytes()).decode("utf-8")
            st.markdown(
                f'<div class="ztf-header-image">'
                f'<img src="data:image/png;base64,{image_data}" alt="ZTF telescope">'
                f"</div>",
                unsafe_allow_html=True,
            )
    with header_right:
        st.markdown(
            '<div class="ztf-title">ZTF Light Curve Classifier</div>'
            '<div class="ztf-subtitle">'
            'Klasifikasi kurva cahaya astronomi menggunakan Hybrid 1D-CNN + GRU'
            "</div>",
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        input_col, mode_col, action_col = st.columns([2, 1, 2], gap="large")
        with input_col:
            oid_input = st.text_input(
                "Object ID (OID)",
                placeholder="Contoh: ZTF18abklfng",
                key="oid_input",
            )
            st.button(
                "Randomize sample",
                on_click=handle_randomize,
                use_container_width=True,
            )
            st.caption("Quick Demo Samples:")
            sample_oids = get_sample_oids()
            sample_cols = st.columns(3)
            for column, macro_name in zip(sample_cols, ("Periodic", "Stochastic", "Transient")):
                with column:
                    oid = sample_oids[macro_name]
                    st.button(
                        macro_name,
                        key=f"quick_{macro_name.lower()}",
                        on_click=handle_quick_sample,
                        args=(oid,),
                        use_container_width=True,
                        help=f"OID contoh: {oid}",
                    )
            if "randomize_error" in st.session_state:
                st.error(
                    "Randomisasi objek gagal: "
                    f"{st.session_state.pop('randomize_error')}"
                )
        with mode_col:
            mode = st.radio(
                "Classification mode",
                options=("full", "early"),
                format_func=lambda value: (
                    "Full sequence" if value == "full"
                    else "Early Classification"
                ),
            )
        with action_col:
            if mode == "full":
                st.caption(
                    "Menggunakan seluruh sequence yang tersedia untuk klasifikasi final."
                )
            else:
                st.caption(
                    "Menggunakan enam detection pertama untuk estimasi klasifikasi dini."
                )
            load_clicked = st.button(
                "Classify object",
                type="primary",
                use_container_width=True,
            )

    auto_classify = bool(st.session_state.pop("auto_classify", False))

    if not load_clicked and not auto_classify and "object_data" not in st.session_state:
        st.markdown(
            '<div class="ztf-empty">'
            '<div class="ztf-empty-icon">🔭</div>'
            '<strong>Siap memulai klasifikasi</strong>'
            '<span>Pilih atau masukkan OID, lalu tekan “Classify object”.</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        return
    if load_clicked or auto_classify:
        oid = st.session_state["oid_input"].strip() if auto_classify else oid_input.strip()
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
    if len(display_detections) < 6:
        st.warning(
            "Objek ini memiliki kurang dari 6 deteksi g/r. "
            "Jumlah tersebut berada di bawah ambang batas minimal model operasional "
            "ALeRCE BRF, tetapi proses inferensi Hybrid 1D-CNN + GRU tetap dilanjutkan."
        )
    early_count = 6 if mode == "early" and len(display_detections) >= 6 else None
    inference_mode = mode
    if mode == "early" and len(display_detections) < 6:
        inference_mode = "full"
    try:
        inference = load_inference()
        result = inference.predict(valid, mode=inference_mode)
        if mode == "early" and len(display_detections) < 6:
            result["mode"] = "early_fallback"
    except ValueError as exc:
        st.error(f"Inference tidak dapat dijalankan: {exc}")
        return
    except Exception as exc:
        st.error(f"Model atau preprocessing gagal: {exc}")
        return

    r1_left, r1_right = st.columns([1, 1.4], gap="medium")
    with r1_left:
        if result is not None:
            with st.container(border=True):
                render_result(result)
        else:
            st.container(border=True).markdown(
                "**Early classification tidak tersedia**\n\n"
                "Objek memerlukan minimal 6 detection valid."
            )
    with r1_right:
        with st.container(border=True):
            plot_light_curve(display_detections, early_count)

    r2_left, r2_right = st.columns([1.15, 1], gap="medium")
    with r2_left:
        row = object_frame.iloc[0]
        duration = float(row["lastmjd"] - row["firstmjd"])
        active_filters = sorted(
            int(value) for value in display_detections["fid"].dropna().unique()
        )
        filter_badge = " & ".join(
            f"Filter {'g' if value == 1 else 'r'} ({value})"
            for value in active_filters
        )
        with st.container(border=True):
            st.markdown("### Detail Objek")
            st.markdown(
                '<div class="ztf-metadata-grid">'
                '<div class="ztf-meta-card">'
                '<span>Object ID (OID)</span>'
                f'<strong>{oid}</strong>'
                "</div>"
                '<div class="ztf-meta-card">'
                '<span>Koordinat (RA / DEC)</span>'
                f'<strong>{row["meanra"]:.6f}, {row["meandec"]:.6f}</strong>'
                "</div>"
                '<div class="ztf-meta-card">'
                '<span>Durasi Pengamatan</span>'
                f'<strong>{duration:.2f} hari</strong>'
                '<small>Rentang MJD aktif</small>'
                "</div>"
                '<div class="ztf-meta-card">'
                '<span>Total Deteksi</span>'
                f'<strong>{len(display_detections)} deteksi</strong>'
                f'<small>{filter_badge}</small>'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    with r2_right:
        with st.container(border=True):
            st.markdown("### Visual Stamp ALeRCE")
            show_stamps(oid, valid)


if __name__ == "__main__":
    main()
