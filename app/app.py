import streamlit as st


st.set_page_config(
    page_title="ZTF Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

classify_page = st.Page("pages/1_Classify.py", title="Classify object", default=True)
hasil_page = st.Page("pages/2_Hasil_Penelitian.py", title="Result of the research")
about_page = st.Page("pages/3_About.py", title="About")

pg = st.navigation(
    {"ZTF Classifier": [classify_page, hasil_page, about_page]},
    position="top",
)

with st.sidebar:
    # TODO: Tambahkan st.image("logo.png") di bagian paling atas sidebar.
    st.markdown(
        '<h2 style="color:#58a6ff;margin-bottom:0.35rem;">'
        "Welcome to ZTF Classifier!"
        "</h2>",
        unsafe_allow_html=True,
    )
    st.write(
        "Platform klasifikasi kurva cahaya objek astronomi dari survei "
        "Zwicky Transient Facility (ZTF) dan broker ALeRCE. Menggunakan "
        "arsitektur Hybrid 1D-CNN + GRU berbasis deret waktu mentah untuk "
        "mengidentifikasi objek Periodic, Stochastic, dan Transient secara real-time."
    )
    st.divider()
    st.link_button(
        "GitHub Repository",
        "https://github.com/indomiekwah/ztf-classifier",
        use_container_width=True,
    )
    st.caption("Author: **Clara Angelin Pijoh**")

pg.run()