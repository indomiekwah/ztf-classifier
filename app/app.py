import streamlit as st

st.set_page_config(page_title="ZTF Classifier", page_icon="🌌", layout="wide")

classify_page = st.Page("pages/1_Classify.py", title="Classify object", default=True)
hasil_page = st.Page("pages/2_Hasil_Penelitian.py", title="Result of the research")
about_page = st.Page("pages/3_About.py", title="About")

pg = st.navigation(
    {"ZTF Classifier": [classify_page, hasil_page, about_page]}
)

with st.sidebar:
    st.caption("Classify an object from alerce ZTF catalogue")

pg.run()