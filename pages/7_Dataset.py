import streamlit as st
import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Dataset",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Dataset Analisis Sentimen")
st.markdown("---")

# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/dataset_preprocessing.csv"

if not os.path.exists(DATA_PATH):

    st.warning("Dataset belum tersedia.")

    st.stop()

dataset = pd.read_csv(DATA_PATH)

# ============================================================
# HEADER
# ============================================================

st.success(f"Total Dataset : {len(dataset)} Review")

st.markdown("---")

# ============================================================
# FILTER
# ============================================================

col1, col2 = st.columns(2)

with col1:

    keyword = st.text_input(
        "🔍 Cari Review"
    )

with col2:

    sentiment = st.selectbox(

        "Filter Sentimen",

        [

            "Semua",

            "positif",

            "negatif",

            "netral"

        ]

    )

# ============================================================
# FILTER DATA
# ============================================================

filtered = dataset.copy()

if keyword != "":

    filtered = filtered[
        filtered["content"].str.contains(
            keyword,
            case=False,
            na=False
        )
    ]

if sentiment != "Semua":

    filtered = filtered[
        filtered["sentiment"] == sentiment
    ]

# ============================================================
# DATAFRAME
# ============================================================

st.dataframe(

    filtered,

    use_container_width=True,

    height=550

)

st.markdown("---")

# ============================================================
# DOWNLOAD
# ============================================================

csv = filtered.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(

    label="⬇ Download Dataset",

    data=csv,

    file_name="dataset_preprocessing.csv",

    mime="text/csv"

)

st.markdown("---")

# ============================================================
# STATISTIK
# ============================================================

st.subheader("📊 Statistik Dataset")

col1,col2,col3,col4 = st.columns(4)

col1.metric(

    "Total Review",

    len(dataset)

)

col2.metric(

    "Positif",

    len(
        dataset[
            dataset["sentiment"]=="positif"
        ]
    )

)

col3.metric(

    "Negatif",

    len(
        dataset[
            dataset["sentiment"]=="negatif"
        ]
    )

)

col4.metric(

    "Netral",

    len(
        dataset[
            dataset["sentiment"]=="netral"
        ]
    )

)