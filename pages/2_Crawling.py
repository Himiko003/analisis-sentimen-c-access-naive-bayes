import streamlit as st
import pandas as pd
from google_play_scraper import reviews_all, Sort
import os

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Crawling Data",
    page_icon="📥",
    layout="wide"
)

st.title("📥 Crawling Data Google Play Store")
st.markdown("---")

# ============================================================
# INPUT
# ============================================================

app_id = st.text_input(
    "Package Name",
    value="com.kci.access"
)

jumlah_data = st.number_input(
    "Jumlah Maksimal Data",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

st.info(
    "Package aplikasi C-Access : com.kci.access"
)

# ============================================================
# BUTTON
# ============================================================

if st.button("📥 Mulai Crawling", use_container_width=True):

    with st.spinner("Mengambil data dari Google Play Store..."):

        review = reviews_all(
            app_id,
            lang="id",
            country="id",
            sort=Sort.NEWEST
        )

        dataset = pd.DataFrame(review)

        dataset = dataset[
            [
                "content",
                "score",
                "at"
            ]
        ]

        dataset.columns = [
            "content",
            "score",
            "date"
        ]

        dataset = dataset.head(jumlah_data)

        os.makedirs("data", exist_ok=True)

        dataset.to_csv(
            "data/raw_dataset.csv",
            index=False,
            encoding="utf-8-sig"
        )

    st.success("Crawling berhasil!")

    st.write(f"Jumlah Review : {len(dataset)}")

    st.dataframe(
        dataset.head(10),
        use_container_width=True
    )