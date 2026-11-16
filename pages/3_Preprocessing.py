import streamlit as st
import pandas as pd
import re
import string
import os

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Preprocessing",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Preprocessing Data")
st.markdown("---")

# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/raw_dataset.csv"

if not os.path.exists(DATA_PATH):
    st.error("raw_dataset.csv belum tersedia.")
    st.stop()

dataset = pd.read_csv(DATA_PATH)

# ============================================================
# NLP
# ============================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words("indonesian"))

custom_stopwords = {
    "aplikasi",
    "caccess",
    "access",
    "commuter",
    "kci"
}

stop_words.update(custom_stopwords)

normalization_dict = {

    "gk":"tidak",
    "ga":"tidak",
    "gak":"tidak",
    "nggak":"tidak",
    "tdk":"tidak",

    "yg":"yang",
    "dgn":"dengan",
    "dr":"dari",
    "krn":"karena",
    "utk":"untuk",

    "bgt":"banget",
    "bgtt":"banget",
    "bngt":"banget",

    "udh":"sudah",
    "udah":"sudah",

    "dpt":"dapat",
    "trs":"terus",
    "aja":"saja"
}

# ============================================================
# FUNCTION
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize(text):

    words = text.split()

    words = [
        normalization_dict[word]
        if word in normalization_dict
        else word
        for word in words
    ]

    return words


def remove_stopword(words):

    return [
        word
        for word in words
        if word not in stop_words
    ]


def stemming(words):

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


def label(score):

    if score >= 4:
        return "positif"

    elif score == 3:
        return "netral"

    else:
        return "negatif"

# ============================================================
# BUTTON
# ============================================================

if st.button("🧹 Mulai Preprocessing", use_container_width=True):

    with st.spinner("Memproses data..."):

        dataset.dropna(inplace=True)

        dataset.drop_duplicates(
            subset="content",
            inplace=True
        )

        dataset.reset_index(
            drop=True,
            inplace=True
        )

        dataset["sentiment"] = dataset["score"].apply(label)

        dataset["clean_text"] = dataset["content"].apply(clean_text)

        dataset["normalized"] = dataset["clean_text"].apply(normalize)

        dataset["filtered"] = dataset["normalized"].apply(remove_stopword)

        dataset["preprocessed_text"] = dataset["filtered"].apply(stemming)

        dataset.to_csv(
            "data/dataset_preprocessing.csv",
            index=False,
            encoding="utf-8-sig"
        )

    st.success("Preprocessing berhasil!")

    st.write(f"Jumlah Data : {len(dataset)}")

    st.dataframe(
        dataset.head(10),
        use_container_width=True
    )