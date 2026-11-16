
import streamlit as st
import pandas as pd
import joblib
import json
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Training Model", page_icon="🤖", layout="wide")
st.title("🤖 Training Model")
st.markdown("---")

DATA_PATH = "data/dataset_preprocessing.csv"

if not os.path.exists(DATA_PATH):
    st.warning("Silakan lakukan preprocessing terlebih dahulu.")
    st.stop()

dataset = pd.read_csv(DATA_PATH)

if "preprocessed_text" not in dataset.columns:
    st.error("Kolom preprocessed_text tidak ditemukan.")
    st.stop()

dataset["preprocessed_text"] = (
    dataset["preprocessed_text"]
    .fillna("")
    .astype(str)
)

dataset = dataset[
    dataset["preprocessed_text"].str.strip() != ""
].reset_index(drop=True)

if dataset.empty:
    st.error("Dataset kosong setelah preprocessing.")
    st.stop()

st.success(f"Dataset siap digunakan ({len(dataset)} data).")

if st.button("🚀 Mulai Training", use_container_width=True):

    with st.spinner("Training model..."):

        encoder = LabelEncoder()
        dataset["label"] = encoder.fit_transform(dataset["sentiment"])

        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            min_df=2,
            max_df=0.95
        )

        X = tfidf.fit_transform(dataset["preprocessed_text"])
        y = dataset["label"]

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = MultinomialNB()
        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average="weighted",zero_division=0)
        recall = recall_score(y_test,y_pred,average="weighted",zero_division=0)
        f1 = f1_score(y_test,y_pred,average="weighted",zero_division=0)

        cm = confusion_matrix(y_test,y_pred)

        report = classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        os.makedirs("model",exist_ok=True)

        joblib.dump(model,"model/model_naive_bayes.pkl")
        joblib.dump(tfidf,"model/tfidf_vectorizer.pkl")
        joblib.dump(encoder,"model/label_encoder.pkl")

        evaluation = {
            "accuracy": round(accuracy*100,2),
            "precision": round(precision*100,2),
            "recall": round(recall*100,2),
            "f1score": round(f1*100,2),
            "labels": encoder.classes_.tolist(),
            "total_data": int(len(dataset)),
            "train_data": int(len(y_train)),
            "test_data": int(len(y_test)),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

        with open(
            "model/evaluation.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                evaluation,
                f,
                indent=4,
                ensure_ascii=False
            )

    st.success("✅ Training berhasil dan model disimpan.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",f"{accuracy*100:.2f}%")
    c2.metric("Precision",f"{precision*100:.2f}%")
    c3.metric("Recall",f"{recall*100:.2f}%")
    c4.metric("F1 Score",f"{f1*100:.2f}%")

    st.markdown("---")
    st.subheader("Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )
    st.dataframe(cm_df,use_container_width=True)

    st.markdown("---")
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose(),use_container_width=True)
