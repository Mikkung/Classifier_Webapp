import sys, os
# Ensure we can import src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.data_loader import load_csv
from src.models import fit, save, load, predict
from src.evaluation import train_eval
from src.visualization import plot_confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classifier Web App", page_icon="🤖", layout="wide")
st.title("🤖 Tabular Classifier (scikit-learn)")
st.caption("Train & serve a tabular classifier with OneHotEncoder + Logistic/RandomForest.")

mode = st.sidebar.selectbox("Mode", ["Predict", "Train & Evaluate"])

def ensure_loaded():
    try:
        return load("artifacts"), None
    except Exception as e:
        return (None, None), str(e)

if mode == "Predict":
    st.subheader("Predict on a CSV")
    st.write("Upload a CSV with **the same schema as training** (features only).")
    (pipe, le), err = ensure_loaded()
    if err:
        st.info("No saved model found in `artifacts/`. Train one in the other tab.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file and pipe:
        df = pd.read_csv(file)
        labels, prob = predict(pipe, le, df)
        out = pd.DataFrame({"pred": labels, "confidence": (prob if prob is not None else [None]*len(labels))})
        st.dataframe(out.head(50))
        st.download_button("Download predictions CSV", data=out.to_csv(index=False), file_name="predictions.csv")

else:
    st.subheader("Train & Evaluate")
    use_sample = st.checkbox("Use bundled sample (data/sample.csv)", value=True)
    model = st.selectbox("Model", ["logreg","rf"])
    if use_sample:
        target = st.text_input("Target column", value="type")
        run = st.button("Run training")
        if run:
            df = load_csv("data/sample.csv")
            pipe, le, metrics, cm, test = train_eval(df, target=target, model=model)
            save(pipe, le, "artifacts")
            st.success("Training complete. Artifacts saved.")
            c1,c2 = st.columns(2)
            with c1: st.json({k: float(v) for k,v in metrics.items()})
            with c2:
                #fig = plt.figure(figsize=(5,5))
                #plot_confusion_matrix(cm, labels=list(le.classes_), show=False)
                #st.pyplot(fig)
                fig = plot_confusion_matrix(cm, labels=list(le.classes_), show=False)
                st.pyplot(fig)
            st.dataframe(test.head(20))
    else:
        file = st.file_uploader("Upload dataset", type=["csv"])
        target = st.text_input("Target column", value="label")
        run = st.button("Run training")
        if run and file:
            df = pd.read_csv(file)
            pipe, le, metrics, cm, test = train_eval(df, target=target, model=model)
            save(pipe, le, "artifacts")
            st.success("Training complete. Artifacts saved.")
            c1,c2 = st.columns(2)
            with c1: st.json({k: float(v) for k,v in metrics.items()})
            with c2:
                fig = plt.figure(figsize=(5,5))
                plot_confusion_matrix(cm, labels=list(le.classes_), show=False)
                st.pyplot(fig)
            st.dataframe(test.head(20))
