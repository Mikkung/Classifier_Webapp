import joblib, pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from .preprocess import make_preprocessor

def make_pipeline(df: pd.DataFrame, target: str, model="logreg"):
    pre = make_preprocessor(df, target)
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def fit(df: pd.DataFrame, target: str, model="logreg"):
    X = df.drop(columns=[target])
    y_raw = df[target].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    pipe = make_pipeline(df, target, model=model)
    pipe.fit(X, y)
    return pipe, le

def predict(pipe, le, X: pd.DataFrame):
    y = pipe.predict(X)
    labels = le.inverse_transform(y)
    prob = None
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        prob = pipe.predict_proba(X).max(axis=1)
    return labels, prob

def save(pipe, le, dirpath="artifacts"):
    import os
    os.makedirs(dirpath, exist_ok=True)
    joblib.dump(pipe, f"{dirpath}/model.joblib")
    joblib.dump(le, f"{dirpath}/label_encoder.joblib")

def load(dirpath="artifacts"):
    pipe = joblib.load(f"{dirpath}/model.joblib")
    le = joblib.load(f"{dirpath}/label_encoder.joblib")
    return pipe, le