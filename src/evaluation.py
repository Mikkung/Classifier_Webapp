import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .models import fit

def train_eval(df: pd.DataFrame, target: str, test_size=0.2, random_state=42, model="logreg"):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target])
    pipe, le = fit(train, target=target, model=model)
    y_true = le.transform(test[target].values)
    X_test = test.drop(columns=[target])
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    return pipe, le, metrics, cm, test