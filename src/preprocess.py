import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def infer_schema(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c != target]
    cat_cols = [c for c in features if df[c].dtype == 'object']
    num_cols = [c for c in features if df[c].dtype != 'object']
    return features, cat_cols, num_cols

def make_preprocessor(df: pd.DataFrame, target: str):
    _, cat_cols, num_cols = infer_schema(df, target)
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return pre