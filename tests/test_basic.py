import pandas as pd
from src.models import fit, predict

def test_fit_predict():
    df = pd.DataFrame({
        'a':['x','y','x','y','x','y'],
        'b':[1,2,3,4,5,6],
        'label':['L','L','L','R','R','R']
    })
    pipe, le = fit(df.rename(columns={'label':'target'}).rename(columns={'target':'label'}).rename(columns={'label':'target'}), target='target')  # ensure target named correctly
    X = df.drop(columns=['label'])
    labels, prob = predict(pipe, le, X.head(2))
    assert len(labels) == 2