# Classifier Web App (Tabular, scikit-learn)

A GitHub-ready project to train/evaluate a tabular classifier (categorical/numeric features) and serve predictions via **Streamlit**.

## Quickstart
```bash
pip install -r requirements.txt
# Train + save artifacts
python main.py train --data data/sample.csv --target type --save-artifacts
# Evaluate
python main.py eval --data data/sample.csv --target type --save-reports --save-images --no-show
# Predict (CSV with same columns except target)
python main.py predict --data data/sample.csv --target type --head 5
```

## Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```
Or push to GitHub and deploy on **Streamlit Community Cloud**:
- Main file: `app/streamlit_app.py`

## Structure
```
classifier_webapp/
├── app/streamlit_app.py        # Streamlit UI
├── src/
│   ├── data_loader.py          # read CSV
│   ├── preprocess.py           # infer schema + ColumnTransformer
│   ├── models.py               # train/save/load/predict
│   ├── evaluation.py           # metrics & confusion matrix
│   └── visualization.py        # plots
├── data/sample.csv             # sample dataset
├── artifacts/                  # saved model & preprocessor
├── reports/                    # metrics/reports
├── tests/test_basic.py         # minimal pytest
├── main.py                     # CLI
├── requirements.txt
├── LICENSE
└── .github/workflows/ci.yml
```

## Deploy as a Web App (from GitHub)
### Streamlit Community Cloud (free)
https://classifierwebapp-hbpnmrmin5ynjjj9izwb7e.streamlit.app/
