import argparse, os, pandas as pd
from src.data_loader import load_csv
from src.evaluation import train_eval
from src.models import save, load, predict, fit
from src.visualization import plot_confusion_matrix

def parse_args():
    ap = argparse.ArgumentParser(description="Tabular Classifier Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train"); t.add_argument("--data", default="data/sample.csv"); t.add_argument("--target", required=True); t.add_argument("--model", default="logreg", choices=["logreg","rf"]); t.add_argument("--save-artifacts", action="store_true")
    e = sub.add_parser("eval");  e.add_argument("--data", default="data/sample.csv"); e.add_argument("--target", required=True); e.add_argument("--model", default="logreg", choices=["logreg","rf"]); e.add_argument("--no-show", action="store_true"); e.add_argument("--save-images", action="store_true"); e.add_argument("--save-reports", action="store_true")
    p = sub.add_parser("predict"); p.add_argument("--data", required=True); p.add_argument("--target", required=True); p.add_argument("--head", type=int, default=5)

    return ap.parse_args()

def maybe(path):
    if not path: return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def cmd_train(a):
    df = load_csv(a.data)
    pipe, le, metrics, cm, test = train_eval(df, target=a.target, model=a.model)
    print("Metrics:", metrics)
    if a.save_artifacts:
        save(pipe, le, "artifacts"); print("Artifacts saved to artifacts/")
    img = maybe("reports/confusion_matrix.png")
    plot_confusion_matrix(cm, labels=list(le.classes_), show=False, save_path=img)

def cmd_eval(a):
    df = load_csv(a.data)
    pipe, le, metrics, cm, test = train_eval(df, target=a.target, model=a.model)
    print("Metrics:", metrics)
    img = maybe("reports/confusion_matrix.png") if a.save_images else None
    plot_confusion_matrix(cm, labels=list(le.classes_), show=(not a.no_show), save_path=img)
    if a.save_reports:
        os.makedirs("reports", exist_ok=True)
        pd.DataFrame([metrics]).to_csv("reports/metrics.csv", index=False)
        test.to_csv("reports/heldout_sample.csv", index=False)

def cmd_predict(a):
    df = load_csv(a.data)
    X = df.drop(columns=[a.target]).head(a.head)
    pipe, le = load("artifacts")
    labels, prob = predict(pipe, le, X)
    print(pd.DataFrame({"pred": labels, "confidence": (prob if prob is not None else [None]*len(labels))}))

def main():
    a = parse_args()
    if a.cmd == "train":   cmd_train(a)
    elif a.cmd == "eval":  cmd_eval(a)
    elif a.cmd == "predict": cmd_predict(a)

if __name__ == "__main__":
    main()