import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "outputs" / "district"
LABELS = pd.read_csv(IN_DIR / "district_risk_labels.csv")
LABELS["target_date"] = pd.to_datetime(LABELS["target_date"]).dt.date

OUT_DIR = PROJECT_ROOT / "outputs" / "district_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["y_pred_mm", "roll3_mm", "roll7_mm", "roll30_mm"]
(OUT_DIR / "features.json").write_text(json.dumps(FEATURES), encoding="utf-8")

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))

def fit_temperature(logits, y):
    Ts = np.linspace(0.5, 5.0, 91)
    bestT, best = 1.0, 1e18
    for T in Ts:
        p = softmax(logits / T)
        loss = nll(p, y)
        if loss < best:
            best = loss
            bestT = float(T)
    return bestT, float(best)

for H in range(1, 8):
    df = pd.read_csv(IN_DIR / f"district_predictions_H{H}.csv")
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date

    df = df.merge(LABELS, on=["district", "target_date"], how="left")
    df = df.dropna(subset=["label", "roll3_mm", "roll7_mm", "roll30_mm"])

    X = df[FEATURES].astype(float).to_numpy()
    y = df["label"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=400, multi_class="multinomial")
    clf.fit(X_tr, y_tr)

    y_hat = clf.predict(X_te)
    macro_f1 = float(f1_score(y_te, y_hat, average="macro"))

    classes = list(clf.classes_)
    y_idx = np.array([classes.index(v) for v in y_te], dtype=int)

    logits = X_te @ clf.coef_.T + clf.intercept_
    T, val_nll = fit_temperature(logits, y_idx)

    np.save(OUT_DIR / f"coef_H{H}.npy", clf.coef_)
    np.save(OUT_DIR / f"intercept_H{H}.npy", clf.intercept_)
    (OUT_DIR / f"classes_H{H}.json").write_text(json.dumps(classes), encoding="utf-8")
    (OUT_DIR / f"temperature_H{H}.json").write_text(json.dumps({"H": H, "T": T, "val_nll": val_nll}, indent=2), encoding="utf-8")
    (OUT_DIR / f"metrics_H{H}.json").write_text(json.dumps({"H": H, "macro_f1": macro_f1, "classes": classes}, indent=2), encoding="utf-8")

    print(f"H={H} macroF1={macro_f1:.4f} T={T:.3f} valNLL={val_nll:.4f}")
