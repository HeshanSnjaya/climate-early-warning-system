import json
import numpy as np
import pandas as pd
from pathlib import Path
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "outputs" / "district"
MODEL_DIR = PROJECT_ROOT / "outputs" / "district_models"
OUT_DIR = PROJECT_ROOT / "outputs" / "district_alerts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
POLICY_PATH = PROJECT_ROOT / "outputs" / "policy" / "policy.json"
policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
THRESH = policy["thresholds"]

FEATURES = json.loads((MODEL_DIR / "features.json").read_text(encoding="utf-8"))

THRESH = {"Flood": 0.60, "Drought": 0.60}

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

rows = []

for H in range(1, 8):
    df = pd.read_csv(IN_DIR / f"district_predictions_H{H}.csv")
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    df["origin_date"] = pd.to_datetime(df["origin_date"]).dt.date

    labels = pd.read_csv(IN_DIR / "district_risk_labels.csv")
    labels["target_date"] = pd.to_datetime(labels["target_date"]).dt.date
    df = df.merge(labels[["district", "target_date", "roll3_mm", "roll7_mm", "roll30_mm"]], on=["district", "target_date"], how="left")
    df = df.dropna(subset=FEATURES)

    coef = np.load(MODEL_DIR / f"coef_H{H}.npy")
    intercept = np.load(MODEL_DIR / f"intercept_H{H}.npy")
    classes = json.loads((MODEL_DIR / f"classes_H{H}.json").read_text(encoding="utf-8"))
    T = json.loads((MODEL_DIR / f"temperature_H{H}.json").read_text(encoding="utf-8"))["T"]

    X = df[FEATURES].astype(float)
    logits = X.to_numpy() @ coef.T + intercept
    probs = softmax(logits / T)

    for i, c in enumerate(classes):
        df[f"p_{c}"] = probs[:, i]

    def predict_proba(X_in):
        X_in = np.asarray(X_in, dtype=float)
        lg = X_in @ coef.T + intercept
        return softmax(lg / T)

    explainer = shap.Explainer(predict_proba, X, feature_names=FEATURES)
    sv = explainer(X)

    for i in range(len(df)):
        r = df.iloc[i]
        p_f = float(r.get("p_Flood", 0.0))
        p_d = float(r.get("p_Drought", 0.0))

        hazard = None
        if "Flood" in classes and p_f >= THRESH["Flood"]:
            hazard = "Flood"
            cidx = classes.index("Flood")
        elif "Drought" in classes and p_d >= THRESH["Drought"]:
            hazard = "Drought"
            cidx = classes.index("Drought")
        else:
            continue

        vals = sv.values[i, :, cidx]
        top = sorted(zip(FEATURES, vals), key=lambda t: abs(t[1]), reverse=True)[:5]
        expl = [{"feature": f, "shap_value": float(v)} for f, v in top]

        rows.append({
            "origin_date": str(r["origin_date"]),
            "target_date": str(r["target_date"]),
            "horizon_days": int(H),
            "district": r["district"],
            "hazard": hazard,
            "p_Flood": p_f,
            "p_Drought": p_d,
            "p_Normal": float(r.get("p_Normal", 0.0)),
            "threshold_flood": float(THRESH["Flood"]),
            "threshold_drought": float(THRESH["Drought"]),
            "temperature": float(T),
            "explanation_top_features": json.dumps(expl),
            "y_pred_mm": float(r["y_pred_mm"]),
            "roll3_mm": float(r["roll3_mm"]),
            "roll7_mm": float(r["roll7_mm"]),
            "roll30_mm": float(r["roll30_mm"]),
            "n_stations": int(r["n_stations"]),
        })

out = pd.DataFrame(rows)
out_path = OUT_DIR / "district_alerts.csv"
out.to_csv(out_path, index=False)
print("Saved:", out_path, "alerts=", len(out))
