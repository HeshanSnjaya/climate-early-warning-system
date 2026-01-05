from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "outputs" / "district"
OUT = IN_DIR / "district_risk_labels.csv"

df = pd.read_csv(IN_DIR / "district_predictions_H1.csv")
df["target_date"] = pd.to_datetime(df["target_date"])
df = df.sort_values(["district", "target_date"])

df["roll3_mm"] = df.groupby("district")["y_true_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
df["roll7_mm"] = df.groupby("district")["y_true_mm"].transform(lambda s: s.rolling(7, min_periods=1).sum())
df["roll30_mm"] = df.groupby("district")["y_true_mm"].transform(lambda s: s.rolling(30, min_periods=1).sum())

flood_thr = float(df["roll3_mm"].quantile(0.95))
drought_thr = float(df["roll30_mm"].quantile(0.10))

def lab(r):
    if r["roll3_mm"] >= flood_thr:
        return "Flood"
    if r["roll30_mm"] <= drought_thr:
        return "Drought"
    return "Normal"

df["label"] = df.apply(lab, axis=1)

out = df[["target_date", "district", "roll3_mm", "roll7_mm", "roll30_mm", "label"]].copy()
out["target_date"] = out["target_date"].dt.date
out.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Thresholds:", {"flood_roll3_p95": flood_thr, "drought_roll30_p10": drought_thr})
