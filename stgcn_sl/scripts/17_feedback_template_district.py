import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALERTS = PROJECT_ROOT / "outputs" / "district_alerts" / "district_alerts.csv"
OUT = PROJECT_ROOT / "outputs" / "feedback" / "district_feedback.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ALERTS)
df["feedback"] = ""   # TP / FP / FN / ignored
df["comment"] = ""
df.to_csv(OUT, index=False)
print("Saved:", OUT)
