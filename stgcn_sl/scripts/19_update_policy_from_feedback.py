import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FB = PROJECT_ROOT / "outputs" / "feedback" / "district_feedback.csv"
POLICY_PATH = PROJECT_ROOT / "outputs" / "policy" / "policy.json"

# Simple policy adjustment rules:
# - If too many false positives => increase threshold
# - If too many false negatives => decrease threshold
STEP = 0.05
MIN_T, MAX_T = 0.40, 0.90

df = pd.read_csv(FB)

policy = {"thresholds": {"Flood": 0.60, "Drought": 0.60}}
if POLICY_PATH.exists():
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

for hazard in ["Flood", "Drought"]:
    h = df[df["hazard"] == hazard].copy()
    if h.empty:
        continue

    fp = (h["feedback"].fillna("") == "FP").sum()
    fn = (h["feedback"].fillna("") == "FN").sum()
    tp = (h["feedback"].fillna("") == "TP").sum()

    # Avoid division by zero
    total = fp + fn + tp
    if total == 0:
        continue

    fp_rate = fp / total
    fn_rate = fn / total

    t = float(policy["thresholds"].get(hazard, 0.60))

    # Heuristic tuning
    if fp_rate > 0.20:
        t = min(MAX_T, t + STEP)
    elif fn_rate > 0.20:
        t = max(MIN_T, t - STEP)

    policy["thresholds"][hazard] = float(round(t, 3))

POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
POLICY_PATH.write_text(json.dumps(policy, indent=2), encoding="utf-8")
print("Updated policy:", policy)
print("Saved:", POLICY_PATH)
