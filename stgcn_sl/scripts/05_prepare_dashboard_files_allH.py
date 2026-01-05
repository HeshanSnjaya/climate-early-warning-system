import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"
W_PATH = PROC_DIR / "W_knn10.csv"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# 1) neighbors
NEI_OUT = PRED_DIR / "neighbors_knn10.csv"
K = 10

W = pd.read_csv(W_PATH, header=None).to_numpy(dtype=float)
rows = []
for i in range(W.shape[0]):
    w = W[i].copy()
    w[i] = -1.0
    nn = np.argsort(w)[::-1][:K]
    for rank, j in enumerate(nn, start=1):
        rows.append({"station_idx": int(i), "neighbor_idx": int(j), "weight": float(W[i, j]), "rank": int(rank)})

pd.DataFrame(rows).to_csv(NEI_OUT, index=False)
print("Saved:", NEI_OUT)

# 2) convert predictions H=1..7
for H in range(1, 8):
    pred_in = PRED_DIR / f"predictions_P30_H{H}.csv"
    pred_out = PRED_DIR / f"predictions_P30_H{H}_mm.csv"

    pred = pd.read_csv(pred_in)
    pred["y_true_mm"] = np.expm1(pred["y_true"].astype(float))
    pred["y_pred_mm"] = np.expm1(pred["y_pred"].astype(float))
    pred["abs_error_mm"] = (pred["y_true_mm"] - pred["y_pred_mm"]).abs()
    pred["horizon_days"] = int(H)

    pred.to_csv(pred_out, index=False)
    print("Saved:", pred_out)
