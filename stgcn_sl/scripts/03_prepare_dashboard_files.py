import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PRED_IN = PROJECT_ROOT / "outputs" / "predictions" / "predictions_P30_H17.csv"
PRED_OUT = PROJECT_ROOT / "outputs" / "predictions" / "predictions_P30_H17_mm.csv"
W_PATH = PROC_DIR / "W_knn10.csv"
NEI_OUT = PROJECT_ROOT / "outputs" / "predictions" / "neighbors_knn10.csv"

K = 10

# 1) Convert predictions from log1p space to mm
pred = pd.read_csv(PRED_IN)
pred["y_true_mm"] = np.expm1(pred["y_true"].astype(float))
pred["y_pred_mm"] = np.expm1(pred["y_pred"].astype(float))
pred["abs_error_mm"] = (pred["y_true_mm"] - pred["y_pred_mm"]).abs()
pred.to_csv(PRED_OUT, index=False)
print("Saved:", PRED_OUT)

# 2) Precompute top-K neighbors from adjacency W
W = pd.read_csv(W_PATH, header=None).to_numpy(dtype=float)
rows = []
for i in range(W.shape[0]):
    # exclude self
    w = W[i].copy()
    w[i] = -1.0
    nn = np.argsort(w)[::-1][:K]
    for rank, j in enumerate(nn, start=1):
        rows.append({
            "station_idx": i,
            "neighbor_idx": int(j),
            "weight": float(W[i, j]),
            "rank": rank
        })

nei = pd.DataFrame(rows)
nei.to_csv(NEI_OUT, index=False)
print("Saved:", NEI_OUT)
