from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = PROJECT_ROOT / "data" / "processed" / "station_to_district.csv"
PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUT_DIR = PROJECT_ROOT / "outputs" / "district"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGG = "mean" 
m = pd.read_csv(MAP_PATH).dropna(subset=["district"])

for H in range(1, 8):
    df = pd.read_csv(PRED_DIR / f"predictions_P30_H{H}_mm.csv")
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    df["origin_date"] = pd.to_datetime(df["origin_date"]).dt.date

    df = df.merge(m[["station_idx", "district"]], on="station_idx", how="inner")

    g = df.groupby(["origin_date", "target_date", "district"], as_index=False).agg(
        y_pred_mm=("y_pred_mm", AGG),
        y_true_mm=("y_true_mm", AGG),
        abs_error_mm=("abs_error_mm", AGG),
        n_stations=("station_idx", "count"),
    )
    g["horizon_days"] = int(H)

    out_path = OUT_DIR / f"district_predictions_H{H}.csv"
    g.to_csv(out_path, index=False)
    print("Saved:", out_path, "rows=", len(g))
