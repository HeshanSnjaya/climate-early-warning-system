import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth (km)."""
    R = 6371.0
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2) ** 2))
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def pairwise_distance_matrix(lat, lon):
    """
    Build full NxN distance matrix in km.
    For N~429 this is feasible.
    """
    n = len(lat)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lat[i], lon[i], lat[j], lon[j])
            D[i, j] = d
            D[j, i] = d
    return D


def build_knn_weight_matrix(D, k=10, sigma=None, self_loop=True, sym_mode="max"):
    """
    Build adjacency W from distance matrix D using KNN + Gaussian kernel.

    W_ij = exp(-(d_ij^2)/sigma^2) if j in knn(i), else 0

    sym_mode:
      - "max": W = max(W, W.T)
      - "mean": W = (W + W.T)/2
    """
    n = D.shape[0]
    W = np.zeros((n, n), dtype=np.float32)

    if sigma is None:
        nonzero = D[D > 0]
        sigma = float(np.median(nonzero)) if nonzero.size else 1.0
        if sigma == 0:
            sigma = 1.0

    for i in range(n):
        nn = np.argsort(D[i])[1:k + 1]  # skip self at index 0
        for j in nn:
            d = float(D[i, j])
            W[i, j] = math.exp(-(d * d) / (sigma * sigma))

    if sym_mode == "mean":
        W = (W + W.T) / 2.0
    else:
        W = np.maximum(W, W.T)

    if self_loop:
        np.fill_diagonal(W, 1.0)

    return W, sigma


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to Sri_Lanka_Climate_Data.csv (optional).")
    parser.add_argument("--k", type=int, default=10, help="K for KNN graph.")
    parser.add_argument("--log1p", action="store_true",
                        help="Apply log1p transform to precipitation_sum.")
    args = parser.parse_args()

    # Resolve project structure based on script location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        raw_path = Path(args.csv).resolve()
    else:
        raw_path = project_root / "data" / "raw" / "Sri_Lanka_Climate_Data.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"CSV not found: {raw_path}")

    # Load
    df = pd.read_csv(raw_path)

    # Basic checks
    required_cols = ["date", "latitude", "longitude", "precipitation_sum"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}. Found: {list(df.columns)}")

    # Normalize date
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date

    # Build station index
    stations = (df[["latitude", "longitude"]]
                .drop_duplicates()
                .sort_values(["latitude", "longitude"])
                .reset_index(drop=True))
    stations["station_idx"] = np.arange(len(stations), dtype=int)

    df = df.merge(stations, on=["latitude", "longitude"], how="left")

    N = int(stations.shape[0])
    date_min = df["date"].min()
    date_max = df["date"].max()

    # Save stations.csv
    stations_out = stations[["station_idx", "latitude", "longitude"]]
    stations_out.to_csv(out_dir / "stations.csv", index=False)

    # Create V matrix (T x N)
    all_dates = pd.date_range(pd.to_datetime(date_min), pd.to_datetime(date_max), freq="D").date
    T = len(all_dates)

    V = (df.pivot(index="date", columns="station_idx", values="precipitation_sum")
           .reindex(all_dates))

    if V.isna().any().any():
        V = V.interpolate(axis=0).bfill().ffill()

    V = V.astype(np.float32)

    if args.log1p:
        V = np.log1p(V)

    V_df = pd.DataFrame(V, index=all_dates)
    v_name = "V_precip_log1p.csv" if args.log1p else "V_precip.csv"
    V_df.to_csv(out_dir / v_name, index=True)

    # Build W adjacency (N x N)
    lat = stations["latitude"].to_numpy(dtype=float)
    lon = stations["longitude"].to_numpy(dtype=float)

    D = pairwise_distance_matrix(lat, lon)
    W, sigma = build_knn_weight_matrix(D, k=args.k, sigma=None, self_loop=True, sym_mode="max")

    w_name = f"W_knn{args.k}.csv"
    pd.DataFrame(W).to_csv(out_dir / w_name, index=False, header=False)

    # Save metadata
    meta = {
        "csv_path": str(raw_path),
        "N_stations": N,
        "T_days": T,
        "date_min": str(date_min),
        "date_max": str(date_max),
        "V_file": v_name,
        "W_file": w_name,
        "stations_file": "stations.csv",
        "knn_k": args.k,
        "sigma_km": float(sigma),
        "transform": "log1p(precipitation_sum)" if args.log1p else "none",
        "note": "V rows are dates; columns are station_idx (0..N-1)."
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
