import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# ===== UPDATED FILE PATHS =====
V_PATH = "stgcn_sl/data/processed/V_precip_log1p.csv"
STATIONS_PATH = "stgcn_sl/data/processed/stations.csv"

# ===== LOAD DATA =====
V = pd.read_csv(V_PATH)
stations = pd.read_csv(STATIONS_PATH)

# Detect date column
date_col = None
for c in V.columns:
    if "date" in c.lower() or "time" in c.lower():
        date_col = c
        break

if date_col:
    t = pd.to_datetime(V[date_col])
    X = V.drop(columns=[date_col])
else:
    t = pd.RangeIndex(len(V))
    X = V

# ===== FIG 3.2: Station Locations =====
plt.figure(figsize=(8,8))
plt.scatter(stations["longitude"], stations["latitude"], s=15, alpha=0.7, 
            c='steelblue', edgecolors='black', linewidth=0.3)
plt.title("Station Locations (Sri Lanka)", fontsize=14, fontweight='bold')
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_02_station_locations.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_03_02_station_locations.png")

# ===== FIG 3.3: Precipitation Distributions =====
# Convert to numeric and flatten, handling any non-numeric columns
vals_log = pd.to_numeric(X.stack(), errors='coerce').values
vals_log = vals_log[np.isfinite(vals_log)]  # Remove NaN/inf
vals_mm = np.expm1(vals_log)

# Log1p distribution
plt.figure(figsize=(8,5))
sns.histplot(vals_log, bins=80, kde=True, color='steelblue')
plt.title("Distribution of log1p(precipitation_sum)", fontsize=14, fontweight='bold')
plt.xlabel("log1p(mm)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_03a_hist_log1p_precip.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_03_03a_hist_log1p_precip.png")

# MM distribution
plt.figure(figsize=(8,5))
sns.histplot(vals_mm, bins=80, kde=True, color='darkorange')
plt.title("Distribution of precipitation_sum (mm)", fontsize=14, fontweight='bold')
plt.xlabel("mm", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xlim(0, np.percentile(vals_mm, 99))
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_03b_hist_mm_precip.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_03_03b_hist_mm_precip.png")

# Sample time series - pick first numeric column
numeric_cols = X.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    sample_col = numeric_cols[0]
    sample_data = X[sample_col].values
    
    plt.figure(figsize=(12,4))
    plt.plot(t, np.expm1(sample_data), linewidth=0.8, color='steelblue')
    plt.title(f"Sample Station Daily Precipitation (mm) - Station {sample_col}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Precipitation (mm)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_03_03c_sample_timeseries.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_03_03c_sample_timeseries.png")
else:
    print("No numeric columns found for time series plot")

print("\nDataset figures complete!")
