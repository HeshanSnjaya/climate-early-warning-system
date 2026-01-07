import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# ===== UPDATED FILE PATH =====
RISK_PATH = "stgcn_sl/outputs/district/district_risk_labels.csv"

# Load district risk labels
risk = pd.read_csv(RISK_PATH)

# Correct column names (with underscores)
roll3_col = "roll3_mm"
roll7_col = "roll7_mm"
roll30_col = "roll30_mm"

# Thresholds from your report
FLOOD_THR = 59.55199996359
DROUGHT_THR = 24.3713889748

# ===== FIG 3.10a: roll3_mm + Flood Threshold =====
plt.figure(figsize=(8,5))
sns.histplot(risk[roll3_col], bins=60, kde=True, color='steelblue')
plt.axvline(FLOOD_THR, color='red', linestyle='--', linewidth=2.5, 
            label=f'Flood Threshold (p95) = {FLOOD_THR:.2f} mm')
plt.title("roll3_mm Distribution with Flood Threshold", fontsize=14, fontweight='bold')
plt.xlabel("roll3_mm (3-day rolling sum, mm)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_10a_roll3_threshold.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_03_10a_roll3_threshold.png")

# ===== FIG 3.10b: roll30_mm + Drought Threshold =====
plt.figure(figsize=(8,5))
sns.histplot(risk[roll30_col], bins=60, kde=True, color='darkorange')
plt.axvline(DROUGHT_THR, color='red', linestyle='--', linewidth=2.5, 
            label=f'Drought Threshold (p10) = {DROUGHT_THR:.2f} mm')
plt.title("roll30_mm Distribution with Drought Threshold", fontsize=14, fontweight='bold')
plt.xlabel("roll30_mm (30-day rolling sum, mm)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_10b_roll30_threshold.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_03_10b_roll30_threshold.png")

print("\nThreshold histograms complete!")
