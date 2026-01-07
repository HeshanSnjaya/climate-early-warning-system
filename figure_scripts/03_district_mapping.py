import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Load station-to-district mapping
mapping = pd.read_csv("stgcn_sl/data/processed/station_to_district.csv")
counts = mapping.groupby("district").size().reset_index(name="n_stations")

# ===== FIG 3.7: Stations per District =====
plt.figure(figsize=(10,5))
sns.boxplot(data=counts, x="n_stations", color="lightcoral")
plt.title("Distribution of Number of Stations per District", fontsize=14, fontweight='bold')
plt.xlabel("Stations per District", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_07_boxplot_stations_per_district.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_03_07_boxplot_stations_per_district.png")

# Print summary
print("\n District Mapping Summary:")
print(f"Total districts: {len(counts)}")
print(f"Mean stations per district: {counts['n_stations'].mean():.1f}")
print(f"\nTop 5 districts (most stations):")
print(counts.nlargest(5, 'n_stations')[['district','n_stations']].to_string(index=False))
print(f"\nBottom 5 districts (fewest stations):")
print(counts.nsmallest(5, 'n_stations')[['district','n_stations']].to_string(index=False))

print("\n District mapping figure complete!")
