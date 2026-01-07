import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Load adjacency matrix
W = pd.read_csv("stgcn_sl/data/processed/W_knn10.csv", index_col=0)

# Degree = number of neighbors per station
degrees = (W > 0).sum(axis=1)

# ===== FIG 3.5: Degree Distribution =====
plt.figure(figsize=(8,5))
sns.histplot(degrees, bins=20, kde=False, color='teal')
plt.title("Graph Connectivity: Station Degree Distribution (KNN=10)", fontsize=14, fontweight='bold')
plt.xlabel("Number of Neighbors", fontsize=12)
plt.ylabel("Number of Stations", fontsize=12)
plt.axvline(degrees.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {degrees.mean():.1f}')
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_05_graph_degree_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved fig_03_05_graph_degree_distribution.png")

print("\nGraph connectivity figure complete!")
