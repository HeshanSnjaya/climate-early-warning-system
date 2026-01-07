import os, re, ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

LOG_PATH = "log_file.txt"

# Parse test metrics from paste.txt
text = open(LOG_PATH, "r", encoding="utf-8", errors="ignore").read()
re_test = re.compile(r"Test metrics:\s+(\{.*?\})")

rows = []
for match in re_test.finditer(text):
    try:
        d = ast.literal_eval(match.group(1))
        rows.append({
            "horizon": d.get("H"),
            "MAE": d.get("test_MAE"),
            "RMSE": d.get("test_RMSE"),
        })
    except:
        pass

df = pd.DataFrame(rows).dropna().sort_values("horizon")

# ===== FIG 3.4a: Test MAE by Horizon =====
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="horizon", y="MAE", color="steelblue")
plt.title("Test MAE by Horizon", fontsize=14, fontweight='bold')
plt.xlabel("Horizon (days)", fontsize=12)
plt.ylabel("MAE (log1p space)", fontsize=12)
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.3f', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_04a_test_mae_by_horizon.png", dpi=300, bbox_inches='tight')
plt.close()
print(" Saved fig_03_04a_test_mae_by_horizon.png")

# ===== FIG 3.4b: Test RMSE by Horizon =====
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="horizon", y="RMSE", color="darkorange")
plt.title("Test RMSE by Horizon", fontsize=14, fontweight='bold')
plt.xlabel("Horizon (days)", fontsize=12)
plt.ylabel("RMSE (log1p space)", fontsize=12)
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.3f', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_03_04b_test_rmse_by_horizon.png", dpi=300, bbox_inches='tight')
plt.close()
print(" Saved fig_03_04b_test_rmse_by_horizon.png")

print("\nTest metrics bar charts complete!")
