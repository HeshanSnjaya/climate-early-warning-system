import os, re, ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

LOG_PATH = "log_file.txt"

# Parse log file
text = open(LOG_PATH, "r", encoding="utf-8", errors="ignore").read().splitlines()

rows = []
current_H = None
current_P = 30  # Default

re_running = re.compile(r"--P\s+(\d+)\s+--H\s+(\d+)")
re_epoch = re.compile(r"Epoch\s+(\d+)\s+\|\s+train_L1=([0-9.]+)\s+\|\s+val_MAE=([0-9.]+)\s+\|\s+val_RMSE=([0-9.]+)")
re_test = re.compile(r"Test metrics:\s+(\{.*\})")

for line in text:
    # Detect H from "Running: ... --H X"
    m_run = re_running.search(line)
    if m_run:
        current_P = int(m_run.group(1))
        current_H = int(m_run.group(2))
        continue
    
    # Extract epoch metrics
    m_epoch = re_epoch.search(line)
    if m_epoch and current_H is not None:
        rows.append({
            "P": current_P,
            "H": current_H,
            "epoch": int(m_epoch.group(1)),
            "train_L1": float(m_epoch.group(2)),
            "val_MAE": float(m_epoch.group(3)),
            "val_RMSE": float(m_epoch.group(4)),
        })
        continue
    
    # Extract test metrics
    m_test = re_test.search(line)
    if m_test:
        try:
            d = ast.literal_eval(m_test.group(1))
            rows.append({
                "P": d.get("P", 30),
                "H": d.get("H"),
                "epoch": "TEST",
                "train_L1": None,
                "val_MAE": d.get("test_MAE"),
                "val_RMSE": d.get("test_RMSE"),
            })
        except:
            pass

df = pd.DataFrame(rows)
df.to_csv(f"{OUTDIR}/parsed_learning_curves.csv", index=False)
print(f"✓ Saved parsed data: {OUTDIR}/parsed_learning_curves.csv")

# ===== FIG 4.1: Learning Curves for Each Horizon =====
horizons = sorted([h for h in df["H"].dropna().unique() if h != "TEST"])

for H in horizons:
    dH = df[(df["H"] == H) & (df["epoch"] != "TEST")].copy()
    dH["epoch"] = dH["epoch"].astype(int)
    dH = dH.sort_values("epoch")
    
    # Combined plot: train_L1 + val_MAE + val_RMSE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Train L1
    axes[0].plot(dH["epoch"], dH["train_L1"], linewidth=2, color='steelblue', marker='o', markersize=4)
    axes[0].set_title(f"Training Loss (L1) — H={H}", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Train L1", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Right: Val MAE + RMSE
    axes[1].plot(dH["epoch"], dH["val_MAE"], linewidth=2, color='darkorange', marker='s', markersize=4, label="val_MAE")
    axes[1].plot(dH["epoch"], dH["val_RMSE"], linewidth=2, color='green', marker='^', markersize=4, label="val_RMSE")
    axes[1].set_title(f"Validation Metrics — H={H}", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Metric Value", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig_04_01_learning_curve_H{H}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved fig_04_01_learning_curve_H{H}.png")

print("\nLearning curves complete!")
