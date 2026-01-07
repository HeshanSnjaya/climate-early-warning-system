import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import Counter

sns.set(style="whitegrid")
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Load alerts
alerts = pd.read_csv("stgcn_sl/outputs/district_alerts/district_alerts.csv")
alerts["origin_date"] = pd.to_datetime(alerts["origin_date"])
alerts["target_date"] = pd.to_datetime(alerts["target_date"])

# ===== FIG 4.2: Alerts by Hazard Type =====
plt.figure(figsize=(7,5))
sns.countplot(data=alerts, x="hazard", palette="Set2")
plt.title("Alert Counts by Hazard Type", fontsize=14, fontweight='bold')
plt.xlabel("Hazard", fontsize=12)
plt.ylabel("Count", fontsize=12)
for container in plt.gca().containers:
    plt.gca().bar_label(container, fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_04_02_alert_counts_by_hazard.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_04_02_alert_counts_by_hazard.png")

# ===== FIG 4.3: Alerts by Horizon =====
plt.figure(figsize=(8,5))
sns.countplot(data=alerts, x="horizon_days", color="mediumpurple")
plt.title("Alert Counts by Horizon", fontsize=14, fontweight='bold')
plt.xlabel("Horizon (days)", fontsize=12)
plt.ylabel("Count", fontsize=12)
for container in plt.gca().containers:
    plt.gca().bar_label(container, fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_04_03_alert_counts_by_horizon.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_04_03_alert_counts_by_horizon.png")

# ===== FIG 4.4: Top SHAP Features =====
counter = Counter()
for s in alerts["explanation_top_features"].dropna():
    try:
        items = json.loads(s)
        for it in items:
            counter[it["feature"]] += 1
    except:
        pass

df_shap = pd.DataFrame(counter.items(), columns=["feature","count"]).sort_values("count", ascending=False).head(10)

plt.figure(figsize=(8,6))
sns.barplot(data=df_shap, y="feature", x="count", palette="viridis")
plt.title("Most Frequent Features in SHAP Explanations (Top 10)", fontsize=14, fontweight='bold')
plt.xlabel("Count Across Alerts", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_04_04_shap_top_feature_frequency.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_04_04_shap_top_feature_frequency.png")

# ===== FIG 4.5: Example District Timeline =====
example_district = alerts["district"].value_counts().index[0]
ad = alerts[alerts["district"] == example_district].sort_values("target_date")

plt.figure(figsize=(12,5))
plt.plot(ad["target_date"], ad["roll3_mm"], label="roll3_mm", linewidth=2, alpha=0.8)
plt.plot(ad["target_date"], ad["roll7_mm"], label="roll7_mm", linewidth=2, alpha=0.8)
plt.plot(ad["target_date"], ad["roll30_mm"], label="roll30_mm", linewidth=2, alpha=0.8)

# Flood alerts
flood_data = ad[ad["hazard"]=="Flood"]
if len(flood_data) > 0:
    plt.scatter(flood_data["target_date"], flood_data["roll3_mm"], 
                marker="^", s=100, color="red", label="Flood Alert", zorder=5)

# Drought alerts
drought_data = ad[ad["hazard"]=="Drought"]
if len(drought_data) > 0:
    plt.scatter(drought_data["target_date"], drought_data["roll30_mm"],
                marker="v", s=100, color="brown", label="Drought Alert", zorder=5)

plt.title(f"Rolling Features and Alerts — District: {example_district}", fontsize=14, fontweight='bold')
plt.xlabel("Target Date", fontsize=12)
plt.ylabel("Precipitation (mm)", fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig_04_05_example_district_timeline.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_04_05_example_district_timeline.png")

print("\nAlert analysis figures complete!")
