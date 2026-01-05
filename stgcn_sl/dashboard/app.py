import json
from pathlib import Path

import pandas as pd
import geopandas as gpd
import streamlit as st

st.set_page_config(page_title="Agentic Climate EWS – District", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIST_GJ = PROJECT_ROOT / "data" / "geo" / "districts.geojson"
DIST_PRED_DIR = PROJECT_ROOT / "outputs" / "district"
ALERTS_PATH = PROJECT_ROOT / "outputs" / "district_alerts" / "district_alerts.csv"
FB_PATH = PROJECT_ROOT / "outputs" / "feedback" / "district_feedback.csv"
POLICY_PATH = PROJECT_ROOT / "outputs" / "policy" / "policy.json"

# --- Station-level inputs (ADDED) ---
ST_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "station_to_district.csv"
STATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "stations.csv"
ST_PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"

st.title("Agentic Climate Early Warning System – Sri Lanka (District, 1–7 days)")
st.caption("Forecast → calibrated risk → autonomous alerts → explainability → feedback loop (proposal-aligned).")


@st.cache_data
def load_district_centroids():
    gdf = gpd.read_file(DIST_GJ).to_crs("EPSG:4326")
    gdf["centroid"] = gdf.geometry.centroid
    gdf["latitude"] = gdf["centroid"].y
    gdf["longitude"] = gdf["centroid"].x
    return pd.DataFrame({"district": gdf["name"], "latitude": gdf["latitude"], "longitude": gdf["longitude"]})


@st.cache_data
def load_alerts():
    if not ALERTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ALERTS_PATH)
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    df["origin_date"] = pd.to_datetime(df["origin_date"]).dt.date
    return df


@st.cache_data
def load_preds(H):
    p = DIST_PRED_DIR / f"district_predictions_H{H}.csv"
    df = pd.read_csv(p)
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    df["origin_date"] = pd.to_datetime(df["origin_date"]).dt.date
    return df


def load_policy():
    if POLICY_PATH.exists():
        return json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    return {"thresholds": {"Flood": 0.60, "Drought": 0.60}}


def save_feedback(df):
    FB_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FB_PATH, index=False)


# --- Station-level loaders (ADDED) ---
@st.cache_data
def load_station_map():
    return pd.read_csv(ST_MAP_PATH).dropna(subset=["district"])


@st.cache_data
def load_stations():
    return pd.read_csv(STATIONS_PATH)


@st.cache_data
def load_station_preds(H):
    p = ST_PRED_DIR / f"predictions_P30_H{H}_mm.csv"
    df = pd.read_csv(p)
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    df["origin_date"] = pd.to_datetime(df["origin_date"]).dt.date
    return df


centroids = load_district_centroids()
alerts = load_alerts()
policy = load_policy()

# Sidebar controls
st.sidebar.header("Controls")
H = st.sidebar.selectbox("Horizon (days)", list(range(1, 8)), index=0)

pred = load_preds(H)
dates = sorted(pred["target_date"].unique())
sel_date = st.sidebar.selectbox("Target date", dates, index=len(dates) - 1)

hazard_filter = st.sidebar.multiselect("Hazard filter", ["Flood", "Drought"], default=["Flood", "Drought"])
districts = sorted(pred["district"].unique())
sel_district = st.sidebar.selectbox("District", ["(All)"] + districts, index=0)

st.sidebar.subheader("Policy thresholds")
st.sidebar.write(policy.get("thresholds", {}))

# Merge centroids into predictions/alerts for mapping
pred_day = pred[pred["target_date"] == sel_date].merge(centroids, on="district", how="left")
alerts_day = pd.DataFrame()
if not alerts.empty:
    alerts_day = alerts[(alerts["target_date"] == sel_date) & (alerts["horizon_days"] == H)].copy()
    alerts_day = alerts_day[alerts_day["hazard"].isin(hazard_filter)]
    alerts_day = alerts_day.merge(centroids, on="district", how="left")

if sel_district != "(All)":
    pred_day = pred_day[pred_day["district"] == sel_district]
    if not alerts_day.empty:
        alerts_day = alerts_day[alerts_day["district"] == sel_district]

# Layout
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "District details", "Alerts & explanations", "Feedback (HITL)"])

with tab1:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.subheader("District centroids map")
        if not alerts_day.empty:
            st.map(alerts_day[["latitude", "longitude"]], latitude="latitude", longitude="longitude", height=500)
        else:
            st.map(pred_day[["latitude", "longitude"]], latitude="latitude", longitude="longitude", height=500)

    with c2:
        st.subheader("Forecast summary (mm)")
        st.dataframe(
            pred_day[["district", "y_pred_mm", "y_true_mm", "abs_error_mm", "n_stations"]]
            .sort_values("y_pred_mm", ascending=False)
            .head(25),
            width="stretch"
        )

        st.subheader("Alerts count")
        if alerts_day.empty:
            st.info("No alerts for selected date/horizon with current policy.")
        else:
            st.dataframe(alerts_day.groupby("hazard").size().reset_index(name="count"), width="stretch")

with tab2:
    st.subheader("District time series (forecast vs actual)")
    if sel_district == "(All)":
        st.info("Select a district from the sidebar to view time series.")
    else:
        series = pred[pred["district"] == sel_district].sort_values("target_date").set_index("target_date")
        st.line_chart(series[["y_pred_mm", "y_true_mm"]], width="stretch")
        st.dataframe(series.reset_index().tail(20), width="stretch")

        # ---- Station-wise view (FIXED robustly for _x/_y columns) ----
        st.divider()
        st.subheader("Station-wise view (selected district/date/H)")

        try:
            st_map = load_station_map()
            stations = load_stations()
            st_pred = load_station_preds(H)

            st_day = st_pred[st_pred["target_date"] == sel_date].merge(
                st_map[["station_idx", "district"]],
                on="station_idx",
                how="inner",
            )

            # Merge stations; if lat/lon already exists in st_day, pandas may create _x/_y columns [web:356]
            st_day = st_day[st_day["district"] == sel_district].merge(
                stations,
                on="station_idx",
                how="left",
            )

            # Normalize coordinate columns:
            # Prefer plain 'latitude/longitude', else take *_y (from stations merge), else *_x.
            if "latitude" not in st_day.columns:
                if "latitude_y" in st_day.columns:
                    st_day["latitude"] = st_day["latitude_y"]
                elif "latitude_x" in st_day.columns:
                    st_day["latitude"] = st_day["latitude_x"]

            if "longitude" not in st_day.columns:
                if "longitude_y" in st_day.columns:
                    st_day["longitude"] = st_day["longitude_y"]
                elif "longitude_x" in st_day.columns:
                    st_day["longitude"] = st_day["longitude_x"]

            cA, cB = st.columns([1, 1])

            with cA:
                st.write("Stations map")
                if ("latitude" in st_day.columns) and ("longitude" in st_day.columns):
                    coords = st_day.dropna(subset=["latitude", "longitude"])
                    if not coords.empty:
                        st.map(coords[["latitude", "longitude"]], latitude="latitude", longitude="longitude", height=350)
                    else:
                        st.info("Stations found, but coordinates are missing (lat/lon are NaN).")
                else:
                    st.error("Latitude/longitude not found after merges (check stations.csv headers).")

            with cB:
                st.write("Station table (sorted by error)")
                if not st_day.empty:
                    st.dataframe(
                        st_day[["station_idx", "y_pred_mm", "y_true_mm", "abs_error_mm"]]
                        .sort_values("abs_error_mm", ascending=False),
                        width="stretch",
                    )
                else:
                    st.info("No station rows found for this district/date (or unmapped stations).")

        except Exception as e:
            st.error(f"Station-wise view error: {e}")

with tab3:
    st.subheader("Autonomous alerts (agentic layer)")
    if alerts_day.empty:
        st.info("No alerts for selected date/horizon.")
    else:
        st.dataframe(
            alerts_day[["district", "hazard", "p_Flood", "p_Drought", "p_Normal", "y_pred_mm", "roll3_mm", "roll7_mm", "roll30_mm", "n_stations", "explanation_top_features"]]
            .sort_values(["hazard", "p_Flood"], ascending=False),
            width="stretch"
        )

        st.subheader("Explanation (SHAP top features)")
        pick = st.selectbox("Pick district alert", sorted(alerts_day["district"].unique()))
        row = alerts_day[alerts_day["district"] == pick].iloc[0]
        st.write(f"District: {pick} | Hazard: {row['hazard']} | H={H} | Date={sel_date}")

        top = json.loads(row["explanation_top_features"])
        st.dataframe(pd.DataFrame(top), width="stretch")

        drivers = ", ".join([f"{d['feature']} ({d['shap_value']:+.3f})" for d in top[:3]])
        st.write("Explanation summary:", drivers)

with tab4:
    st.subheader("Human-in-the-loop feedback")
    if not FB_PATH.exists():
        st.warning("district_feedback.csv not found. Run scripts/17_feedback_template_district.py")
    else:
        fb = pd.read_csv(FB_PATH)

        # filter feedback to current horizon/date (optional)
        fb["target_date"] = pd.to_datetime(fb["target_date"]).dt.date
        fb_view = fb[(fb["horizon_days"] == H) & (fb["target_date"] == sel_date)].copy()
        if sel_district != "(All)":
            fb_view = fb_view[fb_view["district"] == sel_district]

        st.write("Edit feedback values: TP / FP / FN / ignored")
        edited = st.data_editor(
            fb_view,
            num_rows="dynamic",
            use_container_width=True
        )

        if st.button("Save feedback"):
            # merge edited rows back into full feedback file
            key_cols = ["origin_date", "target_date", "horizon_days", "district", "hazard"]
            fb_full = fb.copy()
            edited_key = edited[key_cols].astype(str)

            fb_full_key = fb_full[key_cols].astype(str)
            fb_full["_k"] = fb_full_key.agg("|".join, axis=1)
            edited["_k"] = edited_key.agg("|".join, axis=1)

            fb_full = fb_full.set_index("_k")
            for _, r in edited.iterrows():
                fb_full.loc[r["_k"], "feedback"] = r.get("feedback", "")
                fb_full.loc[r["_k"], "comment"] = r.get("comment", "")

            fb_full = fb_full.reset_index(drop=True)
            save_feedback(fb_full)
            st.success("Feedback saved.")

        st.subheader("Update policy from feedback")
        st.write("Run: python scripts/19_update_policy_from_feedback.py")
        st.write("Then regenerate alerts: python scripts/16_district_alerts_with_shap.py")
