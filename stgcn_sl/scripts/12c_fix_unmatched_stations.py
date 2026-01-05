from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "stations.csv"
GJ = PROJECT_ROOT / "data" / "geo" / "districts.geojson"
OUT = PROJECT_ROOT / "data" / "processed" / "station_to_district_fixed.csv"

DISTRICT_FIELD = "name"

stations = pd.read_csv(STATIONS_PATH)
districts = gpd.read_file(GJ)
districts = districts.to_crs("EPSG:4326") if districts.crs else districts.set_crs("EPSG:4326")

gdf_st = gpd.GeoDataFrame(
    stations.copy(),
    geometry=[Point(float(lon), float(lat)) for lat, lon in zip(stations["latitude"], stations["longitude"])],
    crs="EPSG:4326",
)

# 1) within join
j1 = gpd.sjoin(gdf_st, districts[[DISTRICT_FIELD, "geometry"]], how="left", predicate="within")

missing_idx = j1[j1[DISTRICT_FIELD].isna()].index
missing = j1.loc[missing_idx].copy()

# 2) fallback join using intersects
if not missing.empty:
    missing = missing.drop(columns=["index_right", "index_left"], errors="ignore")
    j2 = gpd.sjoin(
        missing,
        districts[[DISTRICT_FIELD, "geometry"]],
        how="left",
        predicate="intersects",
    )
    # Fill back
    j1.loc[missing_idx, DISTRICT_FIELD] = j2[DISTRICT_FIELD].values

# export
out = j1.drop(columns=["index_right", "index_left"], errors="ignore")[
    ["station_idx", "latitude", "longitude", DISTRICT_FIELD]
].rename(columns={DISTRICT_FIELD: "district"})

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Missing after fix:", int(out["district"].isna().sum()))
