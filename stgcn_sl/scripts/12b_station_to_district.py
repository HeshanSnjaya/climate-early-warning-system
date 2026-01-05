from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "stations.csv"
GJ = PROJECT_ROOT / "data" / "geo" / "districts.geojson"
OUT = PROJECT_ROOT / "data" / "processed" / "station_to_district.csv"

DISTRICT_FIELD = "name" 

stations = pd.read_csv(STATIONS_PATH)
districts = gpd.read_file(GJ)

# force WGS84
districts = districts.to_crs("EPSG:4326") if districts.crs else districts.set_crs("EPSG:4326")

gdf_st = gpd.GeoDataFrame(
    stations.copy(),
    geometry=[Point(float(lon), float(lat)) for lat, lon in zip(stations["latitude"], stations["longitude"])],
    crs="EPSG:4326",
)

joined = gpd.sjoin(gdf_st, districts[[DISTRICT_FIELD, "geometry"]], how="left", predicate="within")

out = joined[["station_idx", "latitude", "longitude", DISTRICT_FIELD]].rename(columns={DISTRICT_FIELD: "district"})
missing = int(out["district"].isna().sum())

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Stations:", len(out), "Missing matches:", missing)
