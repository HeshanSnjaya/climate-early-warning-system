from pathlib import Path
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GJ = PROJECT_ROOT / "data" / "geo" / "districts.geojson"

gdf = gpd.read_file(GJ)
print("CRS:", gdf.crs)
print("Columns:", list(gdf.columns))
print("Head:")
print(gdf.head(3))
