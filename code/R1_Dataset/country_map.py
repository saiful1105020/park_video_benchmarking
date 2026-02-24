import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import requests

with open("../wandb_username.txt", "r") as f:
    wandb_username = f.read().strip()

with open("../project_name.txt", "r") as f:
    project_name = f.read().strip()

with open("../project_dir.txt", "r") as f:
    project_dir = f.read().strip()

CSV_PATH = f"/localdisk1/{project_dir}/{project_name}/data/metadata/country_counts.csv"  # your file: columns like country,count
CACHE_DIR = Path("geodata_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Natural Earth Admin 0 countries (110m)
NE_ZIP_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
NE_ZIP_PATH = CACHE_DIR / "ne_110m_admin_0_countries.zip"

# Download if needed
if not NE_ZIP_PATH.exists():
    r = requests.get(NE_ZIP_URL, timeout=60)
    r.raise_for_status()
    NE_ZIP_PATH.write_bytes(r.content)

# Read Natural Earth shapefile from the zip
world = gpd.read_file(f"zip://{NE_ZIP_PATH}")

df = pd.read_csv(CSV_PATH)
# If you have duplicates per country, aggregate
df = df.groupby("country", as_index=False)["count"].sum()
df["binary_count"] = (df["count"] > 0).astype(int)

# Natural Earth typically uses 'ADMIN' as the country name column
merged = world.merge(df, left_on="ADMIN", right_on="country", how="left")
merged["count"] = merged["count"].fillna(0)
# merged["count"] = 1+np.log(merged["count"]+1)  # Add 1 to avoid log(0)
merged["binary_count"] = merged["binary_count"].fillna(0)

# ax = merged.plot(column="count", legend=True, figsize=(14, 7))
ax = merged.plot(column="binary_count", legend=False, figsize=(14, 7), cmap="YlGn", edgecolor="black", linewidth=0.5)
ax.set_axis_off()
# plt.title("Counts by Country")
plt.tight_layout()
plt.savefig(f"/localdisk1/{project_dir}/{project_name}/results/R1_Dataset/plots/world_map.png", dpi=300)
plt.show()
