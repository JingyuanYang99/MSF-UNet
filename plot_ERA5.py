import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

nc_path = "/data/era5/20250901/ERA5_20250901000000_level.nc"
save_dir = "/data/era5/20250901/plots_37_levels_region"
os.makedirs(save_dir, exist_ok=True)

ds = xr.open_dataset(nc_path)

ds_sub = ds.sel(
    latitude=slice(90, 0),
    longitude=slice(30, 180)
)

u_all = ds_sub["u"].values
v_all = ds_sub["v"].values
levels = ds_sub["pressure_level"].values
lat = ds_sub["latitude"].values
lon = ds_sub["longitude"].values

lon2d, lat2d = np.meshgrid(lon, lat)
stride = 6

for i, level in enumerate(levels):
    u = u_all[i]
    v = v_all[i]
    ws = np.sqrt(u ** 2 + v ** 2)

    fig, ax = plt.subplots(figsize=(10, 8))

    pcm = ax.pcolormesh(lon2d, lat2d, ws, shading="auto")
    plt.colorbar(pcm, ax=ax, label="Wind Speed (m/s)")

    ax.quiver(
        lon2d[::stride, ::stride],
        lat2d[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        color="k",
        scale=300
    )

    ax.set_title(f"ERA5 Wind Field at {int(level)} hPa")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(30, 180)
    ax.set_ylim(0, 90)

    save_path = os.path.join(save_dir, f"wind_{int(level)}hPa.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"saved: {save_path}")
