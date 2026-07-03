import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 数据目录
base_path = r"F:\fusion_3D_wind\test2"
hdf_files = sorted([f for f in os.listdir(base_path) if f.endswith(".HDF")])
print(f"共找到 {len(hdf_files)} 个文件。")

# 容器
all_lat, all_lon, all_rad = [], [], []

# 读取数据
for f in hdf_files:
    full_path = os.path.join(base_path, f)
    try:
        with h5py.File(full_path, "r") as file:
            lat = file["Geolocation"]["Latitude_LW"][:]
            lon = file["Geolocation"]["Longitude_LW"][:]
            rad = file["Data"]["ES_RealLW"][:]
            all_lat.append(lat)
            all_lon.append(lon)
            all_rad.append(rad)
    except Exception as e:
        print(f"读取失败: {f}，错误：{e}")

# 合并
lat = np.concatenate(all_lat)
lon = np.concatenate(all_lon)
rad_all = np.concatenate(all_rad, axis=1)

# 提取亮温通道（900 cm⁻¹）
band_index = 362
radiance = rad_all[band_index]

# 作图（Cartopy）
fig = plt.figure(figsize=(14, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
sc = ax.scatter(lon, lat, c=radiance, cmap='jet', s=10, marker='s', transform=ccrs.PlateCarree())

# 添加地图元素
ax.coastlines(resolution='10m', linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.gridlines(draw_labels=True, linestyle='--')

# 范围与标签
ax.set_extent([30, 170, 0, 70], crs=ccrs.PlateCarree())
plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7, pad=0.02,
             label="Radiance [mW/(m²·sr·cm⁻¹)]")
plt.title("FY-4B GIIRS 900 cm⁻¹", fontsize=16)
plt.tight_layout()
plt.show()