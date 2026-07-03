import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def radiance_to_BT(nu_cm, R_cm):
    """
    将辐射率 (mW/(m²·sr·cm⁻¹)) 转换为亮温 (K)，基于 Planck 函数反解。

    参数:
        nu_cm : float or np.ndarray
            波数 (单位: cm⁻¹)
        R_cm : float or np.ndarray
            光谱辐射率 (单位: mW/(m²·sr·cm⁻¹))

    返回:
        T_b : float or np.ndarray
            亮温 (单位: K)
    """
    # 常数 (SI 单位)
    h = 6.62607015e-34  # J·s
    c = 2.99792458e8  # m/s
    k_B = 1.380649e-23  # J/K

    # 单位换算
    nu_m = nu_cm[:, np.newaxis] * 100  # cm⁻¹ → m⁻¹
    R_m = R_cm * 0.00001  # mW/(m²·sr·cm⁻¹) → W/(m²·sr·m⁻¹)

    # Planck 反函数公式
    numerator = h * c * nu_m
    denominator = k_B * np.log(1 + (2 * h * c ** 2 * nu_m ** 3) / R_m)

    BT = numerator / denominator
    return BT

# 数据目录
base_path = r"F:\fusion_3D_wind\GIIRS"
hdf_files = sorted([f for f in os.listdir(base_path) if f.endswith(".HDF")])
print(f"共找到 {len(hdf_files)} 个文件。")

# 容器
all_lat_LW, all_lon_LW, all_R_LW, all_BT_LW = [], [], [],[]
all_lat_MW, all_lon_MW, all_R_MW, all_BT_MW = [], [], [],[]

# 读取数据
for f in hdf_files:
    full_path = os.path.join(base_path, f)
    try:
        with h5py.File(full_path, "r") as file:
            Latitude_LW = file["Geolocation"]["Latitude_LW"][:]
            Longitude_LW = file["Geolocation"]["Longitude_LW"][:]
            ES_RealLW = np.array(file["Data"]["ES_RealLW"]).astype("float32")
            WN_LW = np.array(file["Data"]["WN_LW"])
            BT_LW = radiance_to_BT(WN_LW, ES_RealLW)
            Latitude_MW = file["Geolocation"]["Latitude_MW"][:]
            Longitude_MW = file["Geolocation"]["Longitude_MW"][:]
            ES_RealMW = np.array(file["Data"]["ES_RealMW"]).astype("float32")
            WN_MW = np.array(file["Data"]["WN_MW"])
            BT_MW = radiance_to_BT(WN_MW, ES_RealMW)
            all_lat_LW.append(Latitude_LW)
            all_lon_LW.append(Longitude_LW)
            all_R_LW.append(ES_RealLW)
            all_BT_LW.append(BT_LW)
            all_lat_MW.append(Latitude_MW)
            all_lon_MW.append(Longitude_MW)
            all_R_MW.append(ES_RealMW)
            all_BT_MW.append(BT_MW)
    except Exception as e:
        print(f"读取失败: {f}，错误：{e}")

# 合并
lat_LW = np.concatenate(all_lat_LW)
lon_LW = np.concatenate(all_lon_LW)
R_LW = np.concatenate(all_R_LW, axis=1)
BT_LW = np.concatenate(all_BT_LW, axis=1)

# 提取亮温通道（900 cm⁻¹）
band_index = 354
radiance = R_LW[band_index]
BT = BT_LW[band_index]

# 作图（Cartopy）
fig = plt.figure(figsize=(14, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
sc = ax.scatter(lon_LW, lat_LW, c=radiance, cmap='jet', s=10, marker='s', transform=ccrs.PlateCarree())

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