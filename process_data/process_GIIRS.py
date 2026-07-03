import h5py
import numpy as np
import matplotlib.pyplot as plt

import numpy as np


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

# 打开文件
file = h5py.File(r"F:\fusion_3D_wind\test\FY4B-_GIIRS-_N_REGC_1050E_L1-_IRD-_MULT_NUL_20250918010328_20250918010836_012KM_001V1.HDF", "r")

print(file.keys())

# 查看 Data 和 Geolocation 下面的内容
print("Geolocation", list(file["Geolocation"].keys()))
print("Data:", list(file["Data"].keys()))
# 读取辐亮度数据
Latitude_LW = file["Geolocation"]["Latitude_LW"][:]
Longitude_LW = file["Geolocation"]["Longitude_LW"][:]
ES_RealLW  = np.array(file["Data"]["ES_RealLW"]).astype("float32")
WN_LW = np.array(file["Data"]["WN_LW"])
BT_LW = radiance_to_BT(WN_LW,ES_RealLW)
Latitude_MW = file["Geolocation"]["Latitude_MW"][:]
Longitude_MW = file["Geolocation"]["Longitude_MW"][:]
ES_RealMW = np.array(file["Data"]["ES_RealMW"]).astype("float32")
WN_MW = np.array(file["Data"]["WN_MW"])
BT_MW = radiance_to_BT(WN_MW,ES_RealMW)
i=1


# i=0