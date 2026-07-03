import numpy as np
import os, glob
from pyhdf.SD import SD, SDC
import h5py
import os, sys
import csv
import shutil
import copy
from datetime import datetime
import xarray as xr

def find_four_points(lat0, lon0):
    dlat = 0.25
    dlon = 0.25

    lon0 = lon0 % 360.0

    i = int(np.floor((90.0 - lat0) / dlat))
    j = int(np.floor(lon0 / dlon))

    i = np.clip(i, 0, int(180 / dlat) - 1)
    j = np.clip(j, 0, int(360 / dlon) - 1)

    lat_i   = 90.0 - i * dlat
    lat_ip1 = lat_i - dlat
    lon_j   = j * dlon
    lon_jp1 = lon_j + dlon

    # 四个点的坐标
    p11 = (lat_i,   lon_j)
    p12 = (lat_i,   lon_jp1)
    p21 = (lat_ip1, lon_j)
    p22 = (lat_ip1, lon_jp1)

    # 四个点的下标
    idx = [
        (i,   j),     # p11
        (i,   j+1),   # p12
        (i+1, j),     # p21
        (i+1, j+1),   # p22
    ]

    return p11, p12, p21, p22, idx

def bilinear_weights(lat0, lon0, p11, p12, p21, p22):
    lat1, lon1 = p11
    _,    lon2 = p12
    lat2, _    = p21

    tx = (lon0 - lon1) / (lon2 - lon1)
    ty = (lat1 - lat0) / (lat1 - lat2)

    w11 = (1 - tx) * (1 - ty)
    w12 = tx * (1 - ty)
    w21 = (1 - tx) * ty
    w22 = tx * ty

    return w11, w12, w21, w22

def match_point(GIIRS_csvpath, matched_csv_save_path):
    csv_data = np.loadtxt(GIIRS_csvpath, delimiter=',', dtype='str', skiprows=1)
    source_pionts = csv_data
    last_file_records = []
    last_is_save = True
    for i in range(len(source_pionts)):
        source_point = source_pionts[i]
        # 保留经纬度缺测的行，并跳过
        if float(source_point[0]) < -999 or float(source_point[1]) < -999:
            record = [
                -999,
                -999,
                source_point[-1], -999, -999,
                -999, -999,-999, -999,
                -999, -999,-999, -999,
                -999, -999,-999]
            last_file_records.append(record)
            continue
        source_time = source_point[2]
        source_time  = datetime.strptime(source_time, "%Y-%m-%d %H:%M:%S")
        source_time = source_time.strftime("%Y%m%d%H%M%S")
        p11, p12, p21, p22, idx = find_four_points(float(source_point[0]), float(source_point[1]))
        w11, w12, w21, w22 = bilinear_weights(float(source_point[0]), float(source_point[1]), p11, p12, p21, p22)
        (i1, j1), (i2, j2), (i3, j3), (i4, j4) = idx
        record = [
            source_point[0], source_point[1],
            source_point[-1], source_time,
            i1, j1, i2, j2, i3, j3, i4, j4,
            w11, w12, w21, w22
        ]
        last_file_records.append(record)
    # 保存最后一个有效文件
    if last_is_save:
        # 写入数据
        csvFile = open(matched_csv_save_path, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerows(last_file_records)
        csvFile.close()

        print("写入 " + source_time)
# 生成ERA5的经纬度均匀网格
lat_era5 = np.arange(90.0, -90.0 - 0.25, -0.25)
lon_era5 = np.arange(0.0, 360.0, 0.25)
GIIRS_csv_folder = r'/media/ub/SU710/data/GIIRS_csv'
matched_save_folder = r'/media/ub/SU710/data/GIIRS_ERA5_matched'
# p11, p12, p21, p22 = find_four_points(32.1, 118.6)
# w11, w12, w21, w22 = bilinear_weights(32.1, 118.6, p11, p12, p21, p22)
for fname in os.listdir(GIIRS_csv_folder):
    GIIRS_csvpath = os.path.join(GIIRS_csv_folder, fname)
    matched_csv_save_path = os.path.join(matched_save_folder,fname[-33:-4] + '.csv')
    match_point(GIIRS_csvpath,matched_csv_save_path)
i=1



