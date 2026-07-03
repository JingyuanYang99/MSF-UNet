import numpy as np
import os, glob
from pyhdf.SD import SD, SDC
import h5py
import os, sys
import csv
import shutil
import copy
from datetime import datetime


'''读取AGRI4km分辨率数据经纬度'''
file_lon_lat = r'../data/auxiliary_data/FY4B-_DISK_1050E_GEO_NOM_LUT_20240227000000_4000M_V0001.raw'
raw_image = np.fromfile(file_lon_lat)
raw_image = raw_image.reshape(-1, 2)
n = 2748
lat = raw_image[:, 0].reshape(n, n)
lat[lat > 999] = np.nan
lon = raw_image[:, 1].reshape(n, n)
lon[lon > 999] = np.nan
lat = np.array(lat)
lon = np.array(lon)
lon_find = copy.deepcopy(lon)
lon_find[lon_find<0]+=360

def myFind(mesh_lat, mesh_lon, source_lat, source_lon):
    last_dis = 1e10
    last_lat_index = len(mesh_lat[0]) // 2  # 从中间列开始找
    last_lon_index = 0
    while True:
        new_lat_index, new_lon_index, new_dis = oneSearch(last_lat_index, last_lon_index, mesh_lat, mesh_lon,
                                                          source_lat, source_lon)
        if new_dis >= last_dis:  # 没找到更好的
            break
        last_dis = new_dis
        last_lat_index = new_lat_index
        last_lon_index = new_lon_index

    if last_dis < 1:
        res_lat_index = last_lon_index
        res_lon_index = last_lat_index
    else:
        res_lat_index = -1
        res_lon_index = -1
    return res_lat_index, res_lon_index


def oneSearch(start_lat_index, start_lon_index, mesh_lat, mesh_lon, source_lat, source_lon):
    lon_index = binary_search_lat(mesh_lat[:, start_lat_index], source_lat)
    lat_index = binary_search_lon(mesh_lon[lon_index], source_lon)
    find_lat = mesh_lat[lon_index][lat_index]
    find_lon = mesh_lon[lon_index][lat_index]
    if source_lon < 0:
        source_lon += 360
    dis = (source_lat - find_lat) ** 2 + (source_lon - find_lon) ** 2
    return lat_index, lon_index, dis


# 二分查找
def binary_search_lat(arr, target):
    start = 0
    end = len(arr) - 1
    min_index = -1

    if arr[(start + end) // 2 - 10] < arr[(start + end) // 2 + 10]:  # 单调递增数组
        while start <= end:
            mid = (start + end) // 2

            # 更新最小索引
            if min_index == -1 or abs(arr[mid] - target) < abs(arr[min_index] - target):
                min_index = mid

            if arr[mid] == target:
                return mid
            elif np.isnan(arr[mid]):  # 中间值是nan
                if not np.isnan(arr[start]):  # 左边不是nan
                    end = mid - 1
                else:
                    start = mid + 1
            elif arr[mid] > target:
                end = mid - 1
            else:
                start = mid + 1
    else:  # 单调递减数组
        while start <= end:
            mid = (start + end) // 2

            # 更新最小索引
            if min_index == -1 or abs(arr[mid] - target) < abs(arr[min_index] - target):
                min_index = mid

            if arr[mid] == target:
                return mid
            elif np.isnan(arr[mid]):  # 中间值是nan
                if not np.isnan(arr[start]):  # 左边不是nan
                    end = mid - 1
                else:
                    start = mid + 1
            elif arr[mid] > target:
                start = mid + 1
            else:
                end = mid - 1

    return min_index


def binary_search_lon(arr, target):
    if target < 0:
        target += 360
    start = 0
    end = len(arr) - 1
    min_index = -1

    if arr[(start + end) // 2 - 10] < arr[(start + end) // 2 + 10]:  # 单调递增数组
        while start <= end:
            mid = (start + end) // 2

            # 更新最小索引
            if min_index == -1 or abs(arr[mid] - target) < abs(arr[min_index] - target):
                min_index = mid

            if arr[mid] == target:
                return mid
            elif np.isnan(arr[mid]):  # 中间值是nan
                if not np.isnan(arr[start]):  # 左边不是nan
                    end = mid - 1
                else:
                    start = mid + 1
            elif arr[mid] > target:
                end = mid - 1
            else:
                start = mid + 1
    else:  # 单调递减数组
        while start <= end:
            mid = (start + end) // 2

            # 更新最小索引
            if min_index == -1 or abs(arr[mid] - target) < abs(arr[min_index] - target):
                min_index = mid

            if arr[mid] == target:
                return mid
            elif np.isnan(arr[mid]):  # 中间值是nan
                if not np.isnan(arr[start]):  # 左边不是nan
                    end = mid - 1
                else:
                    start = mid + 1
            elif arr[mid] > target:
                start = mid + 1
            else:
                end = mid - 1

    return min_index

def match_point(GIIRS_csvpath, FY4_folder_dir, matched_csv_save_path, lat, lon, lon_find):
    csv_data = np.loadtxt(GIIRS_csvpath, delimiter=',', dtype='str', skiprows=1)
    source_pionts = csv_data
    # 返回满足文件夹路径和路径模式的所有文件路径列表，赋值给match_data_paths变量
    # 递归获取 FY4_folder_dir 下所有 .HDF
    match_data_paths = glob.glob(os.path.join(FY4_folder_dir, '**', '*.HDF'), recursive=True)
    match_data_paths += glob.glob(os.path.join(FY4_folder_dir, '**', '*.hdf'), recursive=True)
    match_data_paths = sorted(set(match_data_paths))

    last_FDI_match_data_path = ''

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
                -999, -999,-999 ]
            last_file_records.append(record)
            continue
        source_time = source_point[2]
        source_time  = datetime.strptime(source_time, "%Y-%m-%d %H:%M:%S")
        source_time = source_time.strftime("%Y%m%d%H%M%S")
        FDI_match_data_path = None
        # 将csv文件中时间符合的文件挑出来
        for j in range(len(match_data_paths)):
            t = match_data_paths[j]
            if source_time in t and 'FDI' in t:
                FDI_match_data_path = t
                # print(source_time, FDI_match_data_path)
                break
        # 如果没找到就找开始下一轮循环
        if (FDI_match_data_path is None):
            # print(source_time + "未匹配")
            continue
        # 找到source经纬度对应在FY-4A全圆盘中的经纬度下标
        lat_index, lon_index = myFind(lat, lon, float(source_point[0]), float(source_point[1]))

        # 如果找不到就找下一行的source数据
        if lat_index == -1 or lon_index == -1:
            print("lat:",source_point[0],"lon:",source_point[1])
            record = [
                -999,
                -999,
                source_point[-1], -999, -999,
                -999, -999,-999 ]
            last_file_records.append(record)
            continue
        # 只有处理的文件名不同才执行以下代码，即一个文件存一次
        if FDI_match_data_path != last_FDI_match_data_path:
            # 保存上一个有效文件
            if last_is_save:  # 默认为True，即第一次一定保存
                # 写入数据
                csvFile = open(matched_csv_save_path, 'a', newline='')
                writer = csv.writer(csvFile)
                writer.writerows(last_file_records)
                csvFile.close()

                print("写入 " + last_FDI_match_data_path)
            else:
                shutil.move(last_FDI_match_data_path, r'/media/ub/SU710/data/agri/trouble')
                print("剔除 " + last_FDI_match_data_path)
            print(i / len(source_pionts))
            # 重置
            last_file_records = []
            last_is_save = True
            last_FDI_match_data_path = FDI_match_data_path
            # print(last_FDI_match_data_path)
            # 打开L1级数据，读取L1级数据，并且处理异常文件，将异常文件直接跳过
            try:
                hdf_obj_L1 = h5py.File(FDI_match_data_path, "r")
                i=1
            except:
                last_FDI_match_data_path = ''
                print(FDI_match_data_path, "无法打开")
                continue
            # 对打开的文件进行处理
            num_channel = 15
            cal_table_list = []
            dn_list = []
            for k in range(num_channel):
                # print(k)
                channel_number = "{:02d}".format(k + 1)
                DN_channel_name = 'NOMChannel' + channel_number
                CAL_channel_name = 'CALChannel' + channel_number
                # 读取DN值
                dn = np.array(hdf_obj_L1['Data'][DN_channel_name][:])
                # dn[dn==65535]=1
                # 读取定标表
                cal_table = np.array(hdf_obj_L1['Calibration'][CAL_channel_name][:])
                dn_list.append(dn)
                cal_table_list.append(cal_table)
            # 读取时间数据
            time_data = np.array(hdf_obj_L1['NOMObs']['NOMObsTime'][:])
            # 关闭文件
            hdf_obj_L1.close()
        # 对每个点进行处理
        # 由于时间的形状为（2748,2）表示每行观测时间的开始与结束，因此应该取lat_index
        match_time = time_data[lat_index][0]
        if match_time == 999:
            match_time = np.nan
        record = [source_point[0], source_point[1], source_point[-1], source_time, lat_index,
                  lon_index, lat[lat_index][lon_index], lon[lat_index, lon_index]]
        last_file_records.append(record)

    # 保存最后一个有效文件
    if last_is_save:
        # 写入数据
        csvFile = open(matched_csv_save_path, 'a', newline='')
        writer = csv.writer(csvFile)
        writer.writerows(last_file_records)
        csvFile.close()

        print("写入 " + last_FDI_match_data_path)
    else:
        shutil.move(last_FDI_match_data_path, r'/media/ub/SU710/data/agri/trouble')
        print("剔除 " + last_FDI_match_data_path)


GIIRS_csv_folder = r'/media/ub/SU710/data/GIIRS_csv'
source_FY4_folder = r'/media/ub/SU710/data/agri/202509'
matched_save_folder = r'/media/ub/SU710/data/GIIRS_AGRI_matched'
for fname in os.listdir(GIIRS_csv_folder):
    GIIRS_csvpath = os.path.join(GIIRS_csv_folder, fname)
    matched_csv_save_path = os.path.join(matched_save_folder,fname[-33:-4] + '.csv')
    match_point(GIIRS_csvpath,source_FY4_folder,matched_csv_save_path,lat,lon,lon_find)