import numpy as np
import os, glob
from pyhdf.SD import SD, SDC
import h5py
import os, sys
import csv
import shutil
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import xarray as xr
"""
{“GIIRS”：（192,216，1000），
“AGRI”：（576,648，通道数），
“ERA5”：（192,216,37,2）
}
"""
def prev_15min_time(t):
    hh = int(t[8:10])
    mm = 0
    mm += 45
    hh -= 1

    return t[:8] + f"{hh:02d}{mm:02d}00"

FIG_DIR = r'F:\fusion_3D_wind\fig_new'
"""制作GIIRS数据"""
def make_GIIRS(GIIRS_folder,group_time_strat,group_time_end):
    tile_h, tile_w = 16, 8
    H, W = 192, 216
    ntr, ntc = H // tile_h, W // tile_w  # 12, 27

    tiles = []
    GIIRS_files = sorted([
        fn for fn in os.listdir(GIIRS_folder)
        if fn.upper().endswith('.HDF') and (group_time_strat in fn or group_time_end in fn)
    ])

    for fn in GIIRS_files:
        print(fn)
        file_path = os.path.join(GIIRS_folder, fn)
        with h5py.File(file_path, "r") as f:
            LW = np.array(f["Data"]["ES_RealLW"], dtype=np.float32).T  # (128, nLW)
            MW = np.array(f["Data"]["ES_RealMW"], dtype=np.float32).T  # (128, nMW)
            spec = np.concatenate((LW, MW), axis=1)  # (128, K)
            K = spec.shape[1]
        tile = spec[::-1, :].reshape(tile_h, tile_w, K,order = 'F')  # (16, 8, K)
        tiles.append(tile)

    if len(tiles) != ntr * ntc:
        raise ValueError(f"tile count mismatch: got {len(tiles)}, need {ntr*ntc}")

    tiles = np.stack(tiles, axis=0)  # (324, 16, 8, K)
    tiles = tiles.reshape(ntr, ntc, tile_h, tile_w, K)  # (12,27,16,8,K)
    GIIRS_grid = tiles.transpose(0, 2, 1, 3, 4).reshape(H, W, K)  # (192,216,K)
    """画图"""
    c = 300  # 第 c 个通道
    x = GIIRS_grid[:, :, c]

    m = x > -900
    plt.figure()
    plt.imshow(np.where(m, x, np.nan))
    plt.colorbar()
    plt.title(f"GIIRS channel {c}"+group_time_strat+'_'+group_time_end)
    out_png = os.path.join(
        FIG_DIR,
        f"GIIRS_channel_{c}_{group_time_strat}_{group_time_end}.png"
    )
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    return GIIRS_grid

"""制作AGRI数据"""
def make_AGRI(AGRI_folder, group_time_strat, group_time_end, GIIRS_AGRI_csv_path):
    GIIRS_AGRI_csv = np.loadtxt(GIIRS_AGRI_csv_path, delimiter=',', dtype='str')
    prev_time = prev_15min_time(group_time_strat)
    AGRI_files = sorted([
        fn for fn in os.listdir(AGRI_folder)
        if fn.upper().endswith('.HDF')
        and (prev_time in fn or group_time_strat in fn or group_time_end in fn)
        and ("FDI" in fn)
    ])

    AGRI_grids = []
    file_times = []
    for fn in AGRI_files:
        print(fn)
        file_time = fn[-45:-31]
        file_path = os.path.join(AGRI_folder, fn)

        with h5py.File(file_path, "r") as hdf_obj_L1:
            num_channel = 15
            cal_table_list = []
            dn_list = []

            for k in range(num_channel):
                channel_number = "{:02d}".format(k + 1)
                DN_channel_name = 'NOMChannel' + channel_number
                CAL_channel_name = 'CALChannel' + channel_number

                dn = np.array(hdf_obj_L1['Data'][DN_channel_name][:])
                cal_table = np.array(hdf_obj_L1['Calibration'][CAL_channel_name][:])

                dn_list.append(dn)
                cal_table_list.append(cal_table)

            dn_cube = np.stack(dn_list, axis=-1).astype(np.float32)  # (2748,2748,15)

            agri_r = GIIRS_AGRI_csv[:, 4].astype(np.int32)
            agri_c = GIIRS_AGRI_csv[:, 5].astype(np.int32)

            N = agri_r.shape[0]
            C = dn_cube.shape[-1]

            patches = np.full((N, 3, 3, C), -999, dtype=np.float32)
            valid_rc = (agri_r != -999) & (agri_c != -999)

            for i in np.where(valid_rc)[0]:
                r = agri_r[i]
                c = agri_c[i]
                patches[i] = dn_cube[r:r + 3, c:c + 3, :]

            patches_cal = np.full_like(patches, -999.0, dtype=np.float32)
            for k in range(15):
                lut = cal_table_list[k]
                dn_k = patches[..., k].astype(np.int32)
                valid_dn = (dn_k >= 0) & (dn_k < lut.shape[0])
                patches_cal[..., k][valid_dn] = lut[dn_k[valid_dn]]

            n_tiles_r, n_tiles_c = 12, 27
            tile_h, tile_w = 16, 8
            H, W = n_tiles_r * tile_h, n_tiles_c * tile_w  # (192,216)

            patches5 = patches_cal.reshape(n_tiles_r * n_tiles_c, tile_h * tile_w, 3, 3, C)
            patches5 = patches5[:, ::-1, ...]
            patches6 = patches5.reshape(n_tiles_r * n_tiles_c, tile_h, tile_w, 3, 3, C, order='F')
            patches7 = patches6.reshape(n_tiles_r, n_tiles_c, tile_h, tile_w, 3, 3, C)
            grid = patches7.transpose(0, 2, 1, 3, 4, 5, 6).reshape(H, W, 3, 3, C)
            AGRI_grid = grid.transpose(0, 2, 1, 3, 4).reshape(H * 3, W * 3, C)
            """画图"""
            c = 10  # 第 c 个通道
            x = AGRI_grid[:, :, c]

            m = x > -900
            plt.figure()
            plt.imshow(np.where(m, x, np.nan))
            plt.colorbar()
            plt.title(f"AGRI channel {c}"+"_"+file_time)
            out_png = os.path.join(
                FIG_DIR,
                f"AGRI_channel_{c}_{file_time}.png"
            )
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()
            AGRI_grids.append(AGRI_grid)
            file_times.append(file_time)
    if len(AGRI_grids) == 0:
        return None
    AGRI_all = np.stack(AGRI_grids, axis=0)  # (T, 576, 648, 15)
    return AGRI_all, file_times

"""制作ERA5数据"""
def make_ERA5(ERA5_folder, group_time_strat, group_time_end, GIIRS_ERA5_csv_path):
    GIIRS_ERA5_csv = np.loadtxt(GIIRS_ERA5_csv_path, delimiter=',', dtype='str')

    ERA5_files = sorted([
        fn for fn in os.listdir(ERA5_folder)
        if fn.endswith('.nc') and (group_time_strat in fn or group_time_end in fn)
    ])

    H_era5, W_era5 = 721, 1440
    N = GIIRS_ERA5_csv.shape[0]

    r11 = GIIRS_ERA5_csv[:, 4].astype(np.int32)
    c11 = GIIRS_ERA5_csv[:, 5].astype(np.int32)
    r12 = GIIRS_ERA5_csv[:, 6].astype(np.int32)
    c12 = GIIRS_ERA5_csv[:, 7].astype(np.int32)
    r21 = GIIRS_ERA5_csv[:, 8].astype(np.int32)
    c21 = GIIRS_ERA5_csv[:, 9].astype(np.int32)
    r22 = GIIRS_ERA5_csv[:,10].astype(np.int32)
    c22 = GIIRS_ERA5_csv[:,11].astype(np.int32)

    w11 = GIIRS_ERA5_csv[:,12].astype(np.float32)
    w12 = GIIRS_ERA5_csv[:,13].astype(np.float32)
    w21 = GIIRS_ERA5_csv[:,14].astype(np.float32)
    w22 = GIIRS_ERA5_csv[:,15].astype(np.float32)

    valid_rc = (
        (r11 >= 0) & (r11 < H_era5) & (c11 >= 0) & (c11 < W_era5) &
        (r12 >= 0) & (r12 < H_era5) & (c12 >= 0) & (c12 < W_era5) &
        (r21 >= 0) & (r21 < H_era5) & (c21 >= 0) & (c21 < W_era5) &
        (r22 >= 0) & (r22 < H_era5) & (c22 >= 0) & (c22 < W_era5)
    )

    n_tiles_r, n_tiles_c = 12, 27
    tile_h, tile_w = 16, 8
    H, W = n_tiles_r * tile_h, n_tiles_c * tile_w  # (192,216)

    ERA5_grids = []
    file_times = []

    for fn in ERA5_files:
        print(fn)
        file_time = fn[-23:-9]
        file_path = os.path.join(ERA5_folder, fn)
        ds = xr.open_dataset(file_path)

        u = np.array(ds['u'])  # (37,721,1440) 你的数据假设如此
        v = np.array(ds['v'])

        era5_uv = np.full((N, 37, 2), -999, dtype=np.float32)

        if np.any(valid_rc):
            idx11 = r11[valid_rc] * W_era5 + c11[valid_rc]
            idx12 = r12[valid_rc] * W_era5 + c12[valid_rc]
            idx21 = r21[valid_rc] * W_era5 + c21[valid_rc]
            idx22 = r22[valid_rc] * W_era5 + c22[valid_rc]

            idx4 = np.stack([idx11, idx12, idx21, idx22], axis=1)  # (Nv,4)
            w4 = np.stack([w11[valid_rc], w12[valid_rc], w21[valid_rc], w22[valid_rc]], axis=1).astype(np.float32)

            u_flat = u.reshape(37, -1)
            v_flat = v.reshape(37, -1)

            u4 = np.take(u_flat, idx4, axis=1)  # (37,Nv,4)
            v4 = np.take(v_flat, idx4, axis=1)

            w4b = w4[None, :, :]
            u_interp = np.sum(u4 * w4b, axis=2).T  # (Nv,37)
            v_interp = np.sum(v4 * w4b, axis=2).T

            era5_uv[valid_rc, :, 0] = u_interp
            era5_uv[valid_rc, :, 1] = v_interp

        L, UV = 37, 2
        x = era5_uv.reshape(n_tiles_r * n_tiles_c, tile_h * tile_w, L, UV)
        x = x[:, ::-1, ...]
        x = x.reshape(n_tiles_r * n_tiles_c, tile_h, tile_w, L, UV, order='F')
        x = x.reshape(n_tiles_r, n_tiles_c, tile_h, tile_w, L, UV)
        ERA5_grid = x.transpose(0, 2, 1, 3, 4, 5).reshape(H, W, L, UV)
        """画图"""
        lev = 15
        step = 4

        U = ERA5_grid[::step, ::step, lev, 0]
        V = ERA5_grid[::step, ::step, lev, 1]

        # 缺测掩膜
        valid = (U != -999) & (V != -999)

        Y, X = np.where(valid)

        plt.figure(figsize=(10, 6))
        plt.quiver(
            X, Y,
            U[valid],
            -V[valid],
            scale=700
        )

        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')

        plt.title(f"ERA5 wind vectors (lev={lev})" + "_" + file_time)
        plt.xlabel("x")
        plt.ylabel("y")
        out_png = os.path.join(
            FIG_DIR,
            f"ERA5_wind_lev{lev}_{file_time}.png"
        )
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        ERA5_grids.append(ERA5_grid)
        file_times.append(file_time)

        ds.close()

    if len(ERA5_grids) == 0:
        return None

    ERA5_all = np.stack(ERA5_grids, axis=0)  # (T,192,216,37,2)
    return ERA5_all, file_times

"""存为训练和测试的npz文件"""
def save_15min_npz_split(
    out_root: str,
    GIIRS_grid: np.ndarray,
    AGRI_grid_all: np.ndarray,
    AGRI_file_times: list,
    ERA5_grid_all: np.ndarray,
    ERA5_file_times: list,
):
    out_with = os.path.join(out_root, "with_era5")
    out_no = os.path.join(out_root, "no_era5")
    os.makedirs(out_with, exist_ok=True)
    os.makedirs(out_no, exist_ok=True)

    era5_map = {t: i for i, t in enumerate(ERA5_file_times)}

    T = len(AGRI_file_times)
    assert AGRI_grid_all.shape[0] == T

    giirs = GIIRS_grid.astype(np.float32)

    for i in range(1, T):
        t = AGRI_file_times[i]  # YYYYMMDDHHMMSS

        agri_now = AGRI_grid_all[i].astype(np.float32)
        agri_prev = AGRI_grid_all[i - 1].astype(np.float32)

        is_hour = (t[10:14] == "0000")

        if is_hour and t in era5_map:
            era5 = ERA5_grid_all[era5_map[t]].astype(np.float32)
            out_path = os.path.join(out_with, f"{t}.npz")
            np.savez_compressed(
                out_path,
                time=t,
                giirs=giirs,
                agri_now=agri_now,
                agri_prev=agri_prev,
                era5=era5,
            )
        else:
            out_path = os.path.join(out_no, f"{t}.npz")
            np.savez_compressed(
                out_path,
                time=t,
                giirs=giirs,
                agri_now=agri_now,
                agri_prev=agri_prev,
            )


if __name__ == '__main__':
    GIIRS_AGRI_dir = r'F:\fusion_3D_wind\GIIRS_AGRI_matched'
    GIIRS_ERA5_dir = r'F:\fusion_3D_wind\GIIRS_ERA5_matched'
    GIIRS_folder = r'F:\fusion_3D_wind\GIIRS'
    AGRI_folder = r'F:\fusion_3D_wind\AGRI'
    ERA5_root = r'F:\fusion_3D_wind\ERA5'
    out_root = r'F:\fusion_3D_wind\npz_files_new'
    for fname in sorted(os.listdir(GIIRS_AGRI_dir)):
        GIIRS_AGRI_csv_path = os.path.join(GIIRS_AGRI_dir, fname)
        GIIRS_ERA5_csv_path = os.path.join(GIIRS_ERA5_dir, fname)

        group_time_start = fname[-33:-23]
        group_time_end = fname[-18:-8]

        date_str = group_time_start[:8]
        ERA5_folder = os.path.join(ERA5_root, date_str)
        GIIRS_grid = make_GIIRS(GIIRS_folder,group_time_start,group_time_end)
        AGRI_grid_all, AGRI_file_times = make_AGRI(AGRI_folder,group_time_start,group_time_end,GIIRS_AGRI_csv_path)
        ERA5_grid_all, ERA5_file_times = make_ERA5(ERA5_folder,group_time_start,group_time_end,GIIRS_ERA5_csv_path)
        save_15min_npz_split(
            out_root,
            GIIRS_grid,
            AGRI_grid_all,
            AGRI_file_times,
            ERA5_grid_all,
            ERA5_file_times,
        )
    # GIIRS_AGRI_csv_path = r'F:\fusion_3D_wind\GIIRS_AGRI_matched\20250918010000_20250918020000.csv'
    # GIIRS_ERA5_csv_path = r'F:\fusion_3D_wind\GIIRS_ERA5_matched\20250918010000_20250918020000.csv'
    # group_time_strat = GIIRS_AGRI_csv_path[-33:-23]
    # group_time_end = GIIRS_AGRI_csv_path[-18:-8]
    # GIIRS_folder = r'F:\fusion_3D_wind\GIIRS'
    # AGRI_folder = r'F:\fusion_3D_wind\AGRI'
    # ERA5_folder = r'F:\fusion_3D_wind\ERA5\20250918'
    # GIIRS_grid = make_GIIRS(GIIRS_folder,group_time_strat,group_time_end)
    # AGRI_grid_all,AGRI_file_times = make_AGRI(AGRI_folder,group_time_strat,group_time_end,GIIRS_AGRI_csv_path)
    # ERA5_grid_all,ERA5_file_times = make_ERA5(ERA5_folder,group_time_strat,group_time_end,GIIRS_ERA5_csv_path)
    # out_root = r"F:\fusion_3D_wind\npz_files"
    # save_15min_npz_split(
    #     out_root,
    #     GIIRS_grid,
    #     AGRI_grid_all,
    #     AGRI_file_times,
    #     ERA5_grid_all,
    #     ERA5_file_times,
    # )
    i=1