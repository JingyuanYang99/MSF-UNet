import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

"""
{
“GIIRS”：（192,216，1000），
“AGRI”：（576,648，通道数），
“ERA5”：（192,216,37,2）
}
"""

def prev_15min_time(t: str) -> str:
    hh = int(t[8:10])
    mm = 45
    hh -= 1
    return t[:8] + f"{hh:02d}{mm:02d}00"


def list_files_recursive(root_dir: str, ext: str = ".HDF"):
    """递归列出 root_dir 下所有指定后缀文件（返回完整路径）。"""
    ext_u = ext.upper()
    out = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.upper().endswith(ext_u):
                out.append(os.path.join(r, fn))
    out.sort()
    return out


FIG_DIR = r'/media/ub/SU710/data/fig'  # 你如果在Linux上跑，建议改成Linux路径


"""制作GIIRS数据"""
def make_GIIRS(GIIRS_root, group_time_strat, group_time_end, target_time, need_reorientation=False):
    tile_h, tile_w = 16, 8
    H, W = 192, 216
    ntr, ntc = H // tile_h, W // tile_w  # 12, 27

    tiles = []
    GIIRS_files = [
        fp for fp in list_files_recursive(GIIRS_root, ext=".HDF")
        if (group_time_strat in os.path.basename(fp) or group_time_end in os.path.basename(fp))
    ]
    GIIRS_files.sort()

    time_start_all = []
    time_end_all = []

    for file_path in GIIRS_files:
        fn = os.path.basename(file_path)
        print(fn)

        time_start = fn[-45:-31]
        time_end = fn[-30:-16]
        time_start_all.append(time_start)
        time_end_all.append(time_end)

        with h5py.File(file_path, "r") as f:
            LW = np.array(f["Data"]["ES_RealLW"], dtype=np.float32).T  # (128, nLW)
            MW = np.array(f["Data"]["ES_RealMW"], dtype=np.float32).T  # (128, nMW)
            spec = np.concatenate((LW, MW), axis=1)  # (128, K)
            K = spec.shape[1]

        if need_reorientation:
            spec = spec[::-1, :]

        tile = spec.reshape(tile_h, tile_w, K, order='F')  # (16, 8, K)
        tiles.append(tile)

    if len(tiles) != ntr * ntc:
        raise ValueError(f"GIIRS tile count mismatch: got {len(tiles)}, need {ntr * ntc}")

    tiles = np.stack(tiles, axis=0)  # (324, 16, 8, K)
    tiles = tiles.reshape(ntr, ntc, tile_h, tile_w, K)  # (12,27,16,8,K)
    GIIRS_grid = tiles.transpose(0, 2, 1, 3, 4).reshape(H, W, K)  # (192,216,K)

    """制作时间mask——与目标时间的时间差"""
    time_start = pd.to_datetime(time_start_all, format='%Y%m%d%H%M%S')
    time_end = pd.to_datetime(time_end_all, format='%Y%m%d%H%M%S')
    target_time = pd.to_datetime(target_time, format='%Y%m%d%H%M%S')
    time_mean = time_start + (time_end - time_start) / 2

    delta_minutes = (time_mean.values[None, :] - target_time.values[:, None]).astype('timedelta64[s]').astype(float) / 60.0
    delta_tile = delta_minutes.reshape(len(target_time), ntr, ntc)
    delta_time_all = np.repeat(np.repeat(delta_tile, tile_h, axis=1), tile_w, axis=2)  # (T,192,216)

    """画图（可选）"""
    c = 300
    x = GIIRS_grid[:, :, c]
    m = x > -900
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure()
    plt.imshow(np.where(m, x, np.nan))
    plt.colorbar()
    plt.title(f"GIIRS channel {c}" + group_time_strat + '_' + group_time_end)
    out_png = os.path.join(FIG_DIR, f"GIIRS_channel_{c}_{group_time_strat}_{group_time_end}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    return GIIRS_grid, delta_time_all


"""制作AGRI数据"""
def make_AGRI(AGRI_root, group_time_strat, group_time_end, GIIRS_AGRI_csv_path, need_reorientation=False):
    GIIRS_AGRI_csv = np.loadtxt(GIIRS_AGRI_csv_path, delimiter=',', dtype='str')
    prev_time = prev_15min_time(group_time_strat)

    AGRI_files = [
        fp for fp in list_files_recursive(AGRI_root, ext=".HDF")
        if ("FDI" in os.path.basename(fp))
        and (prev_time in os.path.basename(fp) or group_time_strat in os.path.basename(fp) or group_time_end in os.path.basename(fp))
    ]
    AGRI_files.sort()

    AGRI_grids = []
    file_times = []

    for file_path in AGRI_files:
        fn = os.path.basename(file_path)
        print(fn)
        file_time = fn[-45:-31]

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
            if need_reorientation:
                patches5 = patches5[:, ::-1, ...]
            patches6 = patches5.reshape(n_tiles_r * n_tiles_c, tile_h, tile_w, 3, 3, C, order='F')
            patches7 = patches6.reshape(n_tiles_r, n_tiles_c, tile_h, tile_w, 3, 3, C)

            grid = patches7.transpose(0, 2, 1, 3, 4, 5, 6).reshape(H, W, 3, 3, C)
            AGRI_grid = grid.transpose(0, 2, 1, 3, 4).reshape(H * 3, W * 3, C)  # (576,648,15)

            """画图（可选）"""
            c = 10
            x = AGRI_grid[:, :, c]
            m = x > -900
            os.makedirs(FIG_DIR, exist_ok=True)
            plt.figure()
            plt.imshow(np.where(m, x, np.nan))
            plt.colorbar()
            plt.title(f"AGRI channel {c}" + "_" + file_time)
            out_png = os.path.join(FIG_DIR, f"AGRI_channel_{c}_{file_time}.png")
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()

            AGRI_grids.append(AGRI_grid)
            file_times.append(file_time)

    if len(AGRI_grids) == 0:
        return None, None

    AGRI_all = np.stack(AGRI_grids, axis=0)  # (T,576,648,15)
    return AGRI_all, file_times


"""制作ERA5数据（保持你原逻辑：ERA5_folder 下直接找 .nc）"""
def make_ERA5(ERA5_folder, group_time_strat, group_time_end, GIIRS_ERA5_csv_path, need_reorientation=False):
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
    r22 = GIIRS_ERA5_csv[:, 10].astype(np.int32)
    c22 = GIIRS_ERA5_csv[:, 11].astype(np.int32)

    w11 = GIIRS_ERA5_csv[:, 12].astype(np.float32)
    w12 = GIIRS_ERA5_csv[:, 13].astype(np.float32)
    w21 = GIIRS_ERA5_csv[:, 14].astype(np.float32)
    w22 = GIIRS_ERA5_csv[:, 15].astype(np.float32)

    valid_rc = (
        (r11 >= 0) & (r11 < H_era5) & (c11 >= 0) & (c11 < W_era5) &
        (r12 >= 0) & (r12 < H_era5) & (c12 >= 0) & (c12 < W_era5) &
        (r21 >= 0) & (r21 < H_era5) & (c21 >= 0) & (c21 < W_era5) &
        (r22 >= 0) & (r22 < H_era5) & (c22 >= 0) & (c22 < W_era5)
    )

    n_tiles_r, n_tiles_c = 12, 27
    tile_h, tile_w = 16, 8
    H, W = n_tiles_r * tile_h, n_tiles_c * tile_w

    ERA5_grids = []
    file_times = []

    for fn in ERA5_files:
        print(fn)
        file_time = fn[-23:-9]
        file_path = os.path.join(ERA5_folder, fn)

        with xr.open_dataset(file_path) as ds:
            u = np.array(ds['u'])  # (37,721,1440)
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
            u_interp = np.sum(u4 * w4b, axis=2).T
            v_interp = np.sum(v4 * w4b, axis=2).T

            era5_uv[valid_rc, :, 0] = u_interp
            era5_uv[valid_rc, :, 1] = v_interp

        L, UV = 37, 2
        x = era5_uv.reshape(n_tiles_r * n_tiles_c, tile_h * tile_w, L, UV)
        if need_reorientation:
            x = x[:, ::-1, ...]
        x = x.reshape(n_tiles_r * n_tiles_c, tile_h, tile_w, L, UV, order='F')
        x = x.reshape(n_tiles_r, n_tiles_c, tile_h, tile_w, L, UV)
        ERA5_grid = x.transpose(0, 2, 1, 3, 4, 5).reshape(H, W, L, UV)

        """画图（可选）"""
        lev = 15
        step = 4
        U = ERA5_grid[::step, ::step, lev, 0]
        V = ERA5_grid[::step, ::step, lev, 1]
        valid = (U != -999) & (V != -999)
        Y, X = np.where(valid)

        os.makedirs(FIG_DIR, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.quiver(X, Y, U[valid], -V[valid], scale=700)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.title(f"ERA5 wind vectors (lev={lev})" + "_" + file_time)
        out_png = os.path.join(FIG_DIR, f"ERA5_wind_lev{lev}_{file_time}.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()

        ERA5_grids.append(ERA5_grid)
        file_times.append(file_time)
        ds.close()

    if len(ERA5_grids) == 0:
        return None, None

    ERA5_all = np.stack(ERA5_grids, axis=0)
    return ERA5_all, file_times


"""GIIRS单独存储（两小时一份，只存一次）"""
def save_giirs_once(out_root: str, group_time_start: str, group_time_end: str, GIIRS_grid: np.ndarray) -> str:
    giirs_dir = os.path.join(out_root, "GIIRS_store")
    os.makedirs(giirs_dir, exist_ok=True)

    giirs_id = f"{group_time_start}_{group_time_end}"
    giirs_fname = f"GIIRS_{giirs_id}.npz"
    giirs_path = os.path.join(giirs_dir, giirs_fname)

    if not os.path.exists(giirs_path):
        for try_num in range(5):
            try:
                np.savez_compressed(
                    giirs_path,
                    giirs_id=giirs_id,
                    group_time_start=group_time_start,
                    group_time_end=group_time_end,
                    GIIRS=GIIRS_grid.astype(np.float32),
                )
                data = np.load(giirs_path)
                GIIRS= data['GIIRS'].astype(np.float32)
                break
            except:
                print(giirs_path + " save error,try again")
                if try_num == 4:
                    print(giirs_path + " try save 5 times, still wrong")

    return os.path.relpath(giirs_path, out_root)


"""存为训练和测试的npz文件（不再重复存GIIRS，只存GIIRS_ref）"""
def save_15min_npz_split(
    out_root: str,
    GIIRS_ref: str,
    AGRI_grid_all: np.ndarray,
    AGRI_file_times: list,
    ERA5_grid_all: np.ndarray,
    ERA5_file_times: list,
    delta_time_all: np.ndarray,
):
    out_with = os.path.join(out_root, "with_era5")
    out_no = os.path.join(out_root, "no_era5")
    os.makedirs(out_with, exist_ok=True)
    os.makedirs(out_no, exist_ok=True)

    era5_map = {t: i for i, t in enumerate(ERA5_file_times)} if ERA5_file_times is not None else {}

    T = len(AGRI_file_times)
    assert AGRI_grid_all.shape[0] == T

    for i in range(1, T):
        t = AGRI_file_times[i]

        AGRI_curr = AGRI_grid_all[i].astype(np.float32)
        AGRI_prev = AGRI_grid_all[i - 1].astype(np.float32)
        GIIRS_delta_time = delta_time_all[i - 1].astype(np.float32)

        is_hour = (t[10:14] == "0000")

        if is_hour and (t in era5_map) and (ERA5_grid_all is not None):
            ERA5 = ERA5_grid_all[era5_map[t]].astype(np.float32)
            out_path = os.path.join(out_with, f"{t}.npz")
            for try_num in range(5):
                try:
                    np.savez_compressed(
                        out_path,
                        time=t,
                        GIIRS_ref=GIIRS_ref,
                        AGRI_curr=AGRI_curr,
                        AGRI_prev=AGRI_prev,
                        GIIRS_delta_time=GIIRS_delta_time,
                        ERA5=ERA5,
                    )
                    data = np.load(out_path)
                    AGRI_curr = data['AGRI_curr'].astype(np.float32)
                    break
                except:
                    print(out_path+" save error,try again")
                    if try_num == 4:
                        print(out_path+" try save 5 times, still wrong")

        else:
            out_path = os.path.join(out_no, f"{t}.npz")
            for try_num in range(5):
                try:
                    np.savez_compressed(
                        out_path,
                        time=t,
                        GIIRS_ref=GIIRS_ref,
                        AGRI_curr=AGRI_curr,
                        AGRI_prev=AGRI_prev,
                        GIIRS_delta_time=GIIRS_delta_time,
                    )
                    data = np.load(out_path)
                    AGRI_curr = data['AGRI_curr'].astype(np.float32)
                    break
                except:
                    print(out_path + " save error,try again")
                    if try_num == 4:
                        print(out_path + " try save 5 times, still wrong")


def need_reorientation(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        first = f.readline().split(',')[0].strip()
    return first != '-999'


if __name__ == '__main__':
    GIIRS_AGRI_dir = r'/media/ub/SU710/data/GIIRS_AGRI_matched'
    GIIRS_ERA5_dir = r'/media/ub/SU710/data/GIIRS_ERA5_matched'

    # 改成“根目录”（多层目录没问题）
    GIIRS_root = r'/media/ub/SU710/data/giirs'   # 例如：/media/ub/SU710/data/giirs/202509/20250901/...
    AGRI_root = r'/media/ub/SU710/data/agri'     # 按你的实际路径修改：/media/ub/SU710/data/agri/202509/20250901/...

    ERA5_root = r'/media/ub/SU710/data/era5'
    out_root = r'/media/ub/SU710/data/npz_files'

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    for fname in sorted(os.listdir(GIIRS_AGRI_dir)):
        sample_time = fname[-36:-19]

        test_path1 = os.path.join(out_root, "with_era5", f"{sample_time}.npz")
        test_path2 = os.path.join(out_root, "no_era5", f"{sample_time}.npz")

        # if os.path.exists(test_path1) or os.path.exists(test_path2):
        #     print("skip group:", fname)
        #     continue
        GIIRS_AGRI_csv_path = os.path.join(GIIRS_AGRI_dir, fname)
        GIIRS_ERA5_csv_path = os.path.join(GIIRS_ERA5_dir, fname)
        reorientation = need_reorientation(GIIRS_AGRI_csv_path)

        group_time_start = fname[-33:-23]
        group_time_end = fname[-18:-8]

        date_str = group_time_start[:8]
        ERA5_folder = os.path.join(ERA5_root, date_str)

        AGRI_grid_all, AGRI_file_times = make_AGRI(
            AGRI_root, group_time_start, group_time_end, GIIRS_AGRI_csv_path, reorientation
        )
        if AGRI_grid_all is None:
            continue

        target_time = AGRI_file_times[1:]
        GIIRS_grid, delta_time_all = make_GIIRS(
            GIIRS_root, group_time_start, group_time_end, target_time, reorientation
        )

        ERA5_grid_all, ERA5_file_times = make_ERA5(
            ERA5_folder, group_time_start, group_time_end, GIIRS_ERA5_csv_path, reorientation
        )

        GIIRS_ref = save_giirs_once(out_root, group_time_start, group_time_end, GIIRS_grid)

        save_15min_npz_split(
            out_root,
            GIIRS_ref,
            AGRI_grid_all,
            AGRI_file_times,
            ERA5_grid_all,
            ERA5_file_times,
            delta_time_all
        )