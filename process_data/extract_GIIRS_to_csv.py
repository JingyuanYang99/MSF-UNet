# import os
# import re
# import shutil
# from dataclasses import dataclass
# from datetime import datetime, timedelta
#
# import h5py
# import numpy as np
# import pandas as pd
#
# TIME_RE = re.compile(r"_(\d{14})_(\d{14})_")
# EXPECTED_ROWS = 12 * 128 * 27  # 41472
#
#
# def parse_times_from_name(fname: str):
#     m = TIME_RE.search(fname)
#     if not m:
#         raise ValueError(f"文件名未匹配到时间段: {fname}")
#     t0 = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
#     t1 = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
#     return t0, t1
#
#
# def group_start_odd_hour(t0: datetime):
#     # 归并到“奇数整点起始”的 1 小时窗口
#     if t0.hour == 0:
#         return t0.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
#     if t0.hour % 2 == 1:
#         return t0.replace(minute=0, second=0, microsecond=0)
#     return t0.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
#
#
# def group_end_1h(gs: datetime):
#     return gs + timedelta(hours=1)
#
#
# def read_lonlat_lw(file_path: str):
#     with h5py.File(file_path, "r") as f:
#         geo = f["Geolocation"]
#         lat_lw = np.array(geo["Latitude_LW"][:], dtype=np.float64).ravel()
#         lon_lw = np.array(geo["Longitude_LW"][:], dtype=np.float64).ravel()
#     if lat_lw.shape[0] != lon_lw.shape[0]:
#         raise ValueError(f"lat/lon长度不一致: {file_path}, {lat_lw.shape[0]} vs {lon_lw.shape[0]}")
#     return lat_lw, lon_lw
#
#
# def count_csv_rows_fast(csv_path: str) -> int:
#     with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
#         n_lines = sum(1 for _ in f)
#     return max(0, n_lines - 1)  # 减表头
#
#
# def finalize_group(csv_path: str, wrong_dir: str):
#     if not csv_path or not os.path.exists(csv_path):
#         return
#     n = count_csv_rows_fast(csv_path)
#     if n != EXPECTED_ROWS:
#         os.makedirs(wrong_dir, exist_ok=True)
#         dst = os.path.join(wrong_dir, os.path.basename(csv_path))
#         if os.path.exists(dst):
#             base, ext = os.path.splitext(dst)
#             dst = f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
#         shutil.move(csv_path, dst)
#         print(f"[WRONG] 行数={n}，已移入: {dst}")
#     else:
#         print(f"[OK] 行数={n}，保留: {csv_path}")
#
#
# @dataclass
# class ProcessState:
#     current_key: tuple | None = None  # (gs, ge)
#     current_csv: str | None = None
#
#
# def process_one_dir_with_state(in_dir: str, out_dir: str, wrong_dir: str, state: ProcessState):
#     """
#     处理单个日目录，但不在函数结束时 finalize 最后一个 group，
#     以便跨天继续追加同一个小时窗。
#     """
#     files = sorted([fn for fn in os.listdir(in_dir) if fn.upper().endswith(".HDF")])
#
#     for fn in files:
#         t0, t1 = parse_times_from_name(fn)
#         gs = group_start_odd_hour(t0)
#         ge = group_end_1h(gs)
#         key = (gs, ge)
#
#         # 时间窗切换：先 finalize 上一个窗，再切换到新窗
#         if key != state.current_key:
#             finalize_group(state.current_csv, wrong_dir)
#             state.current_key = key
#             state.current_csv = os.path.join(
#                 out_dir,
#                 f"GIIRS_FULLCOVER_{gs.strftime('%Y%m%d%H%M%S')}_{ge.strftime('%Y%m%d%H%M%S')}.csv",
#             )
#
#         file_path = os.path.join(in_dir, fn)
#         lat_lw, lon_lw = read_lonlat_lw(file_path)
#
#         df = pd.DataFrame(
#             {
#                 "lat": lat_lw,
#                 "lon": lon_lw,
#                 "group_start_utc": gs.strftime("%Y-%m-%d %H:%M:%S"),
#                 "group_end_utc": ge.strftime("%Y-%m-%d %H:%M:%S"),
#                 "file_start_utc": t0.strftime("%Y-%m-%d %H:%M:%S"),
#                 "file_end_utc": t1.strftime("%Y-%m-%d %H:%M:%S"),
#                 "src_file": fn,
#             }
#         )
#
#         # 追加写入；如果跨天接着写同一个 CSV，header 也会自动正确处理
#         df.to_csv(state.current_csv, mode="a", header=not os.path.exists(state.current_csv), index=False)
#
#
# def process_month_dirs(base_dir: str, yyyymm: str, out_dir: str, wrong_dir: str):
#     """
#     base_dir: /media/ub/SU710/data/giirs
#     yyyymm:   202509
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     os.makedirs(wrong_dir, exist_ok=True)
#
#     month_dir = os.path.join(base_dir, yyyymm)
#     if not os.path.isdir(month_dir):
#         raise FileNotFoundError(f"月份目录不存在: {month_dir}")
#
#     # 找到形如 YYYYMMDD 的日目录
#     day_dirs = sorted([d for d in os.listdir(month_dir) if re.fullmatch(r"\d{8}", d)])
#     if not day_dirs:
#         raise FileNotFoundError(f"未找到日目录(YYYYMMDD): {month_dir}")
#
#     state = ProcessState()
#
#     for day in day_dirs:
#         in_dir = os.path.join(month_dir, day)
#         if not os.path.isdir(in_dir):
#             continue
#         print(f"== Processing day dir: {in_dir}")
#         process_one_dir_with_state(in_dir, out_dir, wrong_dir, state)
#
#     # 所有日目录处理完，最后再 finalize 一次
#     finalize_group(state.current_csv, wrong_dir)
#
#
# if __name__ == "__main__":
#     BASE_DIR = "/media/ub/SU710/data/giirs"
#     OUT_DIR = "/media/ub/SU710/data/giirs_csv"
#     WRONG_DIR = "/media/ub/SU710/data/giirs_csv_wrong"
#
#     process_month_dirs(
#         base_dir=BASE_DIR,
#         yyyymm="202509",
#         out_dir=OUT_DIR,
#         wrong_dir=WRONG_DIR,
#     )

import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd

TIME_RE = re.compile(r"_(\d{14})_(\d{14})_")
EXPECTED_ROWS = 12 * 128 * 27  # 41472


def parse_times_from_name(fname: str):
    m = TIME_RE.search(fname)
    if not m:
        raise ValueError(f"文件名未匹配到时间段: {fname}")
    t0 = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    t1 = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
    return t0, t1


def group_start_odd_hour(t0: datetime):
    # 归并到“奇数整点起始”的 1 小时窗口
    if t0.hour == 0:
        return t0.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    if t0.hour % 2 == 1:
        return t0.replace(minute=0, second=0, microsecond=0)
    return t0.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def group_end_1h(gs: datetime):
    return gs + timedelta(hours=1)


def read_lonlat_lw(file_path: str):
    with h5py.File(file_path, "r") as f:
        geo = f["Geolocation"]
        lat_lw = np.array(geo["Latitude_LW"][:], dtype=np.float64).ravel()
        lon_lw = np.array(geo["Longitude_LW"][:], dtype=np.float64).ravel()
    if lat_lw.shape[0] != lon_lw.shape[0]:
        raise ValueError(f"lat/lon长度不一致: {file_path}, {lat_lw.shape[0]} vs {lon_lw.shape[0]}")
    return lat_lw, lon_lw


def count_csv_rows_fast(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        n_lines = sum(1 for _ in f)
    return max(0, n_lines - 1)  # 减表头


def finalize_group(csv_path: str, wrong_dir: str):
    if not csv_path or not os.path.exists(csv_path):
        return
    n = count_csv_rows_fast(csv_path)
    if n != EXPECTED_ROWS:
        os.makedirs(wrong_dir, exist_ok=True)
        dst = os.path.join(wrong_dir, os.path.basename(csv_path))
        if os.path.exists(dst):
            base, ext = os.path.splitext(dst)
            dst = f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
        shutil.move(csv_path, dst)
        print(f"[WRONG] 行数={n}，已移入: {dst}")
    else:
        print(f"[OK] 行数={n}，保留: {csv_path}")


@dataclass
class ProcessState:
    current_key: tuple | None = None  # (gs, ge)
    current_csv: str | None = None


def process_one_dir_with_state(in_dir: str, out_dir: str, wrong_dir: str, state: ProcessState):
    """
    处理单个日目录，但不在函数结束时 finalize 最后一个 group，
    以便跨天继续追加同一个小时窗。
    """
    files = sorted([fn for fn in os.listdir(in_dir) if fn.upper().endswith(".HDF")])

    for fn in files:
        t0, t1 = parse_times_from_name(fn)
        gs = group_start_odd_hour(t0)
        ge = group_end_1h(gs)
        key = (gs, ge)

        # 时间窗切换：先 finalize 上一个窗，再切换到新窗
        if key != state.current_key:
            finalize_group(state.current_csv, wrong_dir)
            state.current_key = key
            state.current_csv = os.path.join(
                out_dir,
                f"GIIRS_FULLCOVER_{gs.strftime('%Y%m%d%H%M%S')}_{ge.strftime('%Y%m%d%H%M%S')}.csv",
            )

        file_path = os.path.join(in_dir, fn)
        lat_lw, lon_lw = read_lonlat_lw(file_path)

        df = pd.DataFrame(
            {
                "lat": lat_lw,
                "lon": lon_lw,
                "group_start_utc": gs.strftime("%Y-%m-%d %H:%M:%S"),
                "group_end_utc": ge.strftime("%Y-%m-%d %H:%M:%S"),
                "file_start_utc": t0.strftime("%Y-%m-%d %H:%M:%S"),
                "file_end_utc": t1.strftime("%Y-%m-%d %H:%M:%S"),
                "src_file": fn,
            }
        )

        # 追加写入；如果跨天接着写同一个 CSV，header 也会自动正确处理
        df.to_csv(state.current_csv, mode="a", header=not os.path.exists(state.current_csv), index=False)


def process_month_dirs(base_dir: str, yyyymm: str, out_dir: str, wrong_dir: str):
    """
    base_dir: /media/ub/SU710/data/giirs
    yyyymm:   202509
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    month_dir = os.path.join(base_dir, yyyymm)
    if not os.path.isdir(month_dir):
        raise FileNotFoundError(f"月份目录不存在: {month_dir}")

    # 找到形如 YYYYMMDD 的日目录
    day_dirs = sorted([d for d in os.listdir(month_dir) if re.fullmatch(r"\d{8}", d)])
    if not day_dirs:
        raise FileNotFoundError(f"未找到日目录(YYYYMMDD): {month_dir}")

    state = ProcessState()

    for day in day_dirs:
        in_dir = os.path.join(month_dir, day)
        if not os.path.isdir(in_dir):
            continue
        print(f"== Processing day dir: {in_dir}")
        process_one_dir_with_state(in_dir, out_dir, wrong_dir, state)

    # 所有日目录处理完，最后再 finalize 一次
    finalize_group(state.current_csv, wrong_dir)


if __name__ == "__main__":
    BASE_DIR = "/media/ub/SU710/data/giirs"
    OUT_DIR = "/media/ub/SU710/data/giirs_csv"
    WRONG_DIR = "/media/ub/SU710/data/giirs_csv_wrong"

    process_month_dirs(
        base_dir=BASE_DIR,
        yyyymm="202509",
        out_dir=OUT_DIR,
        wrong_dir=WRONG_DIR,
    )

i=1