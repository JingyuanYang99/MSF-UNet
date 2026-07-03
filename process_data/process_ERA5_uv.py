import os
import time

import pandas as pd
import xarray as xr

ROOT = r"/media/ub/Extreme SSD/UV2"  # 根目录：遍历这里所有 nc
OUT_BASE = r"/media/ub/Extreme SSD/data/uv"  # 根目录：下面会自动建 20240723 / 20240724

for root, _, files in os.walk(ROOT):
    for f in files:
        if not f.endswith(".nc"):
            continue
        # 跳过已经拆分出来的文件，避免递归重复处理
        if f.startswith("ERA5_") and f.endswith("_level.nc"):
            continue

        SRC = os.path.join(root, f)
        print(f"\n=== 处理: {SRC} ===")

        # 打开数据
        ds = xr.open_dataset(SRC)
        print(ds.data_vars)

        # 只保留 u/v/w，减小体积（若需要其它变量，自己加进去）
        vars_to_keep = [v for v in ['u', 'v'] if v in ds.variables]
        if not vars_to_keep:
            ds.close()
            raise RuntimeError("文件中没有找到变量 'u' 或 'v'。")

        times = pd.to_datetime(ds['valid_time'].values)
        print(f"总时次：{len(times)} 个（{times.min()} ~ {times.max()}）")

        for t in times:
            start_time = time.time()
            yyyy, mm, dd, HH = t.year, t.month, t.day, t.hour
            date_dir = os.path.join(OUT_BASE, f"{yyyy:04d}{mm:02d}{dd:02d}")
            os.makedirs(date_dir, exist_ok=True)

            out_path = os.path.join(
                date_dir, f"ERA5_{yyyy:04d}{mm:02d}{dd:02d}{HH:02d}0000_level.nc"
            )
            print(out_path)
            if os.path.exists(out_path):
                print(f"[跳过] 已存在：{out_path}")
                continue

            # 取该小时切片，并去掉单一维度
            sub = ds[vars_to_keep].sel(valid_time=t).squeeze(drop=True)

            # 去掉某些容易干扰保存的坐标变量（若不存在会忽略）
            sub = sub.drop_vars(['number', 'expver'], errors='ignore')

            # 统一变量维度顺序：pressure_level, latitude, longitude
            for v in vars_to_keep:
                sub[v] = sub[v].transpose('pressure_level', 'latitude', 'longitude')

            # 压缩设置（显著减小文件体积）
            comp = dict(zlib=True, complevel=1)
            encoding = {v: comp for v in vars_to_keep}

            sub.to_netcdf(out_path, format='NETCDF4', engine='netcdf4', encoding=encoding)
            use_time = time.time() - start_time
            print(f"[已写出] {out_path}, 用时：{use_time:.2f}")

        ds.close()

print("全部完成 ✅")
