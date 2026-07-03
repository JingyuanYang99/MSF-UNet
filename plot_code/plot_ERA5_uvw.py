import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


data_dir = r"/home/ub/yjy/3D wind field/fusion_3D_wind/data/with_era5_uvw/all"
img_dir = r"/home/ub/yjy/3D wind field/fusion_3D_wind/fig/era5_images"


pressure_levels = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
    250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
    20, 10, 7, 5, 3, 2, 1
]


def save_uv_wind_image(pred_u, pred_v, save_png_path, level, time_str,
                       stride=4, vmin=None, vmax=None, cmap="viridis"):
    u = pred_u[level]
    v = pred_v[level]
    ws = np.sqrt(u ** 2 + v ** 2)

    pressure = pressure_levels[level]

    h, w = u.shape
    yy, xx = np.mgrid[0:h, 0:w]

    xx_sub = xx[::stride, ::stride]
    yy_sub = yy[::stride, ::stride]
    u_sub = u[::stride, ::stride]
    v_sub = v[::stride, ::stride]

    valid = np.isfinite(u_sub) & np.isfinite(v_sub)

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_axes([0.08, 0.12, 0.72, 0.76])
    cax = fig.add_axes([0.84, 0.12, 0.03, 0.76])

    im = ax.imshow(ws, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{time_str}  {pressure} hPa", fontsize=12)
    ax.axis("off")

    ax.quiver(
        xx_sub[valid],
        yy_sub[valid],
        u_sub[valid],
        -v_sub[valid],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.002
    )

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Horizontal Wind Speed (m/s)")

    plt.savefig(save_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_w_image(pred_u, pred_v, pred_w, save_png_path, level, time_str,
                 stride=4, vmin=None, vmax=None, cmap="RdBu_r"):
    w = pred_w[level]
    u = pred_u[level]
    v = pred_v[level]
    pressure = pressure_levels[level]

    h, wid = w.shape
    yy, xx = np.mgrid[0:h, 0:wid]

    xx_sub = xx[::stride, ::stride]
    yy_sub = yy[::stride, ::stride]
    u_sub = u[::stride, ::stride]
    v_sub = v[::stride, ::stride]

    valid = np.isfinite(u_sub) & np.isfinite(v_sub)

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_axes([0.08, 0.12, 0.72, 0.76])
    cax = fig.add_axes([0.84, 0.12, 0.03, 0.76])

    im = ax.imshow(w, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{time_str}  {pressure} hPa  w", fontsize=12)
    ax.axis("off")

    ax.quiver(
        xx_sub[valid],
        yy_sub[valid],
        u_sub[valid],
        -v_sub[valid],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.002,
        color="k"
    )

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Vertical Velocity")

    plt.savefig(save_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_era5_from_npz(npz_path):
    with np.load(npz_path) as data:
        era5 = data["ERA5"].astype(np.float32)

    era5[era5 <= -900] = np.nan

    pred_u = np.transpose(era5[:, :, :, 0], (2, 0, 1))
    pred_v = np.transpose(era5[:, :, :, 1], (2, 0, 1))
    pred_w = np.transpose(era5[:, :, :, 2], (2, 0, 1))

    return pred_u, pred_v, pred_w


for pressure in pressure_levels:
    os.makedirs(os.path.join(img_dir, "uv", f"{pressure}hPa"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "w", f"{pressure}hPa"), exist_ok=True)


file_list = sorted(
    [f for f in os.listdir(data_dir) if f.endswith(".npz")]
)

print("data_dir absolute path:", os.path.abspath(data_dir))
print("img_dir absolute path:", os.path.abspath(img_dir))
print("file_list length:", len(file_list))


ws_samples = []
w_samples = []

for filename in file_list:
    npz_path = os.path.join(data_dir, filename)
    pred_u, pred_v, pred_w = load_era5_from_npz(npz_path)

    ws = np.sqrt(pred_u ** 2 + pred_v ** 2)
    ws_valid = ws[np.isfinite(ws)]
    if ws_valid.size > 0:
        ws_samples.append(ws_valid[::100])

    w_valid = pred_w[np.isfinite(pred_w)]
    if w_valid.size > 0:
        w_samples.append(w_valid[::100])


if len(ws_samples) == 0:
    raise ValueError("没有可用于统计水平风速分位数的有效数据。")
if len(w_samples) == 0:
    raise ValueError("没有可用于统计 w 分位数的有效数据。")

ws_samples = np.concatenate(ws_samples)
w_samples = np.concatenate(w_samples)

global_ws_min = np.percentile(ws_samples, 1)
global_ws_max = np.percentile(ws_samples, 99)

global_w_abs = np.percentile(np.abs(w_samples), 99)
global_w_min = -global_w_abs
global_w_max = global_w_abs

print(f"global ws percentile 1%   = {global_ws_min:.4f}")
print(f"global ws percentile 99%  = {global_ws_max:.4f}")
print(f"global |w| percentile 99% = {global_w_abs:.4f}")


for filename in file_list:
    print(filename)
    npz_path = os.path.join(data_dir, filename)
    npz_path = os.path.join(data_dir, filename)
    pred_u, pred_v, pred_w = load_era5_from_npz(npz_path)

    time_str = Path(filename).stem

    for level, pressure in enumerate(pressure_levels):
        uv_level_dir = os.path.join(img_dir, "uv", f"{pressure}hPa")
        w_level_dir = os.path.join(img_dir, "w", f"{pressure}hPa")

        uv_png_path = os.path.join(
            uv_level_dir,
            f"{time_str}_{pressure}hPa_uv.png"
        )
        w_png_path = os.path.join(
            w_level_dir,
            f"{time_str}_{pressure}hPa_w.png"
        )

        save_uv_wind_image(
            pred_u=pred_u,
            pred_v=pred_v,
            save_png_path=uv_png_path,
            level=level,
            time_str=time_str,
            stride=4,
            vmin=global_ws_min,
            vmax=global_ws_max,
            cmap="viridis"
        )

        save_w_image(
            pred_u=pred_u,
            pred_v=pred_v,
            pred_w=pred_w,
            save_png_path=w_png_path,
            level=level,
            time_str=time_str,
            stride=4,
            vmin=global_w_min,
            vmax=global_w_max,
            cmap="RdBu_r"
        )

        print(f"saved uv png: {uv_png_path}")
        print(f"saved w  png: {w_png_path}")