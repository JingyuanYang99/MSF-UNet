import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_wind_image(pred_u, pred_v, save_png_path, level, time_str,
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

    fig = plt.figure(figsize=(6, 5), dpi=300)

    ax = fig.add_axes([0.08, 0.12, 0.72, 0.76])   # 左，下，宽，高
    cax = fig.add_axes([0.84, 0.12, 0.03, 0.76])  # colorbar 固定位置

    im = ax.imshow(ws, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{time_str}  {pressure} hPa", fontsize=12)
    ax.axis("off")

    ax.quiver(
        xx_sub[valid],
        yy_sub[valid],
        u_sub[valid],
        v_sub[valid],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.002
    )

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Wind Speed (m/s)")

    plt.savefig(save_png_path, dpi=150)
    plt.close(fig)


output_dim =37
data_dir = "../data/plot_predict_wind_data"
img_dir = "../data/plot_predict_wind_image"

pressure_levels = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
    250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
    20, 10, 7, 5, 3, 2, 1
]
os.makedirs(img_dir, exist_ok=True)

file_list = sorted(
    [f for f in os.listdir(data_dir) if f.endswith(".npz")],
    key=lambda f: os.path.getmtime(os.path.join(data_dir, f))
)

# ====== 第二遍：用统一分位数色标画图 ======
for filename in file_list:
    pred_npz_path = os.path.join(data_dir, filename)
    data = np.load(pred_npz_path)

    pred_u = data["pred_u"]
    pred_v = data["pred_v"]

    time_str = Path(filename).stem

    for level in range(output_dim):
        pressure = pressure_levels[level]

        level_dir = os.path.join(img_dir, f"{pressure}hPa")
        os.makedirs(level_dir, exist_ok=True)

        save_png_path = os.path.join(
            level_dir,
            f"{Path(filename).stem}_{pressure}hPa.png"
        )

        save_wind_image(
            pred_u=pred_u,
            pred_v=pred_v,
            save_png_path=save_png_path,
            level=level,
            time_str=time_str,
            stride=3,
            # vmin=global_ws_min,
            # vmax=global_ws_max,
            cmap="viridis"
        )

        print(f"saved png: {save_png_path}")