import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# =========================
# 配置
# =========================
pred_npz_path = "../data/pred_all_data_uvw/20250923011500_pred.npz"
out_dir = "../data/simple_3d_slices"
os.makedirs(out_dir, exist_ok=True)

pressure_levels = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
    250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
    20, 10, 7, 5, 3, 2, 1
]

# 选几个代表层，和你示意图差不多
target_pressures = [825, 750, 650, 550, 350, 300]
level_indices = [pressure_levels.index(p) for p in target_pressures]

# 箭头抽稀
stride = 6

# 可选：裁剪局地区域。None 表示不裁
# 例如 crop = (x0, x1, y0, y1)
crop = None
# crop = (80, 200, 20, 140)


def crop_3d(arr, crop_box):
    if crop_box is None:
        return arr
    x0, x1, y0, y1 = crop_box
    return arr[:, y0:y1, x0:x1]


def plot_3d_slices(pred_u, pred_v, scalar_3d, pressure_levels, level_indices,
                   save_path, cbar_label, cmap="viridis",
                   symmetric=False, stride=6, title=None):
    """
    pred_u, pred_v, scalar_3d: shape = (L, H, W)
    """
    _, H, W = scalar_3d.shape
    yy, xx = np.mgrid[0:H, 0:W]

    valid_scalar = scalar_3d[np.isfinite(scalar_3d)]
    if valid_scalar.size == 0:
        raise ValueError("没有有效填色数据。")

    if symmetric:
        vmax = np.percentile(np.abs(valid_scalar), 99)
        vmin = -vmax
    else:
        vmin = np.percentile(valid_scalar, 1)
        vmax = np.percentile(valid_scalar, 99)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)

    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    for lev in level_indices:
        p = pressure_levels[lev]
        u = pred_u[lev]
        v = pred_v[lev]
        scalar = scalar_3d[lev]

        z_plane = np.full_like(xx, p, dtype=np.float32)

        # ===== 等压面填色 =====
        facecolors = cmap_obj(norm(scalar))
        ax.plot_surface(
            xx, yy, z_plane,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            shade=False,
            linewidth=0,
            antialiased=False,
            alpha=0.5   # 半透明，避免把箭头完全盖住
        )

        # ===== 箭头抽稀 =====
        xx_sub = xx[::stride, ::stride]
        yy_sub = yy[::stride, ::stride]
        u_sub = u[::stride, ::stride]
        v_sub = v[::stride, ::stride]

        valid = np.isfinite(u_sub) & np.isfinite(v_sub)

        # ===== 关键：让箭头浮在切片平面上方 =====
        # 注意 pressure 轴反转后，更小的 pressure 看起来更“高”
        dz = 15
        zz_sub = z_plane[::stride, ::stride] - dz

        ax.quiver(
            xx_sub[valid],
            yy_sub[valid],
            zz_sub[valid],
            u_sub[valid],
            -v_sub[valid],   # 图像坐标里 y 向下，故取负
            np.zeros_like(u_sub[valid]),
            length=1.0,
            normalize=False,
            color="k",
            linewidth=0.7,
            arrow_length_ratio=0.35
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Pressure (hPa)")

    p_sel = [pressure_levels[i] for i in level_indices]
    ax.set_zlim(max(p_sel), min(p_sel))  # 气压轴反转
    ax.view_init(elev=22, azim=-120)

    try:
        ax.set_box_aspect((1.1, 1.0, 1.2))
    except Exception:
        pass

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    cb = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
    cb.set_label(cbar_label)

    if title is not None:
        ax.set_title(title, pad=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =========================
# 主程序
# =========================
with np.load(pred_npz_path) as data:
    pred_u = data["pred_u"]
    pred_v = data["pred_v"]
    pred_w = data["pred_w"]

pred_u = crop_3d(pred_u, crop)
pred_v = crop_3d(pred_v, crop)
pred_w = crop_3d(pred_w, crop)

time_str = os.path.basename(pred_npz_path).replace("_pred.npz", "")

# 1) 水平风速切片图
ws = np.sqrt(pred_u ** 2 + pred_v ** 2)
save_path_ws = os.path.join(out_dir, f"{time_str}_uvw_slices_ws.png")
plot_3d_slices(
    pred_u=pred_u,
    pred_v=pred_v,
    scalar_3d=ws,
    pressure_levels=pressure_levels,
    level_indices=level_indices,
    save_path=save_path_ws,
    cbar_label="Horizontal Wind Speed (m/s)",
    cmap="viridis",
    symmetric=False,
    stride=stride,
    title=f"{time_str}  horizontal wind speed slices"
)
print("saved:", save_path_ws)

# 2) 垂直速度切片图
save_path_w = os.path.join(out_dir, f"{time_str}_uvw_slices_w.png")
plot_3d_slices(
    pred_u=pred_u,
    pred_v=pred_v,
    scalar_3d=pred_w,
    pressure_levels=pressure_levels,
    level_indices=level_indices,
    save_path=save_path_w,
    cbar_label="Vertical Velocity (Pa/s)",
    cmap="RdBu_r",
    symmetric=True,
    stride=stride,
    title=f"{time_str}  vertical velocity slices"
)
print("saved:", save_path_w)