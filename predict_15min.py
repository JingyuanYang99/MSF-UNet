import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model import U_Net


# ============================================================
# 全局字体设置
# ============================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# 基础配置
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim_AGRI = 15
input_dim_GIIRS = 1690
output_dim = 37

ckpt_path = "../unet_model/checkpoint/unet_best.pth"

data_dir = "/3D wind field/fusion_3D_wind_uv/15min_wind/data"
save_dir = "/3D wind field/fusion_3D_wind_uv/15min_wind/15min_save_path"
img_dir = "/3D wind field/fusion_3D_wind_uv/15min_wind/fig/all_data_images_uv"

norm_param_path = "../data/auxiliary_data/normalized_parameters_mean_std_uvw.npz"

AGRI_mask = np.load("../data/auxiliary_data/AGRI_mask.npy")
GIIRS_mask = np.load("../data/auxiliary_data/GIIRS_mask.npy")

giirs_root = "/home/ub/data/npz_file/GIIRS_store"

pressure_levels = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
    250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
    20, 10, 7, 5, 3, 2, 1
]


# ============================================================
# 论文图配置
# ============================================================
CASE_PRESSURE = 850

# 两个代表性小时段，每个小时 5 个 15 min 时次
REPRESENTATIVE_BLOCKS = [
    [
        "20250412000000",
        "20250412001500",
        "20250412003000",
        "20250412004500",
        "20250412010000",
    ],
    [
        "20250412050000",
        "20250412051500",
        "20250412053000",
        "20250412054500",
        "20250412060000",
    ],
]

FORCE_REMAKE_NPZ = False

# 根据 AGRI_curr[:, :, 0:15] 的真实顺序确认
AGRI_BT108_IDX = 12

# BT 色标
BT_CMAP = "gray_r"
BT_VMIN = 200.0
BT_VMAX = 300.0

# 风速色标
WS_VMIN = 0.0
WS_VMAX = 35.0

# 自定义低饱和深蓝风速色带
WIND_CMAP = LinearSegmentedColormap.from_list(
    "wind_soft",
    ["#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5", "#084594", "#08306B"]
)

# 也可以直接使用：
# WIND_CMAP = "Blues"
# WIND_CMAP = "cividis"

# BT 等值线
BT_CONTOUR_LEVELS = [220, 235]
BT_CONTOUR_COLORS = ["#D62728", "#FFFFFF"]
BT_CONTOUR_LINEWIDTHS = [1.3, 1.0]

# 风矢量
WIND_VECTOR_STRIDE = 8
QUIVER_COLOR = "#222222"
QUIVER_WIDTH = 0.0020
QUIVER_ALPHA = 0.9

# 组合图设置
FIG_DPI = 600
PANEL_FIGSIZE_PER_COL = 3.0
PANEL_FIGSIZE_PER_ROW = 2.6

case_img_dir = os.path.join(img_dir, "case_study_850hPa_top_journal_style_4rows")
bt_dir = os.path.join(case_img_dir, "01_bt_individual")
wind_dir = os.path.join(case_img_dir, "02_wind_individual")
composite_dir = os.path.join(case_img_dir, "03_composite")

for d in [save_dir, case_img_dir, bt_dir, wind_dir, composite_dir]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 工具函数
# ============================================================
def normalize_giirs_ref(ref):
    ref = str(ref).strip()
    ref = ref.replace("array(", "")
    ref = ref.replace(")", "")
    ref = ref.replace("[", "")
    ref = ref.replace("]", "")
    ref = ref.replace("'", "")
    ref = ref.replace('"', "")
    ref = os.path.basename(ref)
    return ref


def zscore_channel_last(x, mean, std, eps=1e-6):
    mean = mean.reshape((1,) * (x.ndim - 1) + (-1,))
    std = std.reshape((1,) * (x.ndim - 1) + (-1,))

    std_safe = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    mean_safe = np.where(np.isfinite(mean), mean, 0.0)

    out = (x - mean_safe) / std_safe
    out = np.where(np.isnan(x), np.nan, out)
    return out


def inverse_norm(pred, mean, std):
    pred = pred[0].detach().cpu().numpy()
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    return pred * std + mean


def get_level_index(pressure):
    if pressure not in pressure_levels:
        raise ValueError(f"{pressure} hPa 不在 pressure_levels 中。")
    return pressure_levels.index(pressure)


def resize_2d_to_shape(arr, target_shape):
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"resize_2d_to_shape 只接受二维数组，当前 shape={arr.shape}")

    if arr.shape == target_shape:
        return arr

    try:
        from scipy.ndimage import zoom

        zoom_y = target_shape[0] / arr.shape[0]
        zoom_x = target_shape[1] / arr.shape[1]

        out = zoom(arr, zoom=(zoom_y, zoom_x), order=1)

        if out.shape != target_shape:
            fixed = np.full(target_shape, np.nan, dtype=out.dtype)
            h = min(out.shape[0], target_shape[0])
            w = min(out.shape[1], target_shape[1])
            fixed[:h, :w] = out[:h, :w]
            out = fixed

        return out

    except Exception:
        y_idx = np.linspace(0, arr.shape[0] - 1, target_shape[0]).astype(int)
        x_idx = np.linspace(0, arr.shape[1] - 1, target_shape[1]).astype(int)
        return arr[np.ix_(y_idx, x_idx)]


def prepare_bt_for_overlay(agri_bt108, target_shape):
    bt = agri_bt108.copy()

    if AGRI_mask.shape == bt.shape:
        bt[AGRI_mask == 1] = np.nan

    bt = resize_2d_to_shape(bt, target_shape)
    return bt


def format_time_label(time_str):
    dt = datetime.strptime(time_str, "%Y%m%d%H%M%S")
    return dt.strftime("%H:%M")


def add_bt_contours(ax, bt, add_labels=True):
    bt_valid = bt[np.isfinite(bt)]

    if bt_valid.size == 0:
        return None

    bt_min = np.nanmin(bt_valid)
    bt_max = np.nanmax(bt_valid)

    valid_levels = []
    valid_colors = []
    valid_lw = []

    for lev, col, lw in zip(
        BT_CONTOUR_LEVELS,
        BT_CONTOUR_COLORS,
        BT_CONTOUR_LINEWIDTHS
    ):
        if bt_min <= lev <= bt_max:
            valid_levels.append(lev)
            valid_colors.append(col)
            valid_lw.append(lw)

    if len(valid_levels) == 0:
        return None

    cs = ax.contour(
        bt,
        levels=valid_levels,
        colors=valid_colors,
        linewidths=valid_lw
    )

    if add_labels:
        labels = ax.clabel(
            cs,
            fmt="%d K",
            fontsize=6,
            inline=True,
            inline_spacing=2
        )

        for txt in labels:
            txt.set_fontfamily("Times New Roman")

    return cs


def make_quiver_grid(u, v, stride):
    h, w = u.shape
    yy, xx = np.mgrid[0:h, 0:w]

    xx_sub = xx[::stride, ::stride]
    yy_sub = yy[::stride, ::stride]
    u_sub = u[::stride, ::stride]
    v_sub = v[::stride, ::stride]

    valid = np.isfinite(u_sub) & np.isfinite(v_sub)

    return xx_sub, yy_sub, u_sub, v_sub, valid


def get_all_case_times():
    all_times = []
    for block in REPRESENTATIVE_BLOCKS:
        all_times.extend(block)
    return sorted(list(set(all_times)))


# ============================================================
# 读取归一化参数
# ============================================================
with np.load(norm_param_path) as s:
    agri_curr_mean = s["AGRI_curr_mean"].astype(np.float32)
    agri_curr_std = s["AGRI_curr_std"].astype(np.float32)
    agri_prev_mean = s["AGRI_prev_mean"].astype(np.float32)
    agri_prev_std = s["AGRI_prev_std"].astype(np.float32)
    giirs_mean = s["GIIRS_mean"].astype(np.float32)
    giirs_std = s["GIIRS_std"].astype(np.float32)

    era5_u_mean = s["ERA5_u_mean"].astype(np.float32)
    era5_u_std = s["ERA5_u_std"].astype(np.float32)
    era5_v_mean = s["ERA5_v_mean"].astype(np.float32)
    era5_v_std = s["ERA5_v_std"].astype(np.float32)

agri_curr_std = np.where(agri_curr_std == 0, 1e-6, agri_curr_std)
agri_prev_std = np.where(agri_prev_std == 0, 1e-6, agri_prev_std)
giirs_std = np.where(giirs_std == 0, 1e-6, giirs_std)
era5_u_std = np.where(era5_u_std == 0, 1e-6, era5_u_std)
era5_v_std = np.where(era5_v_std == 0, 1e-6, era5_v_std)


# ============================================================
# 加载模型
# ============================================================
model = U_Net(input_dim_AGRI, input_dim_GIIRS, output_dim).to(device)
checkpoint = torch.load(ckpt_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]
    new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()


# ============================================================
# 数据读取与预测
# ============================================================
def find_input_npz(time_str):
    npz_path = os.path.join(data_dir, f"{time_str}.npz")

    if os.path.exists(npz_path):
        return npz_path

    candidates = list(Path(data_dir).glob(f"{time_str}*.npz"))

    if len(candidates) == 0:
        raise FileNotFoundError(f"没有找到时次 {time_str} 对应的输入 npz 文件。")

    return str(candidates[0])


def load_one_npz(npz_path, return_agri_raw=False):
    with np.load(npz_path, allow_pickle=True) as data:
        AGRI_curr = data["AGRI_curr"][:, :, 0:15].copy()
        AGRI_prev = data["AGRI_prev"][:, :, 0:15].copy()
        GIIRS_ref = normalize_giirs_ref(data["GIIRS_ref"])
        GIIRS_delta_time = data["GIIRS_delta_time"].copy()

    AGRI_curr_raw = AGRI_curr.copy()

    giirs_path = os.path.join(giirs_root, GIIRS_ref)

    if not os.path.exists(giirs_path):
        raise FileNotFoundError(f"GIIRS 文件不存在: {giirs_path}")

    with np.load(giirs_path, allow_pickle=True) as giirs_file:
        GIIRS = giirs_file["GIIRS"].copy()

    AGRI_curr[AGRI_mask == 1] = np.nan
    AGRI_prev[AGRI_mask == 1] = np.nan
    GIIRS[GIIRS_mask == 1] = np.nan
    GIIRS_delta_time[GIIRS_mask == 1] = np.nan

    AGRI_curr_n = zscore_channel_last(AGRI_curr, agri_curr_mean, agri_curr_std)
    AGRI_prev_n = zscore_channel_last(AGRI_prev, agri_prev_mean, agri_prev_std)
    GIIRS_n = zscore_channel_last(GIIRS, giirs_mean, giirs_std)

    GIIRS_delta_time_max = 88.0
    GIIRS_delta_time_min = -99.0
    GIIRS_delta_time_n = (
        GIIRS_delta_time - GIIRS_delta_time_min
    ) / (GIIRS_delta_time_max - GIIRS_delta_time_min)

    AGRI_curr_n[AGRI_mask == 1] = 0
    AGRI_prev_n[AGRI_mask == 1] = 0
    GIIRS_n[GIIRS_mask == 1] = 0
    GIIRS_delta_time_n[GIIRS_mask == 1] = 0

    AGRI_curr_n = np.moveaxis(AGRI_curr_n, -1, 0)
    AGRI_prev_n = np.moveaxis(AGRI_prev_n, -1, 0)
    GIIRS_n = np.moveaxis(GIIRS_n, -1, 0)
    GIIRS_delta_time_n = np.expand_dims(GIIRS_delta_time_n, axis=0)

    AGRI_curr_n = torch.tensor(
        AGRI_curr_n, dtype=torch.float32
    ).unsqueeze(0).to(device)

    AGRI_prev_n = torch.tensor(
        AGRI_prev_n, dtype=torch.float32
    ).unsqueeze(0).to(device)

    GIIRS_n = torch.tensor(
        GIIRS_n, dtype=torch.float32
    ).unsqueeze(0).to(device)

    GIIRS_delta_time_n = torch.tensor(
        GIIRS_delta_time_n, dtype=torch.float32
    ).unsqueeze(0).to(device)

    if return_agri_raw:
        return (
            AGRI_curr_n,
            AGRI_prev_n,
            GIIRS_n,
            GIIRS_delta_time_n,
            AGRI_curr_raw,
            GIIRS_ref
        )

    return AGRI_curr_n, AGRI_prev_n, GIIRS_n, GIIRS_delta_time_n


def predict_or_load(time_str):
    pred_npz_path = os.path.join(save_dir, f"{time_str}_pred.npz")

    if os.path.exists(pred_npz_path) and not FORCE_REMAKE_NPZ:
        with np.load(pred_npz_path, allow_pickle=True) as data:
            pred_u = data["pred_u"].copy()
            pred_v = data["pred_v"].copy()

            if "agri_bt108" in data:
                agri_bt108 = data["agri_bt108"].copy()
            else:
                input_npz_path = find_input_npz(time_str)
                _, _, _, _, AGRI_curr_raw, _ = load_one_npz(
                    input_npz_path,
                    return_agri_raw=True
                )
                agri_bt108 = AGRI_curr_raw[:, :, AGRI_BT108_IDX].copy()

        return pred_u, pred_v, agri_bt108

    input_npz_path = find_input_npz(time_str)

    (
        AGRI_curr_n,
        AGRI_prev_n,
        GIIRS_n,
        GIIRS_delta_time_n,
        AGRI_curr_raw,
        giirs_ref
    ) = load_one_npz(input_npz_path, return_agri_raw=True)

    with torch.no_grad():
        pred_u_n, pred_v_n = model(
            AGRI_curr_n,
            AGRI_prev_n,
            GIIRS_n,
            GIIRS_delta_time_n
        )

    pred_u = inverse_norm(pred_u_n, era5_u_mean, era5_u_std)
    pred_v = inverse_norm(pred_v_n, era5_v_mean, era5_v_std)

    mask_3d = np.broadcast_to(GIIRS_mask[None, :, :], pred_u.shape)
    pred_u[mask_3d == 1] = np.nan
    pred_v[mask_3d == 1] = np.nan

    agri_bt108 = AGRI_curr_raw[:, :, AGRI_BT108_IDX].copy()

    if AGRI_mask.shape == agri_bt108.shape:
        agri_bt108[AGRI_mask == 1] = np.nan

    np.savez_compressed(
        pred_npz_path,
        pred_u=pred_u,
        pred_v=pred_v,
        agri_bt108=agri_bt108,
        giirs_ref=giirs_ref
    )

    return pred_u, pred_v, agri_bt108


# ============================================================
# 单张图输出
# ============================================================
def save_bt_individual(agri_bt108, save_png_path, time_str):
    bt = agri_bt108.copy()

    if AGRI_mask.shape == bt.shape:
        bt[AGRI_mask == 1] = np.nan

    fig = plt.figure(figsize=(5.0, 4.0), dpi=FIG_DPI)
    ax = fig.add_axes([0.06, 0.08, 0.78, 0.82])
    cax = fig.add_axes([0.87, 0.12, 0.035, 0.74])

    im = ax.imshow(
        bt,
        cmap=BT_CMAP,
        vmin=BT_VMIN,
        vmax=BT_VMAX,
        interpolation="nearest"
    )

    add_bt_contours(ax, bt, add_labels=True)

    ax.set_title(
        f"AGRI 10.8 μm BT  {format_time_label(time_str)} UTC",
        fontsize=11,
        fontfamily="Times New Roman"
    )
    ax.axis("off")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Brightness temperature (K)", fontsize=9, fontfamily="Times New Roman")
    cb.ax.tick_params(labelsize=8)

    for label in cb.ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    plt.savefig(save_png_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_wind_individual(pred_u, pred_v, agri_bt108, save_png_path, time_str):
    level_idx = get_level_index(CASE_PRESSURE)

    u = pred_u[level_idx]
    v = pred_v[level_idx]
    ws = np.sqrt(u ** 2 + v ** 2)

    bt = prepare_bt_for_overlay(agri_bt108, u.shape)

    xx_sub, yy_sub, u_sub, v_sub, valid = make_quiver_grid(
        u,
        v,
        WIND_VECTOR_STRIDE
    )

    fig = plt.figure(figsize=(5.0, 4.0), dpi=FIG_DPI)
    ax = fig.add_axes([0.06, 0.08, 0.78, 0.82])
    cax = fig.add_axes([0.87, 0.12, 0.035, 0.74])

    im = ax.imshow(
        ws,
        cmap=WIND_CMAP,
        vmin=WS_VMIN,
        vmax=WS_VMAX,
        interpolation="nearest"
    )

    add_bt_contours(ax, bt, add_labels=True)

    ax.quiver(
        xx_sub[valid],
        yy_sub[valid],
        u_sub[valid],
        -v_sub[valid],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=QUIVER_WIDTH,
        color=QUIVER_COLOR,
        alpha=QUIVER_ALPHA
    )

    ax.set_xlim(0, ws.shape[1] - 1)
    ax.set_ylim(ws.shape[0] - 1, 0)

    ax.set_title(
        f"{CASE_PRESSURE} hPa wind  {format_time_label(time_str)} UTC",
        fontsize=11,
        fontfamily="Times New Roman"
    )
    ax.axis("off")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Horizontal wind speed (m s$^{-1}$)", fontsize=9, fontfamily="Times New Roman")
    cb.ax.tick_params(labelsize=8)

    for label in cb.ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    plt.savefig(save_png_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 4 行组合图输出
# ============================================================
def save_composite_figure(case_data, save_png_path):
    n_block = len(REPRESENTATIVE_BLOCKS)

    if n_block != 2:
        raise ValueError("REPRESENTATIVE_BLOCKS 需要包含两个小时段。")

    ncol = len(REPRESENTATIVE_BLOCKS[0])

    for block in REPRESENTATIVE_BLOCKS:
        if len(block) != ncol:
            raise ValueError("两个小时段的时次数量必须一致。")

    nrow = 4

    fig_width = PANEL_FIGSIZE_PER_COL * ncol + 0.8

    # 行间距压缩版本
    fig_height = PANEL_FIGSIZE_PER_ROW * nrow + 0.90

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(fig_width, fig_height),
        dpi=FIG_DPI,
        constrained_layout=False
    )

    plt.subplots_adjust(
        left=0.045,
        right=0.91,
        bottom=0.045,
        top=0.925,
        wspace=0.03,
        hspace=0.16
    )

    bt_im = None
    wind_im = None

    row_labels = [
        "AGRI 10.8 μm BT",
        f"{CASE_PRESSURE} hPa wind",
        "AGRI 10.8 μm BT",
        f"{CASE_PRESSURE} hPa wind",
    ]

    for block_idx, block_times in enumerate(REPRESENTATIVE_BLOCKS):
        bt_row = block_idx * 2
        wind_row = bt_row + 1

        for col, time_str in enumerate(block_times):
            pred_u = case_data[time_str]["pred_u"]
            pred_v = case_data[time_str]["pred_v"]
            agri_bt108 = case_data[time_str]["agri_bt108"]

            level_idx = get_level_index(CASE_PRESSURE)
            u = pred_u[level_idx]
            v = pred_v[level_idx]
            ws = np.sqrt(u ** 2 + v ** 2)

            bt_for_bt_panel = agri_bt108.copy()

            if AGRI_mask.shape == bt_for_bt_panel.shape:
                bt_for_bt_panel[AGRI_mask == 1] = np.nan

            bt_for_wind_panel = prepare_bt_for_overlay(agri_bt108, ws.shape)

            ax_bt = axes[bt_row, col]
            ax_wind = axes[wind_row, col]

            bt_im = ax_bt.imshow(
                bt_for_bt_panel,
                cmap=BT_CMAP,
                vmin=BT_VMIN,
                vmax=BT_VMAX,
                interpolation="nearest"
            )
            add_bt_contours(ax_bt, bt_for_bt_panel, add_labels=False)

            wind_im = ax_wind.imshow(
                ws,
                cmap=WIND_CMAP,
                vmin=WS_VMIN,
                vmax=WS_VMAX,
                interpolation="nearest"
            )
            add_bt_contours(ax_wind, bt_for_wind_panel, add_labels=False)

            xx_sub, yy_sub, u_sub, v_sub, valid = make_quiver_grid(
                u,
                v,
                WIND_VECTOR_STRIDE
            )

            ax_wind.quiver(
                xx_sub[valid],
                yy_sub[valid],
                u_sub[valid],
                -v_sub[valid],
                angles="xy",
                scale_units="xy",
                scale=None,
                width=QUIVER_WIDTH,
                color=QUIVER_COLOR,
                alpha=QUIVER_ALPHA
            )

            ax_bt.set_title(
                f"{format_time_label(time_str)} UTC",
                fontsize=10,
                fontfamily="Times New Roman",
                pad=4
            )

            ax_bt.axis("off")
            ax_wind.axis("off")

            ax_wind.set_xlim(0, ws.shape[1] - 1)
            ax_wind.set_ylim(ws.shape[0] - 1, 0)

            if col == 0:
                ax_bt.text(
                    -0.035,
                    0.5,
                    row_labels[bt_row],
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax_bt.transAxes,
                    fontsize=10,
                    fontfamily="Times New Roman"
                )
                ax_wind.text(
                    -0.035,
                    0.5,
                    row_labels[wind_row],
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=ax_wind.transAxes,
                    fontsize=10,
                    fontfamily="Times New Roman"
                )

    cax_bt = fig.add_axes([0.925, 0.565, 0.012, 0.305])
    cax_wind = fig.add_axes([0.925, 0.145, 0.012, 0.305])

    cb_bt = fig.colorbar(bt_im, cax=cax_bt)
    cb_bt.set_label("BT (K)", fontsize=9, fontfamily="Times New Roman")
    cb_bt.ax.tick_params(labelsize=8)

    for label in cb_bt.ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    cb_wind = fig.colorbar(wind_im, cax=cax_wind)
    cb_wind.set_label("Wind speed (m s$^{-1}$)", fontsize=9, fontfamily="Times New Roman")
    cb_wind.ax.tick_params(labelsize=8)

    for label in cb_wind.ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    fig.suptitle(
        "Fifteen-minute evolution of AGRI cloud-top temperature and retrieved 850 hPa wind field",
        fontsize=12,
        fontfamily="Times New Roman",
        y=0.965
    )

    plt.savefig(save_png_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main():
    case_data = {}

    for time_str in get_all_case_times():
        pred_u, pred_v, agri_bt108 = predict_or_load(time_str)

        case_data[time_str] = {
            "pred_u": pred_u,
            "pred_v": pred_v,
            "agri_bt108": agri_bt108,
        }

        bt_png_path = os.path.join(
            bt_dir,
            f"{time_str}_AGRI_BT108_top_style.png"
        )
        save_bt_individual(
            agri_bt108=agri_bt108,
            save_png_path=bt_png_path,
            time_str=time_str
        )

        wind_png_path = os.path.join(
            wind_dir,
            f"{time_str}_{CASE_PRESSURE}hPa_wind_with_bt_contours_top_style.png"
        )
        save_wind_individual(
            pred_u=pred_u,
            pred_v=pred_v,
            agri_bt108=agri_bt108,
            save_png_path=wind_png_path,
            time_str=time_str
        )

    composite_png_path = os.path.join(
        composite_dir,
        f"AGRI_BT_and_{CASE_PRESSURE}hPa_wind_4row_composite_top_style.png"
    )
    save_composite_figure(case_data, composite_png_path)

    composite_pdf_path = os.path.join(
        composite_dir,
        f"AGRI_BT_and_{CASE_PRESSURE}hPa_wind_4row_composite_top_style.pdf"
    )
    save_composite_figure(case_data, composite_pdf_path)

    print("done.")
    print("individual BT:", bt_dir)
    print("individual wind:", wind_dir)
    print("composite:", composite_dir)
    print("main figure:", composite_pdf_path)


if __name__ == "__main__":
    main()