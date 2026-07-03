import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import U_Net

# ====== 配置 ======
device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim_AGRI = 15
input_dim_GIIRS = 1690
output_dim = 37

ckpt_path = "../unet_model/checkpoint/unet_best.pth"
data_dir = "/home/ub/yjy/3D wind field/fusion_3D_wind_uv/15min_wind/data"
save_dir = "/home/ub/yjy/3D wind field/fusion_3D_wind_uv/15min_wind/15min_save_path"
img_dir = "/home/ub/yjy/3D wind field/fusion_3D_wind_uv/15min_wind/fig/all_data_images_uv"
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

os.makedirs(save_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)


def zscore_channel_last(x, mean, std, eps=1e-6):
    mean = mean.reshape((1,) * (x.ndim - 1) + (-1,))
    std = std.reshape((1,) * (x.ndim - 1) + (-1,))
    std_safe = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    mean_safe = np.where(np.isfinite(mean), mean, 0.0)
    out = (x - mean_safe) / std_safe
    out = np.where(np.isnan(x), np.nan, out)
    return out


# ====== 读取归一化参数 ======
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
    era5_w_mean = s["ERA5_w_mean"].astype(np.float32)
    era5_w_std = s["ERA5_w_std"].astype(np.float32)

agri_curr_std = np.where(agri_curr_std == 0, 1e-6, agri_curr_std)
agri_prev_std = np.where(agri_prev_std == 0, 1e-6, agri_prev_std)
giirs_std = np.where(giirs_std == 0, 1e-6, giirs_std)
era5_u_std = np.where(era5_u_std == 0, 1e-6, era5_u_std)
era5_v_std = np.where(era5_v_std == 0, 1e-6, era5_v_std)


# ====== 加载模型 ======
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


def load_one_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        AGRI_curr = data["AGRI_curr"][:,:,0:15].copy()
        AGRI_prev = data["AGRI_prev"][:,:,0:15].copy()
        GIIRS_ref = data["GIIRS_ref"]
        GIIRS_delta_time = data["GIIRS_delta_time"].copy()

    giirs_filename = os.path.basename(str(GIIRS_ref))
    giirs_path = os.path.join(giirs_root, giirs_filename)
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
    GIIRS_delta_time_n = (GIIRS_delta_time - GIIRS_delta_time_min) / (GIIRS_delta_time_max - GIIRS_delta_time_min)

    AGRI_curr_n[AGRI_mask == 1] = 0
    AGRI_prev_n[AGRI_mask == 1] = 0
    GIIRS_n[GIIRS_mask == 1] = 0
    GIIRS_delta_time_n[GIIRS_mask == 1] = 0

    AGRI_curr_n = np.moveaxis(AGRI_curr_n, -1, 0)
    AGRI_prev_n = np.moveaxis(AGRI_prev_n, -1, 0)
    GIIRS_n = np.moveaxis(GIIRS_n, -1, 0)
    GIIRS_delta_time_n = np.expand_dims(GIIRS_delta_time_n, axis=0)

    AGRI_curr_n = torch.tensor(AGRI_curr_n, dtype=torch.float32).unsqueeze(0).to(device)
    AGRI_prev_n = torch.tensor(AGRI_prev_n, dtype=torch.float32).unsqueeze(0).to(device)
    GIIRS_n = torch.tensor(GIIRS_n, dtype=torch.float32).unsqueeze(0).to(device)
    GIIRS_delta_time_n = torch.tensor(GIIRS_delta_time_n, dtype=torch.float32).unsqueeze(0).to(device)

    return AGRI_curr_n, AGRI_prev_n, GIIRS_n, GIIRS_delta_time_n


def inverse_norm(pred, mean, std):
    pred = pred[0].detach().cpu().numpy()
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    return pred * std + mean


def save_uv_wind_image(pred_u, pred_v, save_png_path, level, time_str,
                       stride=4, vmin=None, vmax=None, cmap="viridis"):
    u = pred_u[level]
    v = pred_v[level]
    ws = np.sqrt(u ** 2 + v ** 2)

    pressure = pressure_levels[level]

    ws_max = np.nanmax(ws)
    ws_min = np.nanmin(ws)

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
    ax.set_title(
        f"{time_str}  {pressure} hPa\n"
        f"WS max={ws_max:.2f} m/s, min={ws_min:.2f} m/s",
        fontsize=12
    )
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

    w_max = np.nanmax(w)
    w_min = np.nanmin(w)

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
    ax.set_title(
        f"{time_str}  {pressure} hPa  w\n"
        f"w max={w_max:.4f} Pa/s, min={w_min:.4f} Pa/s",
        fontsize=12
    )
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
    cb.set_label("Vertical Velocity (hPa/s)")

    plt.savefig(save_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ====== 为每个气压层创建文件夹 ======
for pressure in pressure_levels:
    os.makedirs(os.path.join(img_dir, "uv", f"{pressure}hPa"), exist_ok=True)


# ====== 批量预测文件列表 ======
file_list = sorted(
    [f for f in os.listdir(data_dir) if f.endswith(".npz")],
    key=lambda f: os.path.getmtime(os.path.join(data_dir, f))
)

print("data_dir absolute path:", os.path.abspath(data_dir))
print("save_dir absolute path:", os.path.abspath(save_dir))
print("img_dir absolute path:", os.path.abspath(img_dir))
print("file_list length:", len(file_list))

# ====== 第一遍：预测、保存 npz、收集统一色标样本 ======
ws_samples = []
w_samples = []

with torch.no_grad():
    for filename in file_list:
        save_npz_path = os.path.join(save_dir, Path(filename).stem + "_pred.npz")

        # ===== 已有预测结果就直接读取，跳过重新推理 =====
        if os.path.exists(save_npz_path):
            with np.load(save_npz_path) as data:
                pred_u = data["pred_u"]
                pred_v = data["pred_v"]

            print(f"skip existing npz: {save_npz_path}")

        else:
            npz_path = os.path.join(data_dir, filename)

            AGRI_curr_n, AGRI_prev_n, GIIRS_n, GIIRS_delta_time_n = load_one_npz(npz_path)

            pred_u_n, pred_v_n = model(
                AGRI_curr_n, AGRI_prev_n, GIIRS_n, GIIRS_delta_time_n
            )

            pred_u = inverse_norm(pred_u_n, era5_u_mean, era5_u_std)
            pred_v = inverse_norm(pred_v_n, era5_v_mean, era5_v_std)

            mask_3d = np.broadcast_to(GIIRS_mask[None, :, :], pred_u.shape)
            pred_u[mask_3d == 1] = np.nan
            pred_v[mask_3d == 1] = np.nan

            np.savez_compressed(
                save_npz_path,
                pred_u=pred_u,
                pred_v=pred_v,
            )

            print(f"saved npz: {save_npz_path}")

        # ===== 不管是新生成还是旧文件，都参与统一色标统计 =====
        ws = np.sqrt(pred_u ** 2 + pred_v ** 2)
        ws_valid = ws[np.isfinite(ws)]
        if ws_valid.size > 0:
            ws_samples.append(ws_valid[::100])


if len(ws_samples) == 0:
    raise ValueError("没有可用于统计水平风速分位数的有效数据。")


ws_samples = np.concatenate(ws_samples)

global_ws_min = np.percentile(ws_samples, 1)
global_ws_max = np.percentile(ws_samples, 99)


print(f"global ws percentile 1%   = {global_ws_min:.4f}")
print(f"global ws percentile 99%  = {global_ws_max:.4f}")

# ====== 第二遍：统一色标画图 ======
for filename in file_list:
    pred_npz_path = os.path.join(save_dir, Path(filename).stem + "_pred.npz")
    with np.load(pred_npz_path) as data:
        pred_u = data["pred_u"]
        pred_v = data["pred_v"]

    time_str = Path(filename).stem

    for level, pressure in enumerate(pressure_levels):
        uv_level_dir = os.path.join(img_dir, "uv", f"{pressure}hPa")

        uv_png_path = os.path.join(
            uv_level_dir,
            f"{time_str}_{pressure}hPa_uv.png"
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


        print(f"saved uv png: {uv_png_path}")