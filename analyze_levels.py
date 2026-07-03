
import os, sys, csv, math, time
import numpy as np
import torch

# ====== 路径定义 (在此统一修改) ======
PROJECT_ROOT = "/3D wind field/fusion_3D_wind_uv"
UNET_DIR     = os.path.join(PROJECT_ROOT, "unet_model")
CKPT_PATH    = os.path.join(UNET_DIR, "checkpoint", "unet_best.pth")
OUT_CSV      = os.path.join(UNET_DIR, "checkpoint", "test_level_metrics_37levels.csv")
FIG_PATH     = os.path.join(UNET_DIR, "fig_level_metrics.png")

# 添加 unet_model 目录到 Python 路径，以便 import model / dataset
sys.path.insert(0, UNET_DIR)

from model import U_Net
from dataset import UDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}", flush=True)
print(f"PROJECT_ROOT: {PROJECT_ROOT}", flush=True)
print(f"UNET_DIR:     {UNET_DIR}", flush=True)

input_dim_AGRI, input_dim_GIIRS, output_dim = 15, 1690, 37

pressure_levels = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
    250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
    20, 10, 7, 5, 3, 2, 1
]

# ====== 加载模型 ======
model = U_Net(input_dim_AGRI, input_dim_GIIRS, output_dim).to(device)
checkpoint = torch.load(CKPT_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()
print(f"模型加载完成: {CKPT_PATH}", flush=True)

# ====== 数据集（需要 cd 到 unet_model 目录让 dataset.py 找到相对路径的 aux data）=====
cwd_backup = os.getcwd()
os.chdir(UNET_DIR)
test_dataset = UDataset(r"/home/ub/data/npz_file/with_era5_uvw/test")
os.chdir(cwd_backup)  # 切回来
total = len(test_dataset)
print(f"测试集样本数: {total}", flush=True)

# ====== 统计累积器 (37层) ======
level_u_se = torch.zeros(output_dim, device=device)
level_v_se = torch.zeros(output_dim, device=device)
level_u_ae = torch.zeros(output_dim, device=device)
level_v_ae = torch.zeros(output_dim, device=device)
level_u_be = torch.zeros(output_dim, device=device)
level_v_be = torch.zeros(output_dim, device=device)
level_count = torch.zeros(output_dim, device=device)

# ====== 推理循环 ======
n_processed = 0
t_start = time.time()

for idx in range(total):
    AGRI_curr_n, AGRI_prev_n, GIIRS_n, GIIRS_delta_time_n, ERA5_u_n, ERA5_v_n = test_dataset[idx]

    AGRI_curr_n = AGRI_curr_n.unsqueeze(0).to(device)
    AGRI_prev_n = AGRI_prev_n.unsqueeze(0).to(device)
    GIIRS_n = GIIRS_n.unsqueeze(0).to(device)
    GIIRS_delta_time_n = GIIRS_delta_time_n.unsqueeze(0).to(device)
    ERA5_u_n = ERA5_u_n.unsqueeze(0).to(device)
    ERA5_v_n = ERA5_v_n.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_u, pred_v = model(AGRI_curr_n, AGRI_prev_n, GIIRS_n, GIIRS_delta_time_n)

    diff_u = pred_u - ERA5_u_n
    diff_v = pred_v - ERA5_v_n

    level_u_se += (diff_u ** 2).sum(dim=(0, 2, 3))
    level_v_se += (diff_v ** 2).sum(dim=(0, 2, 3))
    level_u_ae += diff_u.abs().sum(dim=(0, 2, 3))
    level_v_ae += diff_v.abs().sum(dim=(0, 2, 3))
    level_u_be += diff_u.sum(dim=(0, 2, 3))
    level_v_be += diff_v.sum(dim=(0, 2, 3))
    level_count += ERA5_u_n.shape[2] * ERA5_u_n.shape[3]

    n_processed += 1
    if n_processed % 100 == 0:
        elapsed = time.time() - t_start
        rate = n_processed / elapsed
        remain = (total - n_processed) / rate
        print(f"  进度: {n_processed}/{total} ({n_processed/total*100:.0f}%), "
              f"已用 {elapsed:.0f}s, 预计剩余 {remain:.0f}s", flush=True)

# ====== 计算指标 ======
cnt = level_count.clamp(min=1e-12)
u_rmse = torch.sqrt(level_u_se / cnt)
v_rmse = torch.sqrt(level_v_se / cnt)
u_mae  = level_u_ae / cnt
v_mae  = level_v_ae / cnt
u_bias = level_u_be / cnt
v_bias = level_v_be / cnt

# ====== 保存 CSV ======
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
header = ["pressure_hPa", "u_rmse", "v_rmse", "u_mae", "v_mae", "u_bias", "v_bias"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for i, p in enumerate(pressure_levels):
        w.writerow([p,
            f"{u_rmse[i].item():.6f}", f"{v_rmse[i].item():.6f}",
            f"{u_mae[i].item():.6f}", f"{v_mae[i].item():.6f}",
            f"{u_bias[i].item():.6f}", f"{v_bias[i].item():.6f}"])
print(f"\n37层指标已保存: {OUT_CSV}", flush=True)

# ====== 终端打印表格 ======
print("\n" + "=" * 110, flush=True)
print(f"{'气压层':>8} | {'u_RMSE':>9} {'v_RMSE':>9} {'u_MAE':>9} {'v_MAE':>9} {'u_Bias':>9} {'v_Bias':>9} | {'u+v RMS':>9}")
print("-" * 110, flush=True)
for i, p in enumerate(pressure_levels):
    ur, vr = u_rmse[i].item(), v_rmse[i].item()
    ua, va = u_mae[i].item(), v_mae[i].item()
    ub, vb = u_bias[i].item(), v_bias[i].item()
    uv_rmse = math.sqrt(ur**2 + vr**2)
    print(f"{p:>6}hPa | {ur:>9.4f} {vr:>9.4f} {ua:>9.4f} {va:>9.4f} {ub:>9.4f} {vb:>9.4f} | {uv_rmse:>9.4f}", flush=True)
print("-" * 110, flush=True)

# ====== 汇总统计 ======
print(f"\n=== 汇总 (37层) ===", flush=True)
print(f"平均 u_RMSE: {u_rmse.mean().item():.4f}", flush=True)
print(f"平均 v_RMSE: {v_rmse.mean().item():.4f}", flush=True)

u_best_lv = pressure_levels[u_rmse.argmin().item()]
u_worst_lv = pressure_levels[u_rmse.argmax().item()]
v_best_lv = pressure_levels[v_rmse.argmin().item()]
v_worst_lv = pressure_levels[v_rmse.argmax().item()]
print(f"u_RMSE 最优层: {u_best_lv}hPa ({u_rmse.min().item():.4f})", flush=True)
print(f"u_RMSE 最差层: {u_worst_lv}hPa ({u_rmse.max().item():.4f})", flush=True)
print(f"v_RMSE 最优层: {v_best_lv}hPa ({v_rmse.min().item():.4f})", flush=True)
print(f"v_RMSE 最差层: {v_worst_lv}hPa ({v_rmse.max().item():.4f})", flush=True)

print(f"\n=== 垂直分层 RMSE ===", flush=True)
groups = [("底层 1000-700hPa", 1000, 700), ("中层 700-300hPa", 700, 300),
          ("高层 300-100hPa", 300, 100), ("平流层 <100hPa", 100, 0)]
for name, hi, lo in groups:
    mask = [(l <= hi) and (l > lo) for l in pressure_levels]
    print(f"  u {name}: {u_rmse[mask].mean().item():.4f}", flush=True)
    print(f"  v {name}: {v_rmse[mask].mean().item():.4f}", flush=True)

# ====== 绘图 ======
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    levels_np = np.array(pressure_levels)

    plots = [
        (axes[0], [u_rmse.cpu().numpy(), v_rmse.cpu().numpy()],
         ["u RMSE", "v RMSE"], "RMSE by Pressure Level", None),
        (axes[1], [u_mae.cpu().numpy(), v_mae.cpu().numpy()],
         ["u MAE", "v MAE"], "MAE by Pressure Level", None),
        (axes[2], [u_bias.cpu().numpy(), v_bias.cpu().numpy()],
         ["u Bias", "v Bias"], "Bias by Pressure Level", 0),
    ]
    for ax, vals, labels, title, vline in plots:
        ax.plot(vals[0], levels_np, "b-o", label=labels[0], markersize=4)
        ax.plot(vals[1], levels_np, "r-s", label=labels[1], markersize=4)
        ax.set_ylim(1050, -50)
        ax.set_xlabel("Value")
        ax.set_ylabel("Pressure (hPa)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if vline is not None:
            ax.axvline(vline, color="gray", linestyle="--", alpha=0.5)

    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n剖面图已保存: {FIG_PATH}", flush=True)
except Exception as e:
    print(f"\n绘图失败 (可忽略): {e}", flush=True)

elapsed = time.time() - t_start
print(f"\n总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
print("完成!", flush=True)
