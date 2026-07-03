import numpy as np
import os

AGRI_mask = np.load('../data/auxiliary_data/AGRI_mask.npy')
GIIRS_mask = np.load('../data/auxiliary_data/GIIRS_mask.npy')
folder_path = r'../data/with_era5_uvw/train'
giirs_folder_path = r'../data/GIIRS_store'
class OnlineNanStatsVec:
    def __init__(self, d):
        self.n = np.zeros(d, dtype=np.int64)
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros(d, dtype=np.float64)

    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).astype(np.float64, copy=False)
        mask = ~np.isnan(x)
        n2 = mask.sum(axis=0).astype(np.int64)

        valid = n2 > 0
        if not np.any(valid):
            return

        s2 = np.where(valid, np.nansum(x, axis=0), 0.0)
        ss2 = np.where(valid, np.nansum(x * x, axis=0), 0.0)

        mean2 = np.zeros_like(self.mean)
        mean2[valid] = s2[valid] / n2[valid]
        M2_2 = np.zeros_like(self.M2)
        M2_2[valid] = ss2[valid] - n2[valid] * mean2[valid] * mean2[valid]

        n1 = self.n.copy()
        mean1 = self.mean.copy()
        M2_1 = self.M2.copy()

        only_new = (n1 == 0) & valid
        self.n[only_new] = n2[only_new]
        self.mean[only_new] = mean2[only_new]
        self.M2[only_new] = M2_2[only_new]

        both = (n1 > 0) & valid
        if np.any(both):
            delta = mean2[both] - mean1[both]
            n = n1[both] + n2[both]
            self.mean[both] = mean1[both] + delta * (n2[both] / n)
            self.M2[both] = M2_1[both] + M2_2[both] + delta * delta * (n1[both] * n2[both] / n)
            self.n[both] = n

    def finalize(self):
        mean = self.mean.astype(np.float64, copy=True)
        std = np.full_like(mean, np.nan, dtype=np.float64)
        ok = self.n >= 2
        std[ok] = np.sqrt(self.M2[ok] / self.n[ok])
        mean[~ok] = np.nan
        return mean, std

stats = {
    'AGRI_curr': OnlineNanStatsVec(15),
    'AGRI_prev': OnlineNanStatsVec(15),
    'GIIRS': OnlineNanStatsVec(1690),
    'ERA5_u': OnlineNanStatsVec(37),
    'ERA5_v': OnlineNanStatsVec(37),
    'ERA5_w': OnlineNanStatsVec(37),
}
# ---------- 第1遍：算 AGRI + ERA5（不读GIIRS） ----------
for filename in os.listdir(folder_path):
    if not filename.endswith('.npz'):
        continue
    file_path = os.path.join(folder_path, filename)
    print(filename)

    data = np.load(file_path)
    AGRI_curr = data['AGRI_curr'].astype(np.float32)
    AGRI_prev = data['AGRI_prev'].astype(np.float32)
    ERA5_u = data['ERA5'][:, :, :, 0].astype(np.float32)
    ERA5_v = data['ERA5'][:, :, :, 1].astype(np.float32)
    ERA5_w = data['ERA5'][:, :, :, 2].astype(np.float32)

    AGRI_curr[AGRI_mask == 1] = np.nan
    AGRI_prev[AGRI_mask == 1] = np.nan
    ERA5_u[GIIRS_mask == 1] = np.nan
    ERA5_v[GIIRS_mask == 1] = np.nan
    ERA5_w[GIIRS_mask == 1] = np.nan
    stats['AGRI_curr'].update(AGRI_curr)
    stats['AGRI_prev'].update(AGRI_prev)
    stats['ERA5_u'].update(ERA5_u)
    stats['ERA5_v'].update(ERA5_v)
    stats['ERA5_w'].update(ERA5_w)

# ---------- 第2遍：单独算 GIIRS ----------
for filename in sorted(os.listdir(giirs_folder_path)):
    if not filename.endswith('.npz'):
        continue
    giirs_path = os.path.join(giirs_folder_path, filename)
    print('GIIRS:', filename)

    g = np.load(giirs_path)
    GIIRS = g['GIIRS'].astype(np.float32)   # 如果你的key不是'GIIRS'，改这里
    GIIRS[GIIRS_mask == 1] = np.nan
    stats['GIIRS'].update(GIIRS)

for k, s in stats.items():
    mean, std = s.finalize()
    print(f'\n{k}: shape={mean.shape}')
    for i in range(mean.size):
        print(i, float(mean[i]), float(std[i]) if np.isfinite(std[i]) else np.nan)

out_path = r'../data/auxiliary_data/normalized_parameters_mean_std_uvw.npz'

np.savez(
    out_path,
    AGRI_curr_mean=stats['AGRI_curr'].finalize()[0],
    AGRI_curr_std=stats['AGRI_curr'].finalize()[1],
    AGRI_prev_mean=stats['AGRI_prev'].finalize()[0],
    AGRI_prev_std=stats['AGRI_prev'].finalize()[1],
    GIIRS_mean=stats['GIIRS'].finalize()[0],
    GIIRS_std=stats['GIIRS'].finalize()[1],
    ERA5_u_mean=stats['ERA5_u'].finalize()[0],
    ERA5_u_std=stats['ERA5_u'].finalize()[1],
    ERA5_v_mean=stats['ERA5_v'].finalize()[0],
    ERA5_v_std=stats['ERA5_v'].finalize()[1],
    ERA5_w_mean=stats['ERA5_w'].finalize()[0],
    ERA5_w_std=stats['ERA5_w'].finalize()[1],
)

print('Saved to:', out_path)
