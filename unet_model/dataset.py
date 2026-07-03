from torch.utils.data import Dataset
import numpy as np
import torch
import os
import random


def zscore_channel_last(x, mean, std, eps=1e-6):
    """
    x: (..., C)
    mean/std: (C,)
    """
    mean = mean.reshape((1,) * (x.ndim - 1) + (-1,))
    std = std.reshape((1,) * (x.ndim - 1) + (-1,))

    std_safe = np.where(np.isfinite(std) & (std > eps), std, 1.0)
    mean_safe = np.where(np.isfinite(mean), mean, 0.0)

    out = (x - mean_safe) / std_safe
    out = np.where(np.isnan(x), np.nan, out)
    return out


class UDataset(Dataset):
    def __init__(self, npz_dir, max_retry=10):
        self.npz_dir = npz_dir
        self.max_retry = max_retry

        file_list = []
        for filename in os.listdir(npz_dir):
            if filename.endswith('.npz'):
                file_path = os.path.join(npz_dir, filename)
                file_list.append(file_path)

        self.file_list = file_list

        s = np.load('../data/auxiliary_data/normalized_parameters_mean_std_uvw.npz')
        self.agri_curr_mean = s["AGRI_curr_mean"].astype(np.float32)
        self.agri_curr_std = s["AGRI_curr_std"].astype(np.float32)
        self.agri_prev_mean = s["AGRI_prev_mean"].astype(np.float32)
        self.agri_prev_std = s["AGRI_prev_std"].astype(np.float32)

        self.giirs_mean = s["GIIRS_mean"].astype(np.float32)
        self.giirs_std = s["GIIRS_std"].astype(np.float32)

        self.era5_u_mean = s["ERA5_u_mean"].astype(np.float32)
        self.era5_u_std = s["ERA5_u_std"].astype(np.float32)
        self.era5_v_mean = s["ERA5_v_mean"].astype(np.float32)
        self.era5_v_std = s["ERA5_v_std"].astype(np.float32)

        self.AGRI_mask = np.load('../data/auxiliary_data/AGRI_mask.npy')
        self.GIIRS_mask = np.load('../data/auxiliary_data/GIIRS_mask.npy')

        self.giirs_root = '/home/ub/data/npz_file/GIIRS_store'

    def _load_one_sample(self, index):
        npy_data = np.load(self.file_list[index], allow_pickle=True)

        AGRI_curr = npy_data['AGRI_curr'][:,:,0:15].copy()
        AGRI_prev = npy_data['AGRI_prev'][:,:,0:15].copy()
        GIIRS_ref = npy_data['GIIRS_ref']
        GIIRS_delta_time = npy_data['GIIRS_delta_time'].copy()
        ERA5_u = npy_data['ERA5'][:, :, :, 0].copy()
        ERA5_v = npy_data['ERA5'][:, :, :, 1].copy()

        giirs_filename = os.path.basename(str(GIIRS_ref))
        giirs_path = os.path.join(self.giirs_root, giirs_filename)
        giirs_file = np.load(giirs_path, allow_pickle=True)
        GIIRS = giirs_file['GIIRS'].copy()

        AGRI_curr[self.AGRI_mask == 1] = np.nan
        AGRI_prev[self.AGRI_mask == 1] = np.nan
        GIIRS[self.GIIRS_mask == 1] = np.nan
        GIIRS_delta_time[self.GIIRS_mask == 1] = np.nan
        ERA5_u[self.GIIRS_mask == 1] = np.nan
        ERA5_v[self.GIIRS_mask == 1] = np.nan

        AGRI_curr_n = zscore_channel_last(AGRI_curr, self.agri_curr_mean, self.agri_curr_std)
        AGRI_prev_n = zscore_channel_last(AGRI_prev, self.agri_prev_mean, self.agri_prev_std)
        GIIRS_n = zscore_channel_last(GIIRS, self.giirs_mean, self.giirs_std)
        ERA5_u_n = zscore_channel_last(ERA5_u, self.era5_u_mean, self.era5_u_std)
        ERA5_v_n = zscore_channel_last(ERA5_v, self.era5_v_mean, self.era5_v_std)

        GIIRS_delta_time_max = 88
        GIIRS_delta_time_min = -99
        GIIRS_delta_time_n = (GIIRS_delta_time - GIIRS_delta_time_min) / (GIIRS_delta_time_max - GIIRS_delta_time_min)

        AGRI_curr_n[self.AGRI_mask == 1] = 0
        AGRI_prev_n[self.AGRI_mask == 1] = 0
        GIIRS_n[self.GIIRS_mask == 1] = 0
        GIIRS_delta_time_n[self.GIIRS_mask == 1] = 0
        ERA5_u_n[self.GIIRS_mask == 1] = 0
        ERA5_v_n[self.GIIRS_mask == 1] = 0

        AGRI_curr_n = np.moveaxis(AGRI_curr_n, -1, 0)
        AGRI_prev_n = np.moveaxis(AGRI_prev_n, -1, 0)
        GIIRS_n = np.moveaxis(GIIRS_n, -1, 0)
        GIIRS_delta_time_n = np.expand_dims(GIIRS_delta_time_n, axis=0)
        ERA5_u_n = np.moveaxis(ERA5_u_n, -1, 0)
        ERA5_v_n = np.moveaxis(ERA5_v_n, -1, 0)

        return (
            torch.tensor(AGRI_curr_n, dtype=torch.float32),
            torch.tensor(AGRI_prev_n, dtype=torch.float32),
            torch.tensor(GIIRS_n, dtype=torch.float32),
            torch.tensor(GIIRS_delta_time_n, dtype=torch.float32),
            torch.tensor(ERA5_u_n, dtype=torch.float32),
            torch.tensor(ERA5_v_n, dtype=torch.float32),
        )

    def __getitem__(self, index):
        last_error = None
        cur_index = index

        for _ in range(self.max_retry):
            try:
                return self._load_one_sample(cur_index)
            except Exception as e:
                last_error = e
                bad_file = self.file_list[cur_index]
                print(f"[WARN] 读取失败，index={cur_index}, file={bad_file}, error={repr(e)}")
                cur_index = random.randint(0, len(self.file_list) - 1)

        raise RuntimeError(
            f"连续重试 {self.max_retry} 次仍然读取失败，原始 index={index}, 最后一次错误={repr(last_error)}"
        )

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    # dataset = UDataset(r'/media/ub/Extreme SSD/npz_file/with_era5_uvw/train')
    dataset = UDataset(r'/home/ub/yjy/3D wind field/fusion_3D_wind/data/with_era5_uvw/train')
    AGRI_curr_n, AGRI_prev_n, GIIRS_n, GIIRS_delta_time_n, ERA5_u_n, ERA5_v_n, ERA5_v_w = dataset[0]
    i = 1
