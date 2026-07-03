import numpy as np
import os

file_path = r'/media/ub/Extreme SSD/npz_file/with_era5_uvw/train/20250901010000.npz'
# 读取AGRI数据
data = np.load(file_path, allow_pickle=True)
AGRI_curr = data['AGRI_curr']
AGRI_prev = data['AGRI_prev']
GIIRS_delta_time = data['GIIRS_delta_time']
ERA5 = data['ERA5']

agri_ch = AGRI_curr[:, :, 3]
v = agri_ch[6, 629]
print(repr(v))
print(type(v))
print(agri_ch.dtype)
AGRI_mask = ((agri_ch == -999) | np.isnan(agri_ch)).astype(np.uint8)
GIIRS_mask = (ERA5[:,:,0,0] == -999).astype(np.uint8)
save_dir = '../data/auxiliary_data'
np.save(os.path.join(save_dir, 'AGRI_mask.npy'), AGRI_mask)
np.save(os.path.join(save_dir, 'GIIRS_mask.npy'), GIIRS_mask)
i=1