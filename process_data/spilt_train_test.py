import os
import random
import shutil

data_dir = "../data/with_era5_uvw"
train_ratio = 0.8
seed = 42

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

all_files = [
    f for f in os.listdir(data_dir)
    if f.endswith(".npz") and os.path.isfile(os.path.join(data_dir, f))
]

random.seed(seed)
random.shuffle(all_files)

train_size = int(len(all_files) * train_ratio)
train_files = all_files[:train_size]
test_files = all_files[train_size:]

for file_name in train_files:
    src = os.path.join(data_dir, file_name)
    dst = os.path.join(train_dir, file_name)
    shutil.move(src, dst)

for file_name in test_files:
    src = os.path.join(data_dir, file_name)
    dst = os.path.join(test_dir, file_name)
    shutil.move(src, dst)

print(f"总文件数: {len(all_files)}")
print(f"训练集数量: {len(train_files)}")
print(f"测试集数量: {len(test_files)}")
print(f"训练集目录: {train_dir}")
print(f"测试集目录: {test_dir}")