# MSF-UNet: Four-Dimensional Wind Retrieval From FY-4B AGRI and GIIRS Through Joint Deep Learning

This repository contains the key code for the paper:

> Yang, J., Xu, N.\*, Li, L., Ran, G., Liu, H., Zhang, Y., Dou, F., Zhai, X., Liu, X., Zhao, K., Li, Z., & Zhang, P. Four-Dimensional Wind Retrieval From FY-4B AGRI and GIIRS Through Joint Deep Learning.

## Overview

MSF-UNet (Multi-Source Fusion U-Net) retrieves 15-minute four-dimensional horizontal winds at 37 pressure levels from routine FY-4B satellite observations. The model fuses:

- **AGRI** (Advanced Geostationary Radiation Imager): 15-channel multispectral imager providing 15-min full-disk scans, capturing rapid cloud and radiance evolution
- **GIIRS** (Geostationary Interferometric Infrared Sounder): 1690-channel hyperspectral sounder providing vertically sensitive atmospheric information on an approximately 2-hour cycle

The model outputs zonal (u) and meridional (v) wind components at 37 pressure levels (1000–1 hPa). During inference, only AGRI, GIIRS, and a time-difference mask (Δt) are required — ERA5 is not needed.

### Key Results

| Metric | Value |
|--------|-------|
| Pressure-level-mean vector RMSE (vs ERA5) | 3.645 m s⁻¹ |
| w/o GIIRS degradation | +2.54% |
| w/o AGRI degradation | +1.06% |
| Dropsonde u-wind RMSE / R | 2.75 m s⁻¹ / 0.92 |
| Dropsonde v-wind RMSE / R | 2.91 m s⁻¹ / 0.84 |

## Code Structure

```
.
├── unet_model/              # Main MSF-UNet model
│   ├── model.py             # Network architecture (multi-encoder, dual-decoder U-Net)
│   ├── dataset.py           # PyTorch Dataset with normalization and masking
│   ├── train.py             # Training script (Adam, warmup + cosine decay, checkpointing)
│   ├── predict.py           # Inference on single samples
│   └── analyze_levels.py    # Per-level RMSE/MAE/Bias analysis on test set
│
├── 15min_wind/              # 15-min squall-line case study (Figure 3 in paper)
│   ├── model.py             # Model definition
│   ├── dataset.py           # Dataset for 15-min prediction
│   └── predict_15min_paper.py  # Generate Figures 3 (BT + wind composite)
│
├── process_data/            # Data preprocessing scripts
│   ├── make_npz_uvw.py      # Main NPZ creation: match AGRI+GIIRS+ERA5 into training samples
│   ├── match_GIIRS_AGRI.py  # Spatial matching between GIIRS footprints and AGRI pixels
│   ├── match_GIIRS_ERA5.py  # Spatial matching between GIIRS footprints and ERA5 grid
│   ├── process_ERA5_uv.py   # ERA5 u/v wind extraction
│   ├── process_GIIRS.py     # GIIRS radiance extraction
│   ├── extract_GIIRS_to_csv.py  # GIIRS data extraction to CSV
│   ├── calculate_normalized_parameters.py  # Compute normalization mean/std
│   ├── make_mask.py         # Create spatial masks for invalid regions
│   ├── make_npz_uv.py       # Alternative NPZ creation (u/v only)
│   └── spilt_train_test.py  # Train/test split (8:2 ratio)
│
├── plot_code/               # Visualization scripts
│   ├── plot_3D_wind.py      # 3D wind field visualization
│   ├── plot_ERA5.py         # ERA5 reference wind plotting
│   ├── plot_ERA5_uvw.py     # ERA5 u/v/w component plotting
│   ├── plot_GIIRS_BT.py     # GIIRS brightness temperature plotting
│   ├── plot_GIIRS_R.py      # GIIRS radiance plotting
│   ├── plot_predict_wind.py # Predicted wind field visualization
│   ├── plot_npz.py          # NPZ data inspection
│   └── plot_gif.py          # Animation generation
│
└── (ablation models and earlier development scripts are available upon request)
```

## Environment

- Python 3.10+
- PyTorch 2.x
- CUDA (recommended for training)
- Key dependencies: `numpy`, `h5py`, `xarray`, `pandas`, `matplotlib`, `scipy`, `tensorboard`

Install:
```bash
pip install torch numpy h5py xarray pandas matplotlib scipy tensorboard
```

## Usage

### 1. Data Preprocessing

The training data is constructed from FY-4B Level-1 AGRI/GIIRS observations and ERA5 reanalysis:

```bash
cd process_data/

# Step 1: Match GIIRS footprints with AGRI pixels and ERA5 grid points
python match_GIIRS_AGRI.py
python match_GIIRS_ERA5.py

# Step 2: Process raw satellite data
python process_GIIRS.py
python process_ERA5_uv.py

# Step 3: Create training NPZ files (AGRI + GIIRS + ERA5 samples)
python make_npz_uvw.py

# Step 4: Compute normalization parameters
python calculate_normalized_parameters.py

# Step 5: Create spatial masks
python make_mask.py

# Step 6: Split into train/test sets (8:2)
python spilt_train_test.py
```

Expected auxiliary data structure:
```
data/
├── auxiliary_data/
│   ├── normalized_parameters_mean_std_uvw.npz
│   ├── AGRI_mask.npy
│   └── GIIRS_mask.npy
└── npz_file/
    ├── GIIRS_store/          # GIIRS data (shared across samples)
    └── with_era5_uvw/
        ├── train/            # Training NPZ files
        └── test/             # Test NPZ files
```

### 2. Training

Edit paths in `unet_model/train.py` and `unet_model/dataset.py`, then:

```bash
cd unet_model/
python train.py
```

Key hyperparameters (in `train.py`):
- Batch size: 6
- Learning rate: 5e-5 (warmup 100 steps, cosine decay)
- Epochs: 30
- Optimizer: Adam
- Loss: MSE(u) + MSE(v)

### 3. Inference

```bash
cd unet_model/
python predict.py
```

The trained model checkpoint is loaded from `unet_model/checkpoint/unet_best.pth`.

### 4. Per-Level Error Analysis

```bash
cd unet_model/
python analyze_levels.py
```

Outputs per-level RMSE, MAE, and Bias at all 37 pressure levels (CSV + figure).

### 5. 15-min Squall-Line Case Study (Figure 3)

```bash
cd 15min_wind/
python predict_15min_paper.py
```

Generates AGRI BT + retrieved 850 hPa wind composite figures for the April 12, 2025 squall-line case.

## Model Architecture

MSF-UNet is a multi-encoder, dual-decoder U-Net:

- **Inputs**: AGRI current (15×576×648), AGRI previous (15×576×648), GIIRS (1690×192×216), Δt mask (1×192×216)
- **AGRI encoder**: 6 levels of conv+downsample, AGRI spatial resolution downsampled to match GIIRS grid
- **GIIRS encoder**: 5 levels of conv+downsample
- **Δt encoder**: 5 levels of conv+downsample (smaller channel width)
- **Bottleneck**: Concatenation of AGRI + GIIRS + Δt features
- **Decoders**: Two independent branches for u and v wind components, each with skip connections from AGRI and GIIRS encoders
- **Output**: u (37×192×216) and v (37×192×216)

## Data Availability

- FY-4B AGRI and GIIRS Level-1 data: [FengYun Satellite Data Service](http://satellite.nsmc.org.cn/) (registration required)
- ERA5 hourly pressure-level data: [Copernicus CDS](https://doi.org/10.24381/cds.bd0915c6)
- Dropsonde observations: 2025 South China Sea aircraft dropsonde dataset (Guo et al., 2025)

## Ablation Experiments

The ablation experiments reported in the paper (w/o GIIRS, w/o AGRI, w/o Δt) use the same MSF-UNet backbone as the full model, differing only in input configuration. The ablation model variants are not included in this repository but can be obtained by contacting the corresponding author.

## License

This code is provided for research and review purposes. For other uses, please contact the corresponding author.
