# MSF-UNet: Four-Dimensional Wind Retrieval From FY-4B AGRI and GIIRS Through Joint Deep Learning

This repository contains the prediction and visualization code for the paper:

> Four-Dimensional Wind Retrieval From FY-4B AGRI and GIIRS Through Joint Deep Learning.


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
├── unet_model/
│   ├── predict.py             # Inference pipeline for single samples
│   └── analyze_levels.py      # Per-level RMSE/MAE/Bias analysis on test set
│
├── 15min_wind/
│   └── predict_15min_paper.py # 15-min squall-line case study (Figure 3 in paper)
│                               #   Generates AGRI BT + retrieved 850 hPa wind composites
│
└── plot_code/                 # Visualization scripts
    ├── plot_3D_wind.py        # 3D wind field visualization
    ├── plot_ERA5.py           # ERA5 reference wind plotting
    ├── plot_ERA5_uvw.py       # ERA5 u/v/w component plotting
    ├── plot_GIIRS_BT.py       # GIIRS brightness temperature plotting
    ├── plot_GIIRS_R.py        # GIIRS radiance plotting
    ├── plot_predict_wind.py   # Predicted wind field visualization
    ├── plot_npz.py            # NPZ data inspection
    └── plot_gif.py            # Animation generation
```

## Environment

- Python 3.10+
- PyTorch 2.x
- CUDA (recommended)
- Key dependencies: `numpy`, `h5py`, `xarray`, `pandas`, `matplotlib`, `scipy`, `tensorboard`

Install:
```bash
pip install torch numpy h5py xarray pandas matplotlib scipy tensorboard
```

## Usage

### 1. Prediction (Inference)

```bash
cd unet_model/
python predict.py
```

The trained model checkpoint is loaded from `unet_model/checkpoint/unet_best.pth`.

### 2. Per-Level Error Analysis

```bash
cd unet_model/
python analyze_levels.py
```

Outputs per-level RMSE, MAE, and Bias at all 37 pressure levels (CSV + figure).

### 3. 15-min Squall-Line Case Study (Figure 3 in paper)

```bash
cd 15min_wind/
python predict_15min_paper.py
```

Generates AGRI BT + retrieved 850 hPa wind composite figures for the April 12, 2025 squall-line case over Guangdong, China.


## License

This code is provided for research and review purposes. For other uses, please contact the corresponding author.
