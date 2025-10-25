# File: gam_plots.py
# Author: Hope E. Mullins

import warnings                                                
warnings.filterwarnings('ignore', category=RuntimeWarning)

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pygam import LogisticGAM, s, f
from functools import reduce
from operator import add

from pathlib import Path  # ← added

# Base directory (relative to this script)
base_dir    = Path(__file__).resolve().parent.parent  # Capstone/
input_csv   = base_dir / "Data" / "roi_over_time_gam.csv"  # ← adjusted
figures_dir = base_dir / "Figures"                        # ← added

# Ensure output folder exists
figures_dir.mkdir(parents=True, exist_ok=True)

# Read results
results_df = pd.read_csv(input_csv)

# Plot: Odds Ratio (ROI)
plt.figure(figsize=(12, 6))
plt.plot(results_df.cutoff_days, results_df.roi_or, label="Odds Ratio (ROI)", lw=3, color = 'black')
plt.axhline(1.0, linestyle='--', label='OR = 1 (no effect)', color = 'black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Odds Ratio for $1M", fontsize = 14)
plt.gca().invert_xaxis()
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig(figures_dir / "gam_roi_over_time.pdf")

# Plot: Total ROI β (log-odds)
plt.figure(figsize=(12, 6))
plt.plot(results_df.cutoff_days, results_df.roi_beta, label="Total ROI β (log-odds)", lw=3, color = 'black')
plt.axhline(0.0, linestyle='--', label='β = 0 (no effect)', color = 'black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Total ROI β (log-odds)", fontsize = 14)
plt.gca().invert_xaxis()
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig(figures_dir / "gam_roi_beta_over_time.pdf")  

# Plot: Raw Elasticity
plt.figure(figsize=(12, 6))
plt.plot(results_df.cutoff_days, results_df.elasticity, label="Raw Elasticity", lw=3, color = 'black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Elasticity", fontsize = 14)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(figures_dir / "gam_elasticity_over_time.pdf")

# Plot: Normalized Elasticity
plt.figure(figsize=(12, 6))
plt.bar(results_df.cutoff_days, results_df.norm_elast, width=1.0, color = 'black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Normalized Elasticity", fontsize = 14)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(figures_dir / "gam_norm_elast_over_time.pdf")

