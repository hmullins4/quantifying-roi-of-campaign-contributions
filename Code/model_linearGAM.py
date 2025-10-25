# File: model_linearGAM.py
# Author: Hope E. Mullins

import scipy.sparse
def csr_matrix_A(self):
    return self.toarray()
scipy.sparse.csr_matrix.A = property(csr_matrix_A)

import os
import numpy as np
import pandas as pd

from pygam import LinearGAM, s, f
from functools import reduce
from operator import add

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from pathlib import Path  # ← added

# Base directories (relative to this script) ← added
base_dir   = Path(__file__).resolve().parent.parent  # Capstone/
input_dir  = base_dir / "Data" / "aggregated"        # ← adjusted
tables_dir = base_dir / "Tables"                     # ← added

# Ensure output folder exists ← added
tables_dir.mkdir(parents=True, exist_ok=True)

# Same cutoff days as specified before
cutoffs = [360, 240, 120, 60, 30, 14, 7, 1]

# Prepare containers for metrics and elasticities
elasticities     = []
cv_r2_means      = []
cv_rmse_means    = []
cv_mae_means     = []
ho_r2_list       = []
ho_rmse_list     = []
ho_mae_list      = []
delta_pcts       = []

# Grid of smoothing penalties
lams = np.logspace(-2, 3, 10)
# Number of spline basis functions per continuous term
n_splines = 8
# Cross-validation scheme
n_splits = 5
n_repeats = 3
random_seed = 0
# Finite-difference step for ROI
delta = 1e-4

# Helper functions
def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

if __name__ == "__main__":
    for X in cutoffs:
        print(f"\n=== Cutoff = {X} days ===")

        # Load data and define features
        df = pd.read_parquet(input_dir / f"agg_rid_{X}d.parquet")  # ← adjusted
        money_cols = [f"indiv_mill_{X}d", f"comm_mill_{X}d", f"corp_mill_{X}d"]
        cont_cols = ([f"n_contribs_{X}d", f"avg_tx_freq_{X}d"]
                     + money_cols
                     + ["num_givers", "ind_exp_support", "ind_exp_oppose", "pres_margin"])
        cat_cols = ["party_R", "party_Other", "incumbent"]
        feats = cont_cols + cat_cols

        # Prepare X and y
        X_df = df[feats].copy()
        X_df[money_cols] = np.log1p(X_df[money_cols])
        y = df["gen_vote_pct"].astype(float).values

        # Standardize
        X_raw  = X_df.values
        scaler = StandardScaler().fit(X_raw)
        X_s    = scaler.transform(X_raw)

        # Build GAM terms
        spline_terms = [s(i, n_splines=n_splines) for i in range(len(cont_cols))]
        factor_terms = [f(len(cont_cols) + j) for j in range(len(cat_cols))]
        gam_terms    = reduce(add, spline_terms + factor_terms)

        # Cross-validated tuning & evaluation
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)
        r2_list, rmse_list, mae_list = [], [], []
        for train_idx, test_idx in rkf.split(X_s):
            X_tr, X_te = X_s[train_idx], X_s[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            gam = LinearGAM(gam_terms).gridsearch(X_tr, y_tr, lam=lams, progress=False)
            y_pred = gam.predict(X_te)
            r2, rmse, mae = compute_metrics(y_te, y_pred)
            r2_list.append(r2)
            rmse_list.append(rmse)
            mae_list.append(mae)

        # Store CV means
        cv_r2_means.append(np.mean(r2_list))
        cv_rmse_means.append(np.mean(rmse_list))
        cv_mae_means.append(np.mean(mae_list))

        print(f"CV   R²   = {cv_r2_means[-1]:.3f} ± {np.std(r2_list):.3f}")
        print(f"CV RMSE   = {cv_rmse_means[-1]:.3f} ± {np.std(rmse_list):.3f}")
        print(f"CV MAE    = {cv_mae_means[-1]:.3f} ± {np.std(mae_list):.3f}")

        # Final hold-out fit
        X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.25, random_state=42)
        gam_final = LinearGAM(gam_terms).gridsearch(X_tr, y_tr, lam=lams, progress=False)

        # Hold-out metrics
        y_hold = gam_final.predict(X_te)
        r2_hold, rmse_hold, mae_hold = compute_metrics(y_te, y_hold)
        ho_r2_list.append(r2_hold)
        ho_rmse_list.append(rmse_hold)
        ho_mae_list.append(mae_hold)

        print(f"Hold-out R²   = {r2_hold:.3f}")
        print(f"Hold-out RMSE = {rmse_hold:.3f}")
        print(f"Hold-out MAE  = {mae_hold:.3f}")

        # ROI calculation
        mean_raw    = X_raw.mean(axis=0, keepdims=True)
        mean_scaled = scaler.transform(mean_raw)
        base_pred   = gam_final.predict(mean_scaled)[0]

        derivs = []
        for col in money_cols:
            idx = cont_cols.index(col)
            bumped = mean_scaled.copy()
            bumped[0, idx] += delta
            derivs.append((gam_final.predict(bumped)[0] - base_pred) / delta)

        shares = df[money_cols].mean().values
        shares /= shares.sum()
        delta_pct = np.dot(derivs, shares)
        delta_pcts.append(delta_pct)

        print(f"\nΔ(gen_vote_pct) ≈ {delta_pct:.4f} pp per $1M total (avg mix)")
        print("-" * 60)
        
        # Elasticity calculation
        mean_spend_X = df[money_cols].sum(axis = 1).mean()
        mean_outcome = y_te.mean()
        elasticity_X = delta_pct * (mean_spend_X / mean_outcome)
        elasticities.append(elasticity_X)
        print(f"Elasticity (cutoff {X}d) = {elasticity_X:.3f}")        

    print("\nAll cutoffs processed.")

    # After loop: write LaTeX tables
    # 1) CV & Δ table
    cv_df = pd.DataFrame({
        'Days': cutoffs,
        'CV $R^2$': cv_r2_means,
        'CV RMSE': cv_rmse_means,
        'CV MAE': cv_mae_means,
        '$\\Delta$ (pp per $1M$)': delta_pcts
    })
    with open(tables_dir / "lin_gam_cv.tex", "w", encoding="utf-8") as f:  # ← adjusted
        f.write(cv_df.to_latex(index=False,
                                column_format='||c c c c c||',
                                float_format="%.3f")
               .replace(r"\toprule", r"\hline")
               .replace(r"\midrule", r"\hline\hline")
               .replace(r"\bottomrule", r"\hline"))

    # 2) Hold-out metrics table
    ho_df = pd.DataFrame({
        'Days': cutoffs,
        'HO $R^2$': ho_r2_list,
        'HO RMSE': ho_rmse_list,
        'HO MAE': ho_mae_list
    })
    with open(tables_dir / "lin_gam_ho.tex", "w", encoding="utf-8") as f:  # ← adjusted
        f.write(ho_df.to_latex(index=False,
                                 column_format='||c c c c||',
                                 float_format="%.3f")
               .replace(r"\toprule", r"\hline")
               .replace(r"\midrule", r"\hline\hline")
               .replace(r"\bottomrule", r"\hline"))

    # 3) Elasticity table
    elas_df = pd.DataFrame({
        'Days': cutoffs,
        'Elasticity': elasticities
    })
    with open(tables_dir / "lin_gam_e.tex", "w", encoding="utf-8") as f:  # ← adjusted
        f.write(elas_df.to_latex(index=False,
                                  column_format='||c c||',
                                  float_format="%.3f")
               .replace(r"\toprule", r"\hline")
               .replace(r"\midrule", r"\hline\hline")
               .replace(r"\bottomrule", r"\hline"))

    print("Finished writing GAM tables.")

