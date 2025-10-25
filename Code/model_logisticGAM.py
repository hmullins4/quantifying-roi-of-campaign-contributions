# File: model_logisticGAM.py
# Author: Hope E. Mullins

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import scipy.sparse
def csr_matrix_A(self):
    return self.toarray()
scipy.sparse.csr_matrix.A = property(csr_matrix_A)

import os
import numpy as np
import pandas as pd
import scipy.sparse
from pygam import LogisticGAM, s, f
from functools import reduce
from operator import add
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Set base directories relative to this script location
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
input_dir = os.path.join(base_dir, 'Data', 'aggregated')
fig_dir = os.path.join(base_dir, 'Figures')
table_dir = os.path.join(base_dir, 'Tables')

# Cutoff days as before
cutoffs = [360, 240, 120, 60, 30, 14, 7, 1]

# Prepare containers for metrics & derivatives
aucs = []
accs = []
rec1s = []
rec0s = []
betas = []
per_type_derivs = []
p0s = []
elasticities = []

# Begin ROC plot
plt.figure(figsize=(8,6))

# Constant for finite differences
DELTA = 1e-4

for X in cutoffs:
    print(f"\n=== GAM Logistic & ROI, cutoff = {X}d ===")

    # --- load & prep ---
    df = pd.read_parquet(os.path.join(input_dir, f"agg_rid_{X}d.parquet"))
    money_cols = [f"indiv_mill_{X}d", f"comm_mill_{X}d", f"corp_mill_{X}d"]
    cont_cols  = [f"n_contribs_{X}d", f"avg_tx_freq_{X}d"] + money_cols + [
        "num_givers", "ind_exp_support", "ind_exp_oppose", "pres_margin"
    ]
    cat_cols   = ["party_R", "party_Other", "incumbent"]
    feats      = cont_cols + cat_cols

    df.loc[:, money_cols] = np.log1p(df[money_cols])
    X_raw = df[feats].values
    y     = df["won_general"].values

    scaler = StandardScaler().fit(X_raw)
    X_s    = scaler.transform(X_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y, test_size=0.25, random_state=42, stratify=y
    )

    # Convert sparse matrices to dense arrays if needed
    if scipy.sparse.issparse(X_tr):
        X_tr = X_tr.toarray()
    if scipy.sparse.issparse(X_te):
        X_te = X_te.toarray()

    spline_terms = [s(i, n_splines=10) for i in range(len(cont_cols))]
    factor_terms = [f(len(cont_cols) + j) for j in range(len(cat_cols))]
    gam_terms    = reduce(add, spline_terms + factor_terms)

    gam = LogisticGAM(gam_terms, max_iter=10000, tol=1e-1).gridsearch(X_tr, y_tr)

    # --- performance metrics ---
    y_prob = gam.predict_proba(X_te)
    y_pred = gam.predict(X_te)

    auc  = roc_auc_score(y_te, y_prob)
    acc  = accuracy_score(y_te, y_pred)
    r1   = recall_score(y_te, y_pred)
    r0   = recall_score(y_te, y_pred, pos_label=0)

    aucs.append(auc)
    accs.append(acc)
    rec1s.append(r1)
    rec0s.append(r0)

    print(f"AUC = {auc:.3f}")
    print(f"Accuracy = {acc:.3f}")
    print(f"Recall (1) = {r1:.3f}, Recall (0) = {r0:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))
    print("\n" + classification_report(y_te, y_pred))

    # --- numeric derivatives & ROI ---
    mean_raw = df[feats].mean().values.reshape(1, -1)
    mean_s   = scaler.transform(mean_raw)
    p0       = gam.predict_proba(mean_s)[0]

    derivs = []
    for col in money_cols:
        j    = cont_cols.index(col)
        x_up = mean_s.copy()
        x_up[0, j] += DELTA
        p_up = gam.predict_proba(x_up)[0]
        derivs.append((p_up - p0) / DELTA)

    per_type_derivs.append(derivs)
    p0s.append(p0)

    # total ROI β & OR
    m_raw       = df[money_cols].mean().values
    shares      = m_raw / m_raw.sum()
    delta_p     = np.dot(derivs, shares)
    logit_slope = delta_p / (p0 * (1 - p0))
    betas.append(logit_slope)

    print(f"Total ROI β = {logit_slope:.3f}, OR = {np.exp(logit_slope):.2f}")

    # per-type ROI printout (optional)
    for col, d in zip(money_cols, derivs):
        b_i = d / (p0 * (1 - p0))
        print(f"  {col:14s} β = {b_i: .3f}, OR = {np.exp(b_i):.2f}")

    # elasticity
    mean_spend = df[money_cols].sum(axis=1).mean()
    elasticity = logit_slope * (mean_spend / y_te.mean())
    elasticities.append(elasticity)
    print(f"Elasticity = {elasticity:.3f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    plt.plot(fpr, tpr, label=f"{X}d (AUC={auc:.3f})")

# finalize ROC
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "gam_log_roc.pdf"))

# normalized elasticities
total_elast = sum(elasticities)
norm_elast  = [e / total_elast for e in elasticities]

# —— 1) normalized elasticities table ——
norm_df = pd.DataFrame({
    'Days': cutoffs,
    'Normalized': norm_elast
})
with open(os.path.join(table_dir, "log_gam_e_norm.tex"), "w", encoding = "utf-8") as f:
    f.write(norm_df.to_latex(
        index=False,
        column_format='||c c||',
        header=['Days','Normalized'],
        float_format="%.3f"
    ).replace(r'\toprule',   r'\hline')
     .replace(r'\midrule',   r'\hline\hline')
     .replace(r'\bottomrule',r'\hline')
    )

# —— 2) raw elasticities table ——
elas_df = pd.DataFrame({
    'Days': cutoffs,
    'Elasticity': elasticities
})
with open(os.path.join(table_dir, "log_gam_e.tex"), "w", encoding = "utf-8") as f:
    f.write(elas_df.to_latex(
        index=False,
        column_format='||c c||',
        header=['Days','Elasticity'],
        float_format="%.3f"
    ).replace(r'\toprule',   r'\hline')
     .replace(r'\midrule',   r'\hline\hline')
     .replace(r'\bottomrule',r'\hline')
    )

# —— 3) performance metrics table ——
perf_df = pd.DataFrame({
    'Days': cutoffs,
    'AUC': aucs,
    'Accuracy': accs,
    'Recall (1)': rec1s,
    'Recall (0)': rec0s
})
with open(os.path.join(table_dir, "log_gam_metrics.tex"), "w", encoding = "utf-8") as f:
    f.write(perf_df.to_latex(
        index=False,
        column_format='||c c c c c||',
        float_format="%.3f"
    ).replace(r'\toprule',   r'\hline')
     .replace(r'\midrule',   r'\hline\hline')
     .replace(r'\bottomrule',r'\hline')
    )

# —— 4) total ROI β & OR table ——
roi_df = pd.DataFrame({
    'Days': cutoffs,
    r'Total ROI $\beta$': betas,
    'Odds-Ratio': [np.exp(b) for b in betas]
})
with open(os.path.join(table_dir, "log_gam_roi.tex"), "w", encoding = "utf-8") as f:
    f.write(roi_df.to_latex(
        index=False,
        column_format='||c c c||',
        float_format="%.3f"
    ).replace(r'\toprule',   r'\hline')
     .replace(r'\midrule',   r'\hline\hline')
     .replace(r'\bottomrule',r'\hline')
    )

# —— 5) per-type ROI β & OR table ——
# flatten per_type_derivs & p0s into β’s and OR’s
per_betas, per_ors = [], []
for derivs, p0 in zip(per_type_derivs, p0s):
    b_i = [d / (p0 * (1 - p0)) for d in derivs]
    per_betas.append(b_i)
    per_ors .append([np.exp(b) for b in b_i])

ind_b, comm_b, corp_b = zip(*per_betas)
ind_o, comm_o, corp_o = zip(*per_ors)

per_df = pd.DataFrame({
    'Days': cutoffs,
    'Ind $\\beta$': ind_b,   'Ind OR': ind_o,
    'Comm $\\beta$': comm_b, 'Comm OR': comm_o,
    'Corp $\\beta$': corp_b, 'Corp OR': corp_o,
})
with open(os.path.join(table_dir, "log_gam_type_roi.tex"), "w", encoding = "utf-8") as f:
    f.write(per_df.to_latex(
        index=False,
        column_format='||c ' + 'c c ' * 3 + '||',
        float_format="%.3f"
    ).replace(r'\toprule',   r'\hline')
     .replace(r'\midrule',   r'\hline\hline')
     .replace(r'\bottomrule',r'\hline')
    )

print("Done all cutoffs with GAM‑based ROI interpretations.")

