# File: model_linear.py
# Author: Hope E. Mullins

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Base directories (relative to this script)
base_dir     = Path(__file__).resolve().parent.parent  # Capstone/
input_dir    = base_dir / "Data" / "aggregated"
figures_dir  = base_dir / "Figures"
tables_dir   = base_dir / "Tables"

# Ensure output folders exist
figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

# Cutoff days
cutoffs = [360, 240, 120, 60, 30, 14, 7, 1]

# Containers for metrics & results
elasticities = []
beta_totals   = []
delta_ps      = []
r2s           = []
cv_means      = []
rmses         = []
maes          = []
best_models   = []
best_alphas   = []
best_r2s      = []

# Containers for plotting & significance
importance_path = {
    'ind_exp_oppose': [],
    'ind_exp_support': [],
    'n_contribs': [],
    'incumbent': [],
    'num_givers': []
}
base_feats_template = [
    "n_contribs", "avg_tx_freq",
    "indiv_mill", "comm_mill", "corp_mill",
    "num_givers", "ind_exp_support", "ind_exp_oppose",
    "pres_margin", "party_R", "party_Other", "incumbent"
]
pvals_path = { feat: [] for feat in base_feats_template }

# Regularization grid
alphas   = np.logspace(-3, 3, 25)
n_splits = 5

# Main loop over cutoffs
for X in cutoffs:
    print(f"\n=== Cutoff = {X} days before election ===")

    # Load parquet
    df = pd.read_parquet(input_dir / f"agg_rid_{X}d.parquet")

    # Feature engineering
    base_feats = [
        f"{feat}_{X}d" if (feat.endswith("_mill") or feat.endswith("_freq") or feat.startswith("n_"))
        else feat
        for feat in base_feats_template
    ]
    X_df = df[base_feats].copy()
    money_cols = [f"indiv_mill_{X}d", f"comm_mill_{X}d", f"corp_mill_{X}d"]
    X_df[money_cols] = np.log1p(X_df[money_cols])
    y = df["gen_vote_pct"].astype(float)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.25, random_state=42
    )

    # Simple OLS pipeline
    simple_pipe = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    simple_pipe.fit(X_train, y_train)
    y_pred = simple_pipe.predict(X_test)

    # Metrics
    r2_test = r2_score(y_test, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    mae     = mean_absolute_error(y_test, y_pred)
    cv_r2   = cross_val_score(simple_pipe, X_df, y, cv=5, scoring='r2')
    print(f"[Simple OLS] R² = {r2_test:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
    print(f"[Simple OLS] CV R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    # Store metrics
    r2s.append(r2_test)
    cv_means.append(cv_r2.mean())
    rmses.append(rmse)
    maes.append(mae)

    # Statistical significance
    X_sm   = sm.add_constant(X_train)
    sm_ols = sm.OLS(y_train, X_sm).fit()
    print(sm_ols.summary().tables[1])

    # Standardized coefficients
    lr     = simple_pipe.named_steps['linearregression']
    scaler = simple_pipe.named_steps['standardscaler']
    std_coefs = lr.coef_ * scaler.scale_
    imp = pd.Series(std_coefs, index=base_feats).abs().sort_values(ascending=False)
    for feat in ('ind_exp_oppose','ind_exp_support','n_contribs','incumbent','num_givers'):
        key = feat if feat!='n_contribs' else f'n_contribs_{X}d'
        importance_path[feat].append(imp[key])
    for feat in base_feats_template:
        col = feat + f"_{X}d" if feat in ["n_contribs","avg_tx_freq","indiv_mill","comm_mill","corp_mill"] else feat
        pvals_path[feat].append(sm_ols.pvalues[col])

    # Total ROI β
    beta_total_orig = sum(sm_ols.params[f"{m}_{X}d"] for m in ("indiv_mill","comm_mill","corp_mill"))
    print(f"[Simple OLS] Total ROI β = {beta_total_orig:.4f} pp per $1M")
    beta_totals.append(beta_total_orig)

    # Δ (pp per $1M)
    raw_means      = df[money_cols].mean()
    log_increments = np.log1p(raw_means+1) - np.log1p(raw_means)
    shares         = raw_means / raw_means.sum()
    mean_vals      = [df[c].mean() for c in base_feats]
    mean_df        = pd.DataFrame([mean_vals], columns=base_feats)
    p0 = simple_pipe.predict(mean_df)[0]
    bumped = mean_df.copy()
    for col in money_cols:
        bumped[col] += shares[col] * log_increments[col]
    p1 = simple_pipe.predict(bumped)[0]
    delta = p1 - p0
    print(f"Δ(gen_vote_pct) ≈ {delta:.4f} pp per $1M")
    delta_ps.append(delta)

    # Elasticity
    ele = beta_total_orig * (df[money_cols].sum(axis=1).mean() / y_test.mean())
    elasticities.append(ele)
    print(f"Elasticity (cutoff {X}d) = {ele:.3f}")

    # Regularized models
    X_imp = X_df.copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.25, random_state=42)

    ridge = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, cv=KFold(n_splits, shuffle=True, random_state=1), scoring='r2')
    ).fit(X_tr, y_tr)
    r2_ridge, a_r = ridge.score(X_te, y_te), ridge.named_steps['ridgecv'].alpha_

    lasso = make_pipeline(
        StandardScaler(),
        LassoCV(alphas=alphas, cv=KFold(n_splits, shuffle=True, random_state=1), max_iter=5000)
    ).fit(X_tr, y_tr)
    r2_lasso, a_l = lasso.score(X_te, y_te), lasso.named_steps['lassocv'].alpha_

    if r2_ridge >= r2_lasso:
        best_pipe, name, best_r2, best_alpha = ridge, 'RidgeCV', r2_ridge, a_r
    else:
        best_pipe, name, best_r2, best_alpha = lasso, 'LassoCV', r2_lasso, a_l

    print(f"[Improved] {name}: α={best_alpha:.4f}, R²={best_r2:.3f}")
    best_models.append(name)
    best_alphas.append(best_alpha)
    best_r2s.append(best_r2)

    # Top 5 importances
    lr_imp   = best_pipe.named_steps[name.lower()]
    scaler_i = best_pipe.named_steps['standardscaler']
    imp_i = pd.Series(lr_imp.coef_ * scaler_i.scale_, index=X_imp.columns)\
             .abs().sort_values(ascending=False)
    print(imp_i.head(5).to_string())
    print("-" * 50)

# Finished loop
print("All cutoffs done.")

# Normalize elasticities & print
norm_elast = [e/sum(elasticities) for e in elasticities]
for c, ne in zip(cutoffs, norm_elast):
    print(f"{c}d → {ne:.3f}")

# Plot feature‐importance paths
plt.figure(figsize=(10,5))
for feat, vals in importance_path.items():
    plt.plot(cutoffs, vals, marker='o', label=feat)
ax = plt.gca()
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax.invert_xaxis()
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Absolute Std. Coef. (log)", fontsize = 14)
plt.tight_layout()
plt.savefig(figures_dir / "feat_imp.pdf")
plt.close()

# Heatmap of p‐values
raw_p = pd.DataFrame(pvals_path, index=cutoffs).T
plt.figure(figsize=(8,6))
sns.heatmap(raw_p, annot=True, fmt=".3f", cmap=LinearSegmentedColormap.from_list('gray',['#E6E6FA','#A0A0A0']),
            vmin=0, vmax=0.075, cbar_kws={'label':'p-value'})
plt.gca().invert_xaxis()
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Feature", fontsize = 14)
plt.tight_layout()
plt.savefig(figures_dir / "sig_heatmap.pdf")
plt.close()

# Helper to save LaTeX tables
def save_table(df, filename, col_fmt=None, float_fmt="%.3f"):
    tex = df.to_latex(index=False,
                      column_format=col_fmt,
                      float_format=float_fmt)\
            .replace(r"\toprule",r"\hline")\
            .replace(r"\midrule",r"\hline\hline")\
            .replace(r"\bottomrule",r"\hline")
    with open(tables_dir / filename, "w", encoding="utf-8") as f:
        f.write(tex)

# Write tables
roi_df = pd.DataFrame({
    'Days': cutoffs,
    'Total ROI $\\beta$': beta_totals,
    '$\\Delta$': delta_ps
})
save_table(roi_df, "lin_roi.tex", '||c c c||', float_fmt="%.4f")

perf_df = pd.DataFrame({
    'Days': cutoffs,
    '$R^2$': r2s,
    'CV $R^2$': cv_means,
    'RMSE': rmses,
    'MAE': maes
})
save_table(perf_df, "lin_metrics.tex", '||c c c c c||')

best_df = pd.DataFrame({
    'Days': cutoffs,
    'Best Model ($\\alpha$)': [f"{m} ({a:.4f})" for m,a in zip(best_models, best_alphas)],
    '$R^2$': best_r2s
})
save_table(best_df, "reg_r2.tex", '||c c c||')

elas_df = pd.DataFrame({
    'Days': cutoffs,
    'Elasticity': elasticities
})
save_table(elas_df, "lin_e.tex", '||c c||')

print("Finished writing all tables.")

