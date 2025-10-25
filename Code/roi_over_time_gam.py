# File: roi_over_time_gam.py
# Author: Hope E. Mullins

import warnings                                                
warnings.filterwarnings('ignore', category=RuntimeWarning)  

import scipy.sparse
def csr_matrix_A(self):
    return self.toarray()
scipy.sparse.csr_matrix.A = property(csr_matrix_A)

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pygam import LogisticGAM, s, f
from functools import reduce
from operator import add
from pathlib import Path

# Base directories (relative to this script)
base_dir    = Path(__file__).resolve().parent.parent
db_path     = base_dir / "Data" / "dime_house_cleaner.duckdb"
results_csv = base_dir / "Data" / "roi_over_time_gam.csv"
figures_dir = base_dir / "Figures"                              

# Ensure output folder exists ← added
figures_dir.mkdir(parents=True, exist_ok=True)

# Connect to DuckDB database
con = duckdb.connect(str(db_path))  # ← adjusted
print("Connected to database.")

# Range of cutoffs (1 to 365 days before election)
cutoff_range = range(1, 366)

# Lists to store results
cutoff_days = []
roi_betas   = []
roi_ors     = []
elasticities = []

DELTA = 1e-4

for X in cutoff_range:
    print(f"Processing cutoff = {X} days...")

    query = f"""
        SELECT
          bonica_rid,
          cycle,
          COUNT(*)                                         AS n_contribs,
          AVG(tx_type_freq)                                AS avg_tx_freq,
          SUM((1 - contrib_type) * amount) / 1e6           AS indiv_mill,
          SUM(contrib_type * amount) / 1e6                 AS comm_mill,
          SUM(is_corp * amount) / 1e6                      AS corp_mill,
          MAX(num_givers)                                  AS num_givers,
          MAX(ind_exp_support)                             AS ind_exp_support,
          MAX(ind_exp_oppose)                              AS ind_exp_oppose,
          MAX(district_pres_vs)                            AS pres_margin,
          MAX(CASE WHEN party='R' THEN 1 ELSE 0 END)       AS party_R,
          MAX(CASE WHEN party NOT IN ('D','R') THEN 1 ELSE 0 END) AS party_Other,
          MAX(is_incumbent)                                AS incumbent,
          MAX(gwinner)                                     AS won_general
        FROM house
        WHERE days_before >= {X}
        GROUP BY bonica_rid, cycle;
    """

    df = con.execute(query).fetchdf()

    if len(df) < 500:
        continue

    features = [
        "n_contribs", "avg_tx_freq", "indiv_mill",
        "comm_mill", "corp_mill", "num_givers",
        "ind_exp_support", "ind_exp_oppose", "pres_margin",
        "party_R", "party_Other", "incumbent"
    ]

    money_cols = ["indiv_mill", "comm_mill", "corp_mill"]
    df.loc[:, money_cols] = np.log1p(df[money_cols])
    y = df["won_general"].values
    X_raw = df[features].values

    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    cont_cols = features[:10]  # first 10 are continuous
    cat_cols  = features[10:]  # last 3 are categorical
    spline_terms = [s(i, n_splines=10) for i in range(len(cont_cols))]
    factor_terms = [f(len(cont_cols) + j) for j in range(len(cat_cols))]
    gam_terms    = reduce(add, spline_terms + factor_terms)

    try:
        gam = LogisticGAM(gam_terms, max_iter=10000, tol=1e-1) \
                  .gridsearch(
                      X_tr, y_tr,
                      progress=False,
                      lam=np.logspace(-3, 3, 5)
                  )

        # ROI Estimation
        mean_raw    = df[features].mean().values.reshape(1, -1)
        mean_scaled = scaler.transform(mean_raw)
        p0          = gam.predict_proba(mean_scaled)[0]

        derivs = []
        for col in money_cols:
            j    = features.index(col)
            x_up = mean_scaled.copy()
            x_up[0, j] += DELTA
            p_up = gam.predict_proba(x_up)[0]
            derivs.append((p_up - p0) / DELTA)

        m_raw      = df[money_cols].mean().values
        shares     = m_raw / m_raw.sum()
        logit_slope = np.dot(derivs, shares) / (p0 * (1 - p0))

        mean_spend   = df[money_cols].sum(axis=1).mean()
        mean_outcome = y_te.mean()

        elast = logit_slope * (mean_spend / mean_outcome)

        cutoff_days.append(X)
        roi_betas.append(logit_slope)
        roi_ors.append(np.exp(logit_slope))
        elasticities.append(elast)

    except Exception as e:
        print(f"Skipped cutoff {X} due to error: {e}")
        continue

con.close()

# Save results to DataFrame
results_df = pd.DataFrame({
    "cutoff_days": cutoff_days,
    "roi_beta":    roi_betas,
    "roi_or":      roi_ors,
    "elasticity":  elasticities
})

# Normalize elasticities
results_df["norm_elast"] = results_df["elasticity"] / results_df["elasticity"].sum()

# Save to CSV
results_df.to_csv(results_csv, index=False)  # ← adjusted

# === PLOTTING ===
plt.figure(figsize=(12, 6))
plt.plot(results_df.cutoff_days, results_df.roi_or, label="Odds Ratio (ROI)", lw=3, color='black')
plt.axhline(1.0, linestyle='--', label='OR = 1 (no effect)')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Odds Ratio for $1M", fontsize = 14)
plt.gca().invert_xaxis()
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig(figures_dir / "gam_roi_over_time.pdf")

plt.figure(figsize=(12, 6))
plt.plot(results_df.cutoff_days, results_df.roi_beta, label="Total ROI β (log-odds)", lw=3, color='black')
plt.axhline(0.0, linestyle='--', label='β = 0 (no effect)', color = 'bold')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Total ROI β (log-odds)", fontsize = 14)
plt.gca().invert_xaxis()
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig(figures_dir / "gam_roi_beta_over_time.pdf")

plt.figure(figsize=(12, 6))
plt.plot(results_df.cutoff_days, results_df.elasticity, label="Raw Elasticity", lw=3, color='black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Elasticity", fontsize = 14)
plt.gca().invert_xaxis()
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(figures_dir / "gam_elasticity_over_time.pdf")

plt.figure(figsize=(12, 6))
plt.bar(results_df.cutoff_days, results_df.norm_elast, width=1.0, color='black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Normalized Elasticity", fontsize = 14)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(figures_dir / "gam_norm_elast_over_time.pdf")
