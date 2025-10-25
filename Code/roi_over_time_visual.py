# File: roi_over_time_visual.py
# Author: Hope E. Mullins

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path 

# Base directories (relative to this script)
base_dir    = Path(__file__).resolve().parent.parent
db_path     = base_dir / "Data" / "dime_house_cleaner.duckdb" 
results_csv = base_dir / "Data" / "roi_over_time.csv"
figures_dir = base_dir / "Figures"

# Ensure output folder exists
figures_dir.mkdir(parents=True, exist_ok=True)

# Connect to DuckDB database
con = duckdb.connect(str(db_path))  
print("Connected to database.")

# Range of cutoffs (e.g., 1 to 365 days before election)
cutoff_range = range(1, 366)

# Lists to store results
cutoff_days = []
roi_betas   = []
roi_ors     = []
elasticities = []

# Looping and aggregation
for X in cutoff_range:
    print(f"Processing cutoff = {X} days...")
    
    # Query: Reaggregate on the fly
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
          MAX(CASE WHEN party NOT IN ('D','R') THEN 1 ELSE 0 END)
                                                           AS party_Other,
          MAX(is_incumbent)                                AS incumbent,
          MAX(gwinner)                                     AS won_general
        FROM house
        WHERE days_before >= {X}
        GROUP BY bonica_rid, cycle;
    """

    df = con.execute(query).fetchdf()

    # Skip if too few rows
    if len(df) < 500:
        continue

    features = [
        "n_contribs", "avg_tx_freq", "indiv_mill",
        "comm_mill", "corp_mill", "num_givers",
        "ind_exp_support", "ind_exp_oppose", "pres_margin",
        "party_R", "party_Other", "incumbent"
    ]

    X_df = df[features]
    money_cols = ["indiv_mill", "comm_mill", "corp_mill"]
    X_df.loc[:, money_cols] = np.log1p(X_df[money_cols])
    y = df["won_general"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic regression pipeline
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=1000, random_state=42)
    )

    try:
        clf.fit(X_train, y_train)
        logreg = clf.named_steps['logisticregression']

        # Extract ROI coefficients
        bi = logreg.coef_[0][features.index("indiv_mill")]
        bc = logreg.coef_[0][features.index("comm_mill")]
        bp = logreg.coef_[0][features.index("corp_mill")]
        beta_total = bi + bc + bp
        
        # Compute elasticity
        mean_spend    = df[money_cols].sum(axis = 1).mean()
        mean_outcome  = df["won_general"].mean()
        elast = beta_total * (mean_spend / mean_outcome)

        cutoff_days.append(X)
        roi_betas.append(beta_total)
        roi_ors.append(np.exp(beta_total))
        elasticities.append(elast)
    except Exception as e:
        print(f"Skipped cutoff {X} due to error: {e}")
        continue

con.close()

# Save results to CSV
results_df = pd.DataFrame({
    "cutoff_days": cutoff_days,
    "roi_beta": roi_betas,
    "roi_or": roi_ors,
    "elasticity": elasticities
})

# Normalize elasticities
results_df["norm_elast"] = results_df["elasticity"] / results_df["elasticity"].sum()

# Save to CSV
results_df.to_csv(results_csv, index=False) 

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(cutoff_days, roi_ors, label="Odds Ratio (ROI)", lw=3, color = 'black')
plt.axhline(1.0, linestyle='--', label='OR = 1 (no effect)', color = 'black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Odds Ratio for $1M", fontsize = 14)
plt.gca().invert_xaxis()
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig(figures_dir / "roi_over_time.pdf")

plt.figure(figsize=(12, 6))
plt.plot(cutoff_days, roi_betas, label="Total ROI β (log-odds)", lw=3, color = 'black')
plt.axhline(0.0, linestyle='--', label='β = 0 (no effect)', color = 'black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Total ROI β (log-odds)", fontsize = 14)
plt.gca().invert_xaxis()
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig(figures_dir / "roi_beta_over_time.pdf")

plt.figure(figsize=(12, 6))
plt.plot(results_df.cutoff_days, results_df.elasticity, label="Raw Elasticity", lw=3, color = 'black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Elasticity", fontsize = 14)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(figures_dir / "elasticity_over_time.pdf") 

plt.figure(figsize=(12, 6))
plt.bar(results_df.cutoff_days, results_df.norm_elast, width=1.0, color = 'black')
plt.xlabel("Days Before Election", fontsize = 14)
plt.ylabel("Normalized Elasticity", fontsize = 14)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(figures_dir / "norm_elast_over_time.pdf") 
