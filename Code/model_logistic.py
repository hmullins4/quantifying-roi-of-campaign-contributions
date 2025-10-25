# File: model_logistic.py
# Author: Hope E. Mullins

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    confusion_matrix, classification_report, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Base directories (relative to this script)
base_dir = Path(__file__).resolve().parent.parent  # Capstone/
input_dir = base_dir / "Data" / "aggregated"
figures_dir = base_dir / "Figures"
tables_dir = base_dir / "Tables"

# Ensure output folders exist
figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

cutoffs = [360, 240, 120, 60, 30, 14, 7, 1]

# Containers
aucs, accs, rec1s, rec0s, betas = [], [], [], [], []
per_type_derivs, p0s, elasticities = [], [], []

# ROC plot setup
plt.figure(figsize=(8,6))

for X in cutoffs:
    print(f"\n=== Cutoff = {X} days ===")

    # Load data
    df = pd.read_parquet(input_dir / f"agg_rid_{X}d.parquet")
    features = [
        f"n_contribs_{X}d", f"avg_tx_freq_{X}d", f"indiv_mill_{X}d",
        f"comm_mill_{X}d", f"corp_mill_{X}d", "num_givers",
        "ind_exp_support", "ind_exp_oppose", "pres_margin", "party_R",
        "party_Other", "incumbent"
    ]
    money_cols = features[2:5]
    X_df = df[features].copy()
    X_df.loc[:, money_cols] = np.log1p(X_df[money_cols])
    y = df["won_general"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    # Model
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l2', C=1e6, solver='lbfgs',
                           max_iter=1000, random_state=42)
    )
    clf.fit(X_train.values, y_train)

    # Predictions
    y_prob = clf.predict_proba(X_test.values)[:,1]
    y_pred = clf.predict(X_test.values)

    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    r1  = recall_score(y_test, y_pred)
    r0  = recall_score(y_test, y_pred, pos_label=0)
    aucs.append(auc); accs.append(acc); rec1s.append(r1); rec0s.append(r0)

    print(f"AUC (test) = {auc:.3f}")
    print(f"Accuracy (test) = {acc:.3f}")
    print(f"Recall (1) = {r1:.3f}, Recall (0) = {r0:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred))

    # ROI betas
    logreg = clf.named_steps['logisticregression']
    bi = logreg.coef_[0][features.index(money_cols[0])]
    bc = logreg.coef_[0][features.index(money_cols[1])]
    bp = logreg.coef_[0][features.index(money_cols[2])]
    beta_total = bi + bc + bp
    betas.append(beta_total)
    per_type_derivs.append([bi, bc, bp])
    p0s.append(None)

    print(f"\nTotal ROI β = {beta_total:.3f}, OR = {np.exp(beta_total):.2f}")
    for name, b in zip(money_cols, [bi, bc, bp]):
        print(f"  {name:16s} β = {b: .3f}, OR = {np.exp(b):.2f}")

    # Elasticity
    mean_spend = df[money_cols].sum(axis=1).mean()
    ele = beta_total * (mean_spend / y_test.mean())
    elasticities.append(ele)
    print(f"Elasticity (cutoff {X}d) = {ele:.3f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{X}d (AUC={auc:.3f})")

# Save ROC curve
plt.plot([0,1], [0,1], '--')
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(figures_dir / "log_roc.pdf")

# Helper to save LaTeX tables (no printing)
def save_table(df, filename, columns=None):
    tex = df.to_latex(
        index=False,
        column_format=columns if columns else None,
        float_format="%.3f"
    ).replace(r'\toprule',r'\hline').replace(r'\midrule',r'\hline\hline').replace(r'\bottomrule',r'\hline')
    with open(tables_dir / filename, "w", encoding="utf-8") as f:
        f.write(tex)

# Save tables
norm_el = [e/sum(elasticities) for e in elasticities]
save_table(pd.DataFrame({'Days':cutoffs, 'Normalized':norm_el}), "log_e_norm.tex", '||c c||')
save_table(pd.DataFrame({'Days':cutoffs, 'Elasticity':elasticities}), "log_e.tex", '||c c||')
save_table(pd.DataFrame({'Days':cutoffs,'AUC':aucs,'Accuracy':accs,'Recall (1)':rec1s,'Recall (0)':rec0s}),
           "log_metrics.tex", '||c c c c c||')
save_table(pd.DataFrame({'Days':cutoffs, r'Total ROI $\beta$':betas, 'OR':[np.exp(b) for b in betas]}),
           "log_roi.tex", '||c c c||')

per_b = np.array(per_type_derivs)
per_o = np.exp(per_b)
save_table(pd.DataFrame({
    'Days':cutoffs,
    'Ind $\\beta$':per_b[:,0],'Ind OR':per_o[:,0],
    'Comm $\\beta$':per_b[:,1],'Comm OR':per_o[:,1],
    'Corp $\\beta$':per_b[:,2],'Corp OR':per_o[:,2]
}), "log_type_roi.tex", '||c ' + 'c c ' * 3 + '||')

print("\nDone all cutoffs with logistic-based ROI interpretations.")

