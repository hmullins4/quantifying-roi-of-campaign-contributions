# File: aggregation.py
# Author: Hope E. Mullins

import duckdb
import os
from pathlib import Path  # ← added

""" Connect to the cleaned DuckDB file containing the
fully-featured `house` table.
"""

# Base directories (relative to this script) ← added
base_dir   = Path(__file__).resolve().parent.parent  # Capstone/
db_path    = base_dir / "Data" / "dime_house_cleaner.duckdb"  # ← adjusted
output_dir = base_dir / "Data" / "aggregated"  # ← adjusted

# Ensure output directory exists ← added
output_dir.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(db_path))  # ← adjusted
print("Connected to database.")

""" Choose the cutoffs in days before Election Day. Only
contributions received at or before this cutoff will be
aggregated into each candidate×cycle observation.
"""

cutoffs = [360, 240, 120, 60, 30, 14, 7, 1]
print("Established cutoffs.")

# Aggregate and save as Parquet
for X in cutoffs:
    print(f"Aggregating for {X} days before election...")
    
    con.execute(f"""
        CREATE OR REPLACE TABLE agg_rid_{X}d AS
        SELECT
          bonica_rid,
          cycle,
          COUNT(*)                                         AS n_contribs_{X}d,
          AVG(tx_type_freq)                                AS avg_tx_freq_{X}d,
          SUM((1 - contrib_type) * amount) / 1e6           AS indiv_mill_{X}d,
          SUM(contrib_type * (1 - is_corp) * amount) / 1e6 AS comm_mill_{X}d,
          SUM(contrib_type * is_corp * amount) / 1e6       AS corp_mill_{X}d,
          MAX(num_givers)                                  AS num_givers,
          MAX(total_receipts)                              AS total_receipts,
          MAX(total_disbursements)                         AS total_disbursements,
          MAX(total_indiv_contribs)                        AS total_indiv_contribs,
          MAX(total_unitemized)                            AS total_unitemized,
          MAX(total_pac_contribs)                          AS total_pac_contribs,
          MAX(total_party_contribs)                        AS total_party_contribs,
          MAX(total_contribs_from_candidate)               AS total_self_contribs,
          MAX(ind_exp_support)                             AS ind_exp_support,
          MAX(ind_exp_oppose)                              AS ind_exp_oppose,
          MAX(gen_vote_pct)                                AS gen_vote_pct,
          MAX(district_pres_vs)                            AS pres_margin,
          MAX(CASE WHEN party = 100  THEN 1 ELSE 0 END)       AS party_D,
          MAX(CASE WHEN party = 200 THEN 1 ELSE 0 END)       AS party_R,
          MAX(CASE WHEN party NOT IN (100, 200) THEN 1 ELSE 0 END)
                                                           AS party_Other,
          MAX(is_incumbent)                                AS incumbent,
          MAX(gwinner)                                     AS won_general
        FROM house
        WHERE days_before >= {X}
        GROUP BY bonica_rid, cycle;
    """
    )
    
    # Save to parquet
    out_path = output_dir / f"agg_rid_{X}d.parquet"  # ← adjusted
    con.execute(f"COPY agg_rid_{X}d TO '{out_path}' (FORMAT 'parquet');")
    print(f"Saved: {out_path}")

# Close the connection
con.close()
print("Finished aggregating and saving all tables.")

