# Script: conversion.py
# Author: Hope E. Mullins

import duckdb
from pathlib import Path

""" In this script, I will work with the dime_house_cleaner.duckdb
database, converting everything to numeric values so they may
be utilized in various models. Additionally, if there are any
variables that may now be dropped that were not initially, I will
drop them here. The monetary variables will all need to be
adjusted for inflation, so I shall use 2024 as a baseline. Finally,
I shall create a few new variables to assist with my goals in this
project.
"""

# Base directories (relative to this script)
base_dir = Path(__file__).resolve().parent.parent  # Capstone/
dst_path = base_dir / "Data" / "dime_house_cleaner.duckdb"
src_path = base_dir / "Data" / "dime_house_clean.duckdb"

# Connect to the target database
con = duckdb.connect(str(dst_path))

# Attach source database
con.execute(f"ATTACH '{src_path}' AS src;")

# Copy necessary table into target
con.execute("""
CREATE OR REPLACE TABLE house AS
SELECT * 
FROM src.house_joined;
""")

# Detach source
con.execute("DETACH src;")
print("Seeded `house` from `src.house_joined`.")

# Within the house table, explore the type of each variable
print("Column types in house:")
house_info = con.execute("PRAGMA table_info('house')").fetchdf()
print(house_info[['name', 'type']])

# Next, explore the number of unique values
print("\nUnique value counts in house:")
house_columns = con.execute("SELECT * FROM house LIMIT 1").fetchdf().columns
for col in house_columns:
    result = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM house").fetchone()
    print(f"{col}: {result[0]}")
    
    
""" Before doing anything else, I think it would be wise to
adjust for inflation up-front. Using the Consumer Price Index
data from the Bureau of Labor Statistics, I shall create a
table that contains CPI-based deflators with 2024 as the
base year. 
"""

# Create deflator table
con.execute("""CREATE OR REPLACE TABLE cpi_deflator (
  year INTEGER,
  deflator_to_2024 DOUBLE
);
""")
print("Created deflator table.")

# Insert the CPI‐based deflators
con.execute("""INSERT INTO cpi_deflator VALUES
  (1998, 313.689/163.0),
  (1999, 313.689/166.6),
  (2000, 313.689/172.2),
  (2001, 313.689/177.1),
  (2002, 313.689/179.9),
  (2003, 313.689/184.0),
  (2004, 313.689/188.9),
  (2005, 313.689/195.3),
  (2006, 313.689/201.6),
  (2007, 313.689/207.342),
  (2008, 313.689/215.303),
  (2009, 313.689/214.537),
  (2010, 313.689/218.056),
  (2011, 313.689/224.939),
  (2012, 313.689/229.594),
  (2013, 313.689/232.957),
  (2014, 313.689/236.736),
  (2015, 313.689/237.017),
  (2016, 313.689/240.007),
  (2017, 313.689/245.120),
  (2018, 313.689/251.107),
  (2019, 313.689/255.657),
  (2020, 313.689/258.811),
  (2021, 313.689/270.970),
  (2022, 313.689/292.655),
  (2023, 313.689/304.702),
  (2024, 313.689/313.689);
""")
print("Inserted values into deflator table.")


""" Now I shall join the deflator onto the house table and multiply
every dollar column by deflator_to_2024 to put all money in 2024 
dollars.
"""

con.execute("""
CREATE OR REPLACE TABLE house AS
SELECT
  h.*,
  d.deflator_to_2024,
  h.amount * d.deflator_to_2024                         AS amount_2024,
  h.total_receipts * d.deflator_to_2024                 AS total_receipts_2024,
  h.total_disbursements * d.deflator_to_2024            AS total_disbursements_2024,
  h.total_indiv_contribs * d.deflator_to_2024           AS total_indiv_contribs_2024,
  h.total_unitemized * d.deflator_to_2024               AS total_unitemized_2024,
  h.total_pac_contribs * d.deflator_to_2024             AS total_pac_contribs_2024,
  h.total_party_contribs * d.deflator_to_2024           AS total_party_contribs_2024,
  h.total_contribs_from_candidate * d.deflator_to_2024  AS total_self_contribs_2024
FROM house AS h
JOIN cpi_deflator AS d
    ON EXTRACT(year FROM h.date) = d.year;
""")
print("Adjusted all monetary variables to 2024 dollars via CPI deflator.")

# Compute days_before election in place
con.execute("""
CREATE OR REPLACE TABLE house AS
WITH election_dates AS (
  SELECT
    cycle,
    CAST(concat(cycle,'-11-01') AS DATE) AS nov1,
    ((9 - extract(dow FROM nov1)) % 7) AS offset_days
  FROM (SELECT DISTINCT cycle FROM house)
),
with_days AS (
  SELECT
    h.*,
    nov1 + offset_days::INTEGER AS election_date
  FROM house AS h
  JOIN election_dates USING(cycle)
),
final AS (
  SELECT
    wd.*,
    wd.election_date - wd.date AS days_before
  FROM with_days AS wd
)
SELECT
  * EXCLUDE(election_date)
FROM final;
""")
print("Added `days_before` column.")


""" With this, it is now possible to encode/transform the columns I have.
Right off the bat, election_type may be dropped because I am only working
with general elections. Moreover, all the original monetary value columns
may be dropped due to the deflation. I shall drop district to follow along
better with Jacobson's research. While district-level hererogeneity is
important to capture, the district variable itself introduces too much
cardinality to be useful. Alternatively, Jacobson used a continuous par-
tisanship measure to control for how Democratic or Republican a district
is without over 400 dummy columns. I could retain the district_pres_vs
to account for this. date may also be dropped because now I will utilize
the days_before variable.

contributor_type measures whether the contributor was an individual or a
committee/organization, so I will simply map individual to 0 and the other
to 1. The party variable contains only 5 unique values, so I would like to
encode this; however, doing so now at the contribution level will collapse
it to a single value per candidate/cycle, so I will save this for during
the aggregation process. The ico_staus contains 4 unique values, so I will
one-hot encode it. In order to avoid 51 dummy variables for the 
transaction_type, I shall use frequency encoding. Finally, the gwinner 
variable, measuring whether a candidate won or lost, is binary; thus, I 
will map "W" to 1 and "L" to 0. 
"""

# Drop unnecessary columns
con.execute("""
CREATE OR REPLACE TABLE house AS
SELECT
  cycle,
  transaction_type,
  amount_2024       AS amount,
  bonica_cid,
  contributor_type,
  is_corp,
  bonica_rid,
  party,
  ico_status,
  num_givers,
  total_receipts_2024       AS total_receipts,
  total_disbursements_2024  AS total_disbursements,
  total_indiv_contribs_2024 AS total_indiv_contribs,
  total_unitemized_2024     AS total_unitemized,
  total_pac_contribs_2024   AS total_pac_contribs,
  total_party_contribs_2024 AS total_party_contribs,
  total_self_contribs_2024  AS total_contribs_from_candidate,
  ind_exp_support,
  ind_exp_oppose,
  gen_vote_pct,
  gwinner,
  district_pres_vs,
  days_before
FROM house;
""")
print("Replaced raw money with inflation‐adjusted values and dropped old columns.")

# Count number of winners vs. losers
query = """
SELECT gwinner, COUNT(*) 
FROM house
GROUP BY gwinner
ORDER BY COUNT(*) DESC
"""

# Run the query
result = con.execute(query).fetchall()

# Print the result
for row in result:
    print(row)

# Encode simple binaries/dummies
con.execute("""
CREATE OR REPLACE TABLE house AS
SELECT
  cycle,
  transaction_type,
  amount,
  bonica_cid,
  CASE contributor_type WHEN 'I' THEN 0 ELSE 1 END AS contrib_type,
  is_corp,
  bonica_rid,
  party,
  CASE ico_status WHEN 'I' THEN 1 ELSE 0 END AS is_incumbent,
  CASE ico_status WHEN 'C' THEN 1 ELSE 0 END AS is_challenger,
  CASE ico_status WHEN 'O' THEN 1 ELSE 0 END AS is_open,
  num_givers,
  total_receipts,
  total_disbursements,
  total_indiv_contribs,
  total_unitemized,
  total_pac_contribs,
  total_party_contribs,
  total_contribs_from_candidate,
  ind_exp_support,
  ind_exp_oppose,
  gen_vote_pct,
  CASE gwinner WHEN 'W' THEN 1 ELSE 0 END AS gwinner,
  district_pres_vs,
  days_before
FROM house;
""")
print("Encoded binaries.")

# Count number of winners vs. losers
query = """
SELECT gwinner, COUNT(*) 
FROM house
GROUP BY gwinner
ORDER BY COUNT(*) DESC
"""

# Run the query
result = con.execute(query).fetchall()

# Print the result
for row in result:
    print(row)


# Frequency‑encode transaction_type
con.execute("""
CREATE OR REPLACE TABLE house AS
WITH
  grp_counts AS (
    SELECT
      bonica_rid,
      cycle,
      transaction_type,
      COUNT(*) AS cnt
    FROM house
    GROUP BY bonica_rid, cycle, transaction_type
  ),
  grp_totals AS (
    SELECT
      bonica_rid,
      cycle,
      COUNT(*) AS total_cnt
    FROM house
    GROUP BY bonica_rid, cycle
  ),
  grp_freq AS (
    SELECT
      g.bonica_rid,
      g.cycle,
      g.transaction_type,
      g.cnt::DOUBLE / t.total_cnt AS tx_type_freq
    FROM grp_counts AS g
    JOIN grp_totals AS t
      USING (bonica_rid, cycle)
  )
SELECT
  h.*,
  gf.tx_type_freq
FROM house AS h
LEFT JOIN grp_freq AS gf
  USING (bonica_rid, cycle, transaction_type)
;
""")
print("Frequency‑encoded `transaction_type` → `tx_type_freq`.")

# Drop redundant columns
con.execute("ALTER TABLE house DROP COLUMN IF EXISTS transaction_type;")
print("Dropped raw `transaction_type`.")
# Drop deflator column
con.execute("ALTER TABLE house DROP COLUMN IF EXISTS deflator_to_2024;")
print("Dropped deflator column.")

# Check
print("Final schema:")
schema_df = con.execute("PRAGMA table_info('house')").fetchdf()[['name','type']]
print(schema_df)

# Final check: unique counts
print("\nFinal unique counts:")
cols = schema_df['name'].tolist()
for c in cols:
    cnt = con.execute(f"SELECT COUNT(DISTINCT \"{c}\") FROM house").fetchone()[0]
    print(f"{c}: {cnt}")

con.close()
print("House table is now ready for aggregation and modeling.")
    
    
