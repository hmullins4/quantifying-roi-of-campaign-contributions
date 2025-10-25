# Script: cleaning_nulls.py
# Author: Hope E. Mullins

import duckdb
from pathlib import Path

""" In this script, I will work with the dime_house_clean.duckdb
database, cleaning the data further so that NULL observations do
not hurt the modeling process.
"""

# Base directories (relative to this script)
base_dir = Path(__file__).resolve().parent.parent  # Capstone/
db_path  = base_dir / "Data" / "dime_house_clean.duckdb"  #

# First, connect to the database
con = duckdb.connect(str(db_path))

# Within the candidate table, explore the type of each variable
print("Column types in candDB_house:")
cand_info = con.execute("PRAGMA table_info('candDB_house')").fetchdf()
print(cand_info[['name', 'type']])

# Next, explore the number of unique values
print("\nUnique value counts in candDB_house:")
cand_columns = con.execute("SELECT * FROM candDB_house LIMIT 1").fetchdf().columns
for col in cand_columns:
    result = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM candDB_house").fetchone()
    print(f"{col}: {result[0]}")

# Within the contributions table, explore the type of each variable
print("\nColumn types in contribDB_house:")
contrib_info = con.execute("PRAGMA table_info('contribDB_house')").fetchdf()
print(contrib_info[['name', 'type']])

# Next, explore the number of unique values
print("\nUnique value counts in contribDB_house:")
contrib_columns = con.execute("SELECT * FROM contribDB_house LIMIT 1").fetchdf().columns
for col in contrib_columns:
    result = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM contribDB_house").fetchone()
    print(f"{col}: {result[0]}")


""" Before cleaning everything so it is numeric and ready for
modeling, I shall join both tables on their matching values.
As described before, the candidate table (candDB_house) has
23,379 rows while the contributions table (contribDB_house)
has about 98 million rows.
"""


# Normalize and cast contrib fields
con.execute("""
  CREATE OR REPLACE VIEW contrib_cast AS
  SELECT
    CAST(cycle       AS INTEGER) AS cycle,
    transaction_type,
    CAST(amount      AS DOUBLE)  AS amount,
    CAST(date        AS DATE)    AS date,
    CAST(bonica_cid  AS BIGINT)  AS bonica_cid, 
    contributor_type,
    is_corp,
    LOWER(TRIM(bonica_rid))       AS bonica_rid,  
    election_type
  FROM contribDB_house
""")

# Normalize and cast candidate fields
con.execute("""
  CREATE OR REPLACE VIEW cand_cast AS
  SELECT
    CAST(cycle      AS INTEGER)    AS cycle,
    LOWER(TRIM(bonica_rid))        AS bonica_rid,
    CAST(bonica_cid AS BIGINT)     AS bonica_cid,
    party,
    district,
    ico_status,
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
    gwinner,
    district_pres_vs
  FROM candDB_house
""")

# Materialize join on cycle + bonica_rid only
con.execute("""
  CREATE OR REPLACE TABLE house_joined AS
  SELECT
    cc.*,
    c.party,
    c.district,
    c.ico_status,
    c.num_givers,
    c.total_receipts,
    c.total_disbursements,
    c.total_indiv_contribs,
    c.total_unitemized,
    c.total_pac_contribs,
    c.total_party_contribs,
    c.total_contribs_from_candidate,
    c.ind_exp_support,
    c.ind_exp_oppose,
    c.gen_vote_pct,
    c.gwinner,
    c.district_pres_vs
  FROM contrib_cast AS cc
  INNER JOIN cand_cast AS c
    USING (cycle, bonica_rid)
""")

# Sanity‐check row count
joined_count = con.execute("SELECT COUNT(*) FROM house_joined").fetchone()[0]
print(f"Rows in joined table: {joined_count:,}")


""" To ensure I am working with clean data, I need to observe the
NULL count within each column. This will give me an idea of what
rows I could drop to bring down the size of my data.
"""

# Fetch all column names from house_joined
cols = con.execute("""
    PRAGMA table_info('house_joined')
""").fetchdf()['name'].tolist()

# Build a null‑count expression for each column
exprs = ",\n  ".join(
    f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END) AS \"{col}_nulls\""
    for col in cols
)

# Run one big SQL that returns total_rows + null count per column
sql = f"""
SELECT
  COUNT(*) AS total_rows,
  {exprs}
FROM house_joined
"""
null_counts = con.execute(sql).fetchdf().T

# Pretty-print
null_counts.columns = ['value']
print(null_counts)


""" Just because a value is NULL does not necessarily mean it should
be removed; therefore, I must be careful with what exactly I decide
to eliminate. 

The number of transaction_type NULLs, date NULLs, and contributor_type
NULLs are all relatively small, so it may be okay to get rid of them. 
According to the DIME codebook, is_corp "[i]ndicates whether the
contribution is made by a corporate entity or q trade organization
(only applies to committees). Takes on the value ’corp’ for corporations 
and trade organizations and is blank otherwise." Therefore, removing
this would be problematic --- Removing the NULLs would also remove
the majority of the data.

The variable election_type, according to the codebook, defines whether
the election is primary or general. Before determining if the observations
where this variable is NULL should be eliminated, let's look at the
unique values and their counts.
"""

unique_types = con.execute("""
    SELECT DISTINCT election_type
    FROM house_joined
""").fetchdf()
print("Unique election_type values:")
print(unique_types)

# Count how many contributions of each type:
type_counts = con.execute("""
    SELECT election_type,
           COUNT(*) AS cnt
    FROM house_joined
    GROUP BY election_type
    ORDER BY cnt DESC
""").fetchdf()
print("\nContributions per election_type:")
print(type_counts)


""" From this, we see that there are still a great number of general
elections to work with if we remove the NULl values of the election_type
variable. Thus, let's start by getting rid of these. Additionally, we
can get rid of any elections that are not general.
"""

# Filtered table where election_type is general
con.execute("""
  CREATE OR REPLACE TABLE house_joined AS
  SELECT *
  FROM house_joined
  WHERE election_type = 'G'
""")

# Verify how many rows were kept
kept = con.execute("SELECT COUNT(*) FROM house_joined").fetchone()[0]
dropped = joined_count - kept
print(f"Rows originally: {joined_count:,}")
print(f"Rows after keeping only general elections: {kept:,} (dropped {dropped:,})")


""" So, now I am dealing with 13,933,493 observations. Let's check
the NULL count again.
"""

# Fetch all column names from house_joined
cols = con.execute("""
    PRAGMA table_info('house_joined')
""").fetchdf()['name'].tolist()

# Build a null‑count expression for each column
exprs = ",\n  ".join(
    f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END) AS \"{col}_nulls\""
    for col in cols
)

# Run one big SQL that returns total_rows + null count per column
sql = f"""
SELECT
  COUNT(*) AS total_rows,
  {exprs}
FROM house_joined
"""
null_counts = con.execute(sql).fetchdf().T

# Pretty-print
null_counts.columns = ['value']
print(null_counts)


""" The number of NULL values existing within other columns has,
unsurprisingly, gone down. Focusing on gen_vote_pct and gwinner,
I see that the number of NULL values for the first column is
4,461,587 and for the latter column is 4,416,147.

The gen_vote_pct and gwinner variable should be populated for all
congressional general-election candidates; it would most likely be
NULL if it isn't a federal general election (e.g. it is a state-
level or special-election row). Therefore, it would not hurt to
get rid of the NULL observations for those columns.
"""

# Filtered table without NULL for gen_vote_pct and gwinner
con.execute("""
  CREATE OR REPLACE TABLE house_joined AS
  SELECT *
  FROM house_joined
  WHERE gwinner IS NOT NULL
  AND gen_vote_pct IS NOT NULL
""")

# Verify how many rows were kept
kept = con.execute("SELECT COUNT(*) FROM house_joined").fetchone()[0]
dropped = joined_count - kept
print(f"Rows originally: {joined_count:,}")
print(f"Rows after dropping NULL gwinner and gen_vote_pct: {kept:,} (dropped {dropped:,})")

""" So now I am dealing with 9,447,843 rows. Let's observe the null
count again.
"""

# Fetch all column names from house_joined
cols = con.execute("""
    PRAGMA table_info('house_joined')
""").fetchdf()['name'].tolist()

# Build a null‑count expression for each column
exprs = ",\n  ".join(
    f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END) AS \"{col}_nulls\""
    for col in cols
)

# Run one big SQL that returns total_rows + null count per column
sql = f"""
SELECT
  COUNT(*) AS total_rows,
  {exprs}
FROM house_joined
"""
null_counts = con.execute(sql).fetchdf().T

# Pretty-print
null_counts.columns = ['value']
print(null_counts)


""" Looking at the other variables (excluding is_corp), I see the
number of NULL values is relatively small compared to the large
dataset. transaction_type has 204 NULLs, contributor_type 310, and
district_pres_vs 16,889. I shall drop each observation where these
variables are NULL.
"""

# Filtered table without NULL for everything but is_corp
con.execute("""
  CREATE OR REPLACE TABLE house_joined AS
  SELECT *
  FROM house_joined
  WHERE transaction_type IS NOT NULL
  AND date IS NOT NULL
  AND contributor_type IS NOT NULL
  AND district_pres_vs IS NOT NULL
""")

# Verify how many rows were kept
kept = con.execute("SELECT COUNT(*) FROM house_joined").fetchone()[0]
dropped = joined_count - kept
print(f"Rows originally: {joined_count:,}")
print(f"Rows after dropping NULLs besides is_corp: {kept:,} (dropped {dropped:,})")

# So now I am left with 9.430,440 observations.


# Check on null count

# Fetch all column names from house_joined
cols = con.execute("""
    PRAGMA table_info('house_joined')
""").fetchdf()['name'].tolist()

# Build a null‑count expression for each column
exprs = ",\n  ".join(
    f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END) AS \"{col}_nulls\""
    for col in cols
)

# Run one big SQL that returns total_rows + null count per column
sql = f"""
SELECT
  COUNT(*) AS total_rows,
  {exprs}
FROM house_joined
"""
null_counts = con.execute(sql).fetchdf().T

# Pretty-print
null_counts.columns = ['value']
print(null_counts)


""" Finally, I need to deal with the NULL values within the is_corp
variable. This is pretty straightforward: I will simply convert all
NULL values to 0 and the value 'corp' to 1.
"""

con.execute("""
CREATE OR REPLACE TABLE house_joined AS
SELECT
  * EXCLUDE(is_corp),
  CASE 
    WHEN is_corp IS NULL THEN 0
    ELSE 1
  END AS is_corp
FROM house_joined;
""")


# This concludes the cleaning of NULL values.
con.close()
