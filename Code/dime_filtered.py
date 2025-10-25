# Script: dime_filtered.py
# Author: Hope E. Mullins

""" 
The database being utilized in this script is the dime_v4.sqlite3 database, found
at the Database on Ideology, Money in Politics, and Elections website through Stanford
University. In this capstone project, I intend to only utilize data on U.S. House
elections, so through this script, I will filter the data I currently have based on
this restriction and place it in a new database for efficiency and space.
"""

# Import necessary packages
import duckdb
import os

# Set working directory if needed
os.chdir("D:/")

# Paths
sqlite_db_path = "D:/capstone_data/dime_v4.sqlite3"
duckdb_out_path = "D:/capstone_data/dime_house.duckdb"

# Connect to new DuckDB file
con = duckdb.connect(duckdb_out_path)

# Load SQLite extension and attach the DIME database
con.execute("INSTALL sqlite; LOAD sqlite;")
con.execute(f"ATTACH '{sqlite_db_path}' AS sqlite_db (TYPE sqlite)")

# STEP 1: Filter 'candDB' for federal:house between 2000–2024
print("Filtering candDB...")
con.execute("""
    CREATE TABLE candDB_house AS
    SELECT *
    FROM sqlite_db.candDB
    WHERE seat = 'federal:house'
      AND CAST(cycle AS INTEGER) BETWEEN 2000 AND 2024
      AND bonica_rid IS NOT NULL
""")

# STEP 2: Extract distinct bonica_rid values
print("Creating house_rids table...")
con.execute("""
    CREATE TEMP TABLE house_rids AS
    SELECT DISTINCT bonica_rid
    FROM candDB_house
""")

# STEP 3: Filter 'contribDB' by house_rids, seat, and date
print("Filtering contribDB...")
con.execute("""
    CREATE TABLE contribDB_house AS
    SELECT *
    FROM sqlite_db.contribDB
    WHERE bonica_rid IN (SELECT bonica_rid FROM house_rids)
      AND seat = 'federal:house'
      AND CAST (date as DATE)
          BETWEEN DATE('1998-01-01') AND DATE('2024-12-31')
""")

# STEP 4: Extract bonica_cid from filtered contribDB (normalize as integers)
print("Extracting bonica_cid values...")
con.execute("""
    CREATE TEMP TABLE donor_cids AS
    SELECT DISTINCT CAST(CAST(bonica_cid AS BIGINT) AS VARCHAR) AS bonica_cid
    FROM contribDB_house
    WHERE bonica_cid IS NOT NULL
""")

# STEP 5: Filter 'donorDB' using donor_cids (normalize as integers)
print("Filtering donorDB...")
con.execute("""
    CREATE TABLE donorDB_house AS
    SELECT *
    FROM sqlite_db.donorDB
    WHERE CAST(CAST(bonica_cid AS BIGINT) AS VARCHAR) IN (
        SELECT bonica_cid FROM donor_cids
    )
""")

# STEP 6: Add indexes to improve performance (optional but helpful)
print("Creating indexes...")
con.execute("CREATE INDEX IF NOT EXISTS idx_cand_rid ON candDB_house(bonica_rid)")
con.execute("CREATE INDEX IF NOT EXISTS idx_contrib_rid ON contribDB_house(bonica_rid)")
con.execute("CREATE INDEX IF NOT EXISTS idx_donor_cid ON donorDB_house(bonica_cid)")

# Done
print("Filtering complete. Saved to dime_house.duckdb")
con.close()
