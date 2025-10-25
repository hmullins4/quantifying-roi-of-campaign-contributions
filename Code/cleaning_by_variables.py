# Script: cleaning_by_variables.py
# Author: Hope E. Mullins

import shutil
import duckdb
from pathlib import Path

# Base directories (relative to this script)
base_dir = Path(__file__).resolve().parent.parent  # Capstone/
src_db   = base_dir / "Data" / "dime_house.duckdb"
dst_db   = base_dir / "Data" / "dime_house_clean.duckdb"

# 1. Copy the DuckDB file
shutil.copy(src_db, dst_db)
print(f"Copied database to {dst_db}")

# 2. Connect to the *copy* for your edits
con = duckdb.connect(str(dst_db))

# Observe variables within the candidate table
cols_cand = con.table('candDB_house').columns
print("candDB_house columns:", cols_cand)
print(f"candDB_house has {len(cols_cand)} columns")


""" Within this table, there are a variety of variables that
simply introduce clutter; many of the variables are identifiers,
metadata, redundant, or advanced ideological scores that are not
related to the analysis of this project. Identifiers/names are 
not predictors of votes, and NIMSP (National Institute for Money
in State Politics) come from state-level sources when I've already
filtered to the house level. Ideological scores, though interesting,
will not affect a candidate's probability of winning. Additionally,
metadata from the other tables would be useful for the source mix, 
but in this table it does not serve me. 

Variables that are especially or may be relevant:
    
    cycle --- Election cycle (year)
    bonica_rid --- Candidate ID to join
    bonica_cid --- Candidate ID to join
    party --- Party code (for party fixed-effects)
    district --- Congressional district (for fixed effects and/or
                                          clustering)
    ico_status --- Incumbency status
    num_givers --- Number of distinct donors
    total_receipts --- Total dollar amount raised
    total_disbursements --- Total spent
    total_indiv_contribs --- Amount from individuals
    total_unitemized --- Small gifts (usually "micro" donors)
    total_pac_contribs --- Amount from PACs
    total_party_contribs --- Amount from party committees
    total_contribs_from_candidate --- Candidate self-financing
    ind_exp_support --- Independent expenditures for candidate
    ind_exp_oppose --- Independent expenditures against candidate
    gen_vote_pct --- General-election vote share (continuous outcome)
    gwinner --- General-election win indicator (binary outcome)
    district_pres_vs --- District-level precentage of the two-party
                            vote share won by Democratic presidential
                            nominee in most recent/concurrent presi-
                            dential election
    
Everything else may be dropped, as is completed in the following code.
The index that was previously created needs to be removed to do so.
    
"""

con.execute("DROP INDEX IF EXISTS idx_cand_rid")

drop = [
    'election', 'fecyear', 'name', 'lname', 'ffname', 'fname', 'mname',
    'title', 'suffix', 'state', 'seat', 'distcyc', 'cand_gender',
    'recipient_cfscore', 'recipient_cfscore_dyn', 'contributor_cfscore',
    'dwdime', 'dwnom1', 'dwnom2', 'ps_dwnom1', 'ps_dwnom2', 'irt_cfscore',
    'composite_score', 'num_givers_total', 's_elec_stat', 'r_elec_stat',
    'fec_cand_status', 'recipient_type', 'igcat', 'comtype', 'ICPSR',
    'ICPSR2', 'Cand_ID', 'FEC_ID', 'NID', 'before_switch_ICPSR',
    'after_switch_ICPSR', 'party_orig', 'nimsp_party', 'nimsp_candidate_ICO_code',
    'nimsp_district', 'nimsp_office', 'nimsp_candidate_status',
    'prim_vote_pct', 'pwinner'
]

for col in drop:
    con.execute(f"ALTER TABLE candDB_house DROP COLUMN {col}")


cols_cand = con.table('candDB_house').columns
print("candDB_house columns:", cols_cand)
print(f"candDB_house has {len(cols_cand)} columns")

con.commit()

# Next, let's examine the contributions table

cols_contrib = con.table('contribDB_house').columns
print("contribDB_house columns:", cols_contrib)
print(f"contribDB_house has {len(cols_contrib)} columns")


""" The majority of variables within the contribDB_house table describe
the contributor (last name, first name, occupation, employer, etc.),
but much of that information is unnecessary for this analysis. There
are also geographic (latitude, longitude, etc.) and financial
(efec_memo, bk_ref_transaction_id, efec_form_type, etc.) variables that
do not help answer the proposed questions. 

Variables that are relevant:
    
    cycle --- Validates the election year in the candidate table
    transaction_type --- FEC code for transaction type
    date --- Exact transaction date
    amount --- Dollar value of each transaction (raw investment data)
    bonica_rid --- Candidate ID to join
    bonica_cid --- Candidate ID to join
    contributor_type --- Individual vs. committee or organization
    is_corp --- Indicates whether contribution is made by a corporate
                entity or q trade organization (for committees)
    election_type --- Primary vs. general elections
    
Everything else may be dropped safely.

"""

con.execute("DROP INDEX IF EXISTS idx_contrib_rid")


drop = [
    'transaction_id', 'contributor_name', 'contributor_lname', 
    'contributor_fname', 'contributor_mname', 'contributor_suffix',
    'contributor_title', 'contributor_ffname', 'contributor_gender',
    'contributor_address', 'contributor_city', 'contributor_state',
    'contributor_zipcode', 'contributor_occupation', 
    'contributor_employer', 'occ_standardized', 'recipient_name',
    'recipient_party', 'recipient_type', 'recipient_state', 'seat',
    'latitude', 'longitude', 'gis_confidence', 'contributor_district',
    'censustract', 'efec_memo', 'efec_memo2', 'efec_transaction_id_orig',
    'bk_ref_transaction_id', 'efec_org_orig', 'efec_comid_orig',
    'efec_form_type', 'excluded_from_scaling', 'contributor_cfscore',
    'candidate_cfscore'
]


for col in drop:
    con.execute(f"ALTER TABLE contribDB_house DROP COLUMN {col}")

cols_contrib = con.table('contribDB_house').columns
print("contribDB_house columns:", cols_contrib)
print(f"contribDB_house has {len(cols_contrib)} columns")

con.commit()


# Finally, look at the variables within the donor/contributor table

cols_donor = con.table('donorDB_house').columns
print("donorDB_house columns:", cols_donor)
print(f"donorDB_house has {len(cols_donor)} columns")

""" None of these variables are needed for my analysis. These variables
are more related to specific donors and not their types; contributor_type
is already found in the contributions table, as is bonica_cid and is_corp.
This table does give the total amount given within a particular election
cycle by a donor, but we already have the amount variable in the contri-
butions table as well. Therefore, none of the variables within this table
are necessary. I will simply drop this table from this database.
"""

con.execute("DROP TABLE IF EXISTS donorDB_house;")

# Verify it’s gone
tables = con.execute("SHOW TABLES").fetchall()
print("Remaining tables:", [t[0] for t in tables])

# Close the connection
con.close()
