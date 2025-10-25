# File: spending_trend.py
# Author: Hope E. Mullins

import duckdb
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Base directories (relative to this script)
base_dir    = Path(__file__).resolve().parent.parent  
db_path     = base_dir / "Data" / "dime_house_cleaner.duckdb" 
figures_dir = base_dir / "Figures"  

# Ensure output folder exists
figures_dir.mkdir(parents=True, exist_ok=True)

# Connect to database
con = duckdb.connect(str(db_path))  

# Pull aggregated (deflated) totals by cycle
df = con.execute("""
                 SELECT cycle,
                 SUM(amount) AS total_spent
                 FROM house
                 GROUP BY cycle
                 ORDER BY CAST(cycle AS INTEGER)
                 """).fetchdf()
                 
con.close()

# Define formatter
def in_billions(x, pos):
    return f"{x/1e9:g}" 

formatter = FuncFormatter(in_billions)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    df['cycle'],
    df['total_spent'],
    marker='o',
    linewidth=4,                                 
    color='black'
)

ax.yaxis.set_major_formatter(formatter)

ax.tick_params(
    axis = 'both',
    which = 'major',
    width = 2,      
    labelsize = 10
)

ax.set_xlabel("Election Cycle", fontsize = 14)
ax.set_ylabel("Total Contributions (2024 USD Billions)",
              fontsize = 14)

ax.set_xticks(df['cycle'])
ax.set_xticklabels(df['cycle'], rotation = 45)
ax.tick_params(axis = 'y', labelrotation = 45)

plt.tight_layout()
plt.savefig(figures_dir / "cycle_spending_trend.pdf", dpi = 300)
