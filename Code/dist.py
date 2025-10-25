# File: dist.py
# Author: Hope E. Mullins

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

# Determine project structure
script_dir = Path(__file__).resolve().parent         # e.g., .../Capstone/Code
project_root = script_dir.parent                      # .../Capstone

def main():
    # Build paths
    data_file = project_root / "Data" / "dime_house_cleaner.duckdb"
    fig_dir   = project_root / "Figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file  = fig_dir / "amount_dist.pdf"

    # load data
    con = duckdb.connect(str(data_file))
    df  = con.execute('SELECT * FROM house').df()
    con.close()

    # filter positives
    amounts = df['amount'][df['amount'] > 0]

    # log-spaced bins
    min_edge = amounts.min()
    max_edge = amounts.max()
    bins = np.logspace(np.log10(min_edge), np.log10(max_edge), 50)

    print(f"Min amount: {min_edge}")
    print(f"Max amount: {max_edge}")

    # plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(amounts, bins=bins, density=True,
            color='black', edgecolor='black')
    ax.set_xscale('log')

    # Custom ticks
    ticks = [0.01, 500, 5_000, 100_000, 7_000_000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"${{t:,.0f}}" for t in ticks])

    # Dummy handles for legend
    ax.plot([], [], ' ', label=f"Min amount: ${min_edge:,.2f}")
    ax.plot([], [], ' ', label=f"Max amount: ${max_edge:,.0f}")
    ax.legend(loc='upper right', frameon=True)

    # Disable scientific notation on x-axis
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)

    ax.set_xlabel("Amount (2024 USD)", fontsize = 14)
    ax.set_ylabel("Density", fontsize = 14)

    plt.tight_layout()
    plt.savefig(str(out_file))
    
if __name__ == '__main__':
    main()

