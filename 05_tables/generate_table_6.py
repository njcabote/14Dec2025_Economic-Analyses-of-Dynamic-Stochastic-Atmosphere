"""
Update Table 6 with latest simulation results (revised energy assumptions)
Table 6: Distribution of net revenue differences (DCA − CA) from Monte Carlo simulation
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config

# Load latest simulation results - Find most recent timestamped output
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "04_visualization"))
from _helpers import find_latest_timestamped_output

timestamp_dir, sim_path, figures_path, tables_path, timestamp = find_latest_timestamped_output()

if timestamp:
    print(f"Using latest simulation results from: {sim_path} (timestamp: {timestamp})")
else:
    print(f"Using simulation results from: {sim_path}")

hc_file = sim_path / "dominance_probabilities_Honeycrisp_orchard_year_weather.csv"
ga_file = sim_path / "dominance_probabilities_Gala_orchard_year_weather.csv"

if not hc_file.exists() or not ga_file.exists():
    raise FileNotFoundError(
        f"Simulation results not found. Please run simulation first:\n"
        f"  python 02_simulation/run_monte_carlo_simulation.py"
    )

hc_dom = pd.read_csv(hc_file)
ga_dom = pd.read_csv(ga_file)

print("=== UPDATING TABLE 6 ===")
print(f"\nLoaded: {len(hc_dom)} Honeycrisp scenarios, {len(ga_dom)} Gala scenarios")

# Peak months: May for Gala, July for Honeycrisp
peak_months = {'Gala': 'May', 'Honeycrisp': 'July'}

# Generate table rows
output_lines = []
output_lines.append("Table 6. Distribution of net revenue differences (DCA − CA) from Monte Carlo simulation, showing median outcomes and uncertainty ranges by cultivar, orchard location, and peak period.")
output_lines.append("Cultivar/Orchard\tPeak Month i\tMedian Revenue Diff\tP10 Revenue Diff ii\tP90 Revenue Diff iii\tProb Positive Diff iv")

# Process Gala (peak month: May)
print("\nProcessing Gala (peak month: May)...")
gala_may = ga_dom[ga_dom['month'] == 'May'].copy()
gala_may = gala_may.sort_values(['orchard_id', 'year'])

for idx, row in gala_may.iterrows():
    orchard_name = "Gala Orchard 1" if row['orchard_id'] == 'GA_O1' else row['orchard_id']
    peak_month_str = f"May {int(row['year'])}"
    
    median_diff = row['DCA-CA_median']
    p10_diff = row['DCA-CA_p10']
    p90_diff = row['DCA-CA_p90']
    prob_positive = row['Pr[DCA>CA]'] * 100  # Convert to percentage
    
    # Format median
    if median_diff < 0:
        median_str = f"-${abs(median_diff):,.0f}"
    else:
        median_str = f"${median_diff:,.0f}"
    
    # Format P10
    if p10_diff < 0:
        p10_str = f"-${abs(p10_diff):,.0f}"
    else:
        p10_str = f"${p10_diff:,.0f}"
    
    # Format P90
    if p90_diff < 0:
        p90_str = f"-${abs(p90_diff):,.0f}"
    else:
        p90_str = f"${p90_diff:,.0f}"
    
    if idx == gala_may.index[0]:
        row_line = f"{orchard_name}\t{peak_month_str}\t{median_str}\t{p10_str}\t{p90_str}\t{prob_positive:.0f}%"
    else:
        row_line = f"\t{peak_month_str}\t{median_str}\t{p10_str}\t{p90_str}\t{prob_positive:.0f}%"
    
    output_lines.append(row_line)
    print(f"  {orchard_name} {peak_month_str}: Median={median_str}, P10={p10_str}, P90={p90_str}, Prob={prob_positive:.0f}%")

# Process Honeycrisp Orchard 1 (peak month: July)
print("\nProcessing Honeycrisp Orchard 1 (peak month: July)...")
hc_o1_july = hc_dom[(hc_dom['month'] == 'July') & (hc_dom['orchard_id'] == 'HC_O1')].copy()
hc_o1_july = hc_o1_july.sort_values('year')

for idx, row in hc_o1_july.iterrows():
    peak_month_str = f"July {int(row['year'])}"
    
    median_diff = row['DCA-CA_median']
    p10_diff = row['DCA-CA_p10']
    p90_diff = row['DCA-CA_p90']
    prob_positive = row['Pr[DCA>CA]'] * 100
    
    # Format values
    if median_diff < 0:
        median_str = f"-${abs(median_diff):,.0f}"
    else:
        median_str = f"${median_diff:,.0f}"
    
    if p10_diff < 0:
        p10_str = f"-${abs(p10_diff):,.0f}"
    else:
        p10_str = f"${p10_diff:,.0f}"
    
    if p90_diff < 0:
        p90_str = f"-${abs(p90_diff):,.0f}"
    else:
        p90_str = f"${p90_diff:,.0f}"
    
    if idx == hc_o1_july.index[0]:
        row_line = f"Honeycrisp Orchard 1 \t{peak_month_str}\t{median_str}\t{p10_str}\t{p90_str}\t{prob_positive:.0f}%"
    else:
        row_line = f"\t{peak_month_str}\t{median_str}\t{p10_str}\t{p90_str}\t{prob_positive:.0f}%"
    
    output_lines.append(row_line)
    print(f"  HC_O1 {peak_month_str}: Median={median_str}, P10={p10_str}, P90={p90_str}, Prob={prob_positive:.0f}%")

# Process Honeycrisp Orchard 2 (peak month: July)
print("\nProcessing Honeycrisp Orchard 2 (peak month: July)...")
hc_o2_july = hc_dom[(hc_dom['month'] == 'July') & (hc_dom['orchard_id'] == 'HC_O2')].copy()
hc_o2_july = hc_o2_july.sort_values('year')

for idx, row in hc_o2_july.iterrows():
    peak_month_str = f"July {int(row['year'])}"
    
    median_diff = row['DCA-CA_median']
    p10_diff = row['DCA-CA_p10']
    p90_diff = row['DCA-CA_p90']
    prob_positive = row['Pr[DCA>CA]'] * 100
    
    # Format values
    if median_diff < 0:
        median_str = f"-${abs(median_diff):,.0f}"
    else:
        median_str = f"${median_diff:,.0f}"
    
    if p10_diff < 0:
        p10_str = f"-${abs(p10_diff):,.0f}"
    else:
        p10_str = f"${p10_diff:,.0f}"
    
    if p90_diff < 0:
        p90_str = f"-${abs(p90_diff):,.0f}"
    else:
        p90_str = f"${p90_diff:,.0f}"
    
    if idx == hc_o2_july.index[0]:
        row_line = f"Honeycrisp Orchard 2 \t{peak_month_str}\t{median_str}\t{p10_str}\t{p90_str}\t{prob_positive:.0f}%"
    else:
        row_line = f"\t{peak_month_str}\t{median_str}\t{p10_str}\t{p90_str}\t{prob_positive:.0f}%"
    
    output_lines.append(row_line)
    print(f"  HC_O2 {peak_month_str}: Median={median_str}, P10={p10_str}, P90={p90_str}, Prob={prob_positive:.0f}%")

# Add notes
output_lines.append("Notes: DCA = Dynamic controlled atmosphere; CA = Controlled atmosphere")
output_lines.append("i Peak periods represent months with highest predicted market prices: May for Gala and July for Honeycrisp.")
output_lines.append("ii, iii P10 and P90 percentiles show the 10th (lowest 10 percent) and 90th (highest 10 percent) percentile from 10,000 Monte Carlo simulations, representing uncertainty range.")
output_lines.append("iv Positive probability indicates the percentage of scenarios where DCA generates higher net revenue compared to CA.")

# Write to file - Use config output directory
output_file = tables_path / "Table_6.sty"
os.makedirs(config.TABLES_DIR, exist_ok=True)
with open(output_file, 'w') as f:
    f.write("\n".join(output_lines))

print(f"\n✓ Updated Table 6: {output_file}")
print("\nTable values updated based on latest simulation with revised energy factors.")
