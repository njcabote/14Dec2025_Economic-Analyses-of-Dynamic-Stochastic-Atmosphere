"""
Update Table 7 with latest simulation results (revised energy assumptions)
Table 7: Orchard-Specific DCA Economic Advantages and Success Probabilities by Season and Year
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

print("=== UPDATING TABLE 7 ===")
print(f"\nLoaded: {len(hc_dom)} Honeycrisp scenarios, {len(ga_dom)} Gala scenarios")

# Define seasons
spring_months = ['March', 'April', 'May']
summer_months = ['June', 'July', 'August']

# Generate table rows
output_lines = []
output_lines.append("Table 7. Orchard-Specific DCA Economic Advantages and Success Probabilities by Season and Year")
output_lines.append("Orchard\tYear\tSeason i\tDCA Success")
output_lines.append("Probability ii\tDCA")
output_lines.append("Revenue")
output_lines.append("Advantage iii")

# Process Gala Orchard 1
print("\nProcessing Gala Orchard 1...")
gala_o1 = ga_dom[ga_dom['orchard_id'] == 'GA_O1'].copy()
gala_o1 = gala_o1.sort_values('year')

for year in sorted(gala_o1['year'].unique()):
    year_data = gala_o1[gala_o1['year'] == year]
    
    # Spring (March-May)
    spring_data = year_data[year_data['month'].isin(spring_months)]
    if len(spring_data) > 0:
        spring_prob = spring_data['Pr[DCA>CA]'].mean() * 100
        spring_revenue = spring_data['DCA-CA_median'].median()
        
        if year == sorted(gala_o1['year'].unique())[0]:
            row = f"Gala \nOrchard 1\t{int(year)}\tSpring\t{spring_prob:.1f}%\t"
        else:
            row = f"\t{int(year)}\tSpring\t{spring_prob:.1f}%\t"
        
        if spring_revenue < 0:
            row += f"-${abs(spring_revenue):,.0f}"
        else:
            row += f"${spring_revenue:,.0f}"
        
        output_lines.append(row)
        print(f"  {year} Spring: Prob={spring_prob:.1f}%, Revenue={spring_revenue:,.0f}")
    
    # Summer (June-August)
    summer_data = year_data[year_data['month'].isin(summer_months)]
    if len(summer_data) > 0:
        summer_prob = summer_data['Pr[DCA>CA]'].mean() * 100
        summer_revenue = summer_data['DCA-CA_median'].median()
        
        row = f"\t{int(year)}\tSummer\t{summer_prob:.1f}%\t"
        
        if summer_revenue < 0:
            row += f"-${abs(summer_revenue):,.0f}"
        else:
            row += f"${summer_revenue:,.0f}"
        
        output_lines.append(row)
        print(f"  {year} Summer: Prob={summer_prob:.1f}%, Revenue={summer_revenue:,.0f}")

# Process Honeycrisp Orchard 1
print("\nProcessing Honeycrisp Orchard 1...")
hc_o1 = hc_dom[hc_dom['orchard_id'] == 'HC_O1'].copy()
hc_o1 = hc_o1.sort_values('year')

for year in sorted(hc_o1['year'].unique()):
    year_data = hc_o1[hc_o1['year'] == year]
    
    # Spring (March-May)
    spring_data = year_data[year_data['month'].isin(spring_months)]
    if len(spring_data) > 0:
        spring_prob = spring_data['Pr[DCA>CA]'].mean() * 100
        spring_revenue = spring_data['DCA-CA_median'].median()
        
        if year == sorted(hc_o1['year'].unique())[0]:
            row = f"Honeycrisp \nOrchard 1 \t{int(year)}\tSpring\t{spring_prob:.1f}%\t"
        else:
            row = f"\t{int(year)}\tSpring\t{spring_prob:.1f}%\t"
        
        if spring_revenue < 0:
            row += f"-${abs(spring_revenue):,.0f}"
        else:
            row += f"${spring_revenue:,.0f}"
        
        output_lines.append(row)
        print(f"  {year} Spring: Prob={spring_prob:.1f}%, Revenue={spring_revenue:,.0f}")
    
    # Summer (June-August)
    summer_data = year_data[year_data['month'].isin(summer_months)]
    if len(summer_data) > 0:
        summer_prob = summer_data['Pr[DCA>CA]'].mean() * 100
        summer_revenue = summer_data['DCA-CA_median'].median()
        
        row = f"\t{int(year)}\tSummer\t{summer_prob:.1f}%\t"
        
        if summer_revenue < 0:
            row += f"-${abs(summer_revenue):,.0f}"
        else:
            row += f"${summer_revenue:,.0f}"
        
        output_lines.append(row)
        print(f"  {year} Summer: Prob={summer_prob:.1f}%, Revenue={summer_revenue:,.0f}")

# Process Honeycrisp Orchard 2
print("\nProcessing Honeycrisp Orchard 2...")
hc_o2 = hc_dom[hc_dom['orchard_id'] == 'HC_O2'].copy()
hc_o2 = hc_o2.sort_values('year')

for year in sorted(hc_o2['year'].unique()):
    year_data = hc_o2[hc_o2['year'] == year]
    
    # Spring (March-May)
    spring_data = year_data[year_data['month'].isin(spring_months)]
    if len(spring_data) > 0:
        spring_prob = spring_data['Pr[DCA>CA]'].mean() * 100
        spring_revenue = spring_data['DCA-CA_median'].median()
        
        if year == sorted(hc_o2['year'].unique())[0]:
            row = f"Honeycrisp \nOrchard 2 \t{int(year)}\tSpring\t{spring_prob:.1f}%\t"
        else:
            row = f"\t{int(year)}\tSpring\t{spring_prob:.1f}%\t"
        
        if spring_revenue < 0:
            row += f"-${abs(spring_revenue):,.0f}"
        else:
            row += f"${spring_revenue:,.0f}"
        
        output_lines.append(row)
        print(f"  {year} Spring: Prob={spring_prob:.1f}%, Revenue={spring_revenue:,.0f}")
    
    # Summer (June-August)
    summer_data = year_data[year_data['month'].isin(summer_months)]
    if len(summer_data) > 0:
        summer_prob = summer_data['Pr[DCA>CA]'].mean() * 100
        summer_revenue = summer_data['DCA-CA_median'].median()
        
        row = f"\t{int(year)}\tSummer\t{summer_prob:.1f}%\t"
        
        if summer_revenue < 0:
            row += f"-${abs(summer_revenue):,.0f}"
        else:
            row += f"${summer_revenue:,.0f}"
        
        output_lines.append(row)
        print(f"  {year} Summer: Prob={summer_prob:.1f}%, Revenue={summer_revenue:,.0f}")

# Add notes
output_lines.append("i Storage seasons when Dynamic Controlled Atmosphere (DCA) shows meaningful economic advantages. Spring (March-May) represents the transition period when DCA advantages begin to emerge as storage extends beyond 6 months. Summer (June-August) represents the peak period for DCA economic advantage, when extended storage (7-9 months) maximizes the technology's quality preservation benefits and coincides with higher seasonal prices.")
output_lines.append("ii Percentage chance that DCA outperforms Controlled Atmosphere (CA) storage in revenue generation")
output_lines.append("iii Median additional net revenue per room when using DCA instead of CA storage")

# Write to file - Use config output directory
output_file = tables_path / "Table_7.sty"
os.makedirs(config.TABLES_DIR, exist_ok=True)
with open(output_file, 'w') as f:
    f.write("\n".join(output_lines))

print(f"\n✓ Updated Table 7: {output_file}")
print("\nTable values updated based on latest simulation with revised energy factors.")
