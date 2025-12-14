"""
Update Table 3 with latest simulation results (revised energy assumptions)
Table 3: Marketable fruit percentage ranges by storage technology and orchard location
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config

# Load the raw packout data - Use config path
infile = config.PACKOUT_DATA_FILE
# Convert Path to string if needed
infile_str = str(infile) if isinstance(infile, Path) else infile
df = pd.read_excel(infile_str)

def to_frac(x):
    if pd.isna(x):
        return np.nan
    v = float(x)
    return v/100.0 if v > 1.0 else v

df['marketable'] = df['marketable_pct'].apply(to_frac)
df = df[df['interval_months'].isin([3, 6, 9])].copy()
df = df.dropna(subset=['marketable'])

# Pool by orchard, year, technology, interval to get cell-level means
pool_keys = ["cultivar", "orchard_id", "year", "technology", "interval_months"]
cell = (df.groupby(pool_keys, as_index=False)
        .agg(mean_marketable=("marketable", "mean"))
       )

# Convert to percentages
cell['mean_pct'] = cell['mean_marketable'] * 100

# Aggregate across years for each orchard×technology×interval to get IQR
agg_keys = ["cultivar", "orchard_id", "technology", "interval_months"]
table3_data = (cell.groupby(agg_keys, as_index=False)
               .agg(
                   Mean=("mean_pct", "mean"),
                   Q25=("mean_pct", lambda x: x.quantile(0.25)),
                   Q75=("mean_pct", lambda x: x.quantile(0.75))
               ))

# Round values
table3_data['Mean'] = table3_data['Mean'].round(1)
table3_data['Q25'] = table3_data['Q25'].round(1)
table3_data['Q75'] = table3_data['Q75'].round(1)

# Format IQR - match original format (e.g., "99.7-100%")
def format_iqr(q25, q75):
    # Format to match original style
    if q75 >= 100.0:
        return f"{q25:.1f}-100%"
    else:
        return f"{q25:.1f}-{q75:.1f}%"

table3_data['IQR'] = table3_data.apply(lambda row: format_iqr(row['Q25'], row['Q75']), axis=1)

# Generate table in exact format
output_lines = []
output_lines.append("Table 3. Marketable fruit percentage ranges (min-max from 10,000 Monte Carlo simulations, without disorders and decay) by storage technology and orchard location for organic 'Gala' and 'Honeycrisp' apples.")
output_lines.append("Apple ")
output_lines.append("cultivar\tStorage ")
output_lines.append("duration\tRegular ")
output_lines.append("atmosphere\tControlled ")
output_lines.append("atmosphere\tDynamic ")
output_lines.append("controlled ")
output_lines.append("atmosphere")
output_lines.append("\t\tMean\tIQR\tMean\tIQR\tMean\tIQR")

# Process Gala Orchard 1
gala_o1 = table3_data[(table3_data['cultivar'] == 'Gala') & (table3_data['orchard_id'] == 'GA_O1')]
for i, interval in enumerate([3, 6, 9]):
    duration = f"{interval} months + 7 days"
    
    # Get values for each technology
    ra_data = gala_o1[(gala_o1['interval_months'] == interval) & (gala_o1['technology'] == 'RA')]
    ca_data = gala_o1[(gala_o1['interval_months'] == interval) & (gala_o1['technology'] == 'CA')]
    dca_data = gala_o1[(gala_o1['interval_months'] == interval) & (gala_o1['technology'] == 'DCA')]
    
    if i == 0:
        row = f"Gala \nOrchard 1\n \t{duration}"
    else:
        row = f"\t{duration}"
    
    # RA
    if len(ra_data) > 0:
        row += f"\t{ra_data['Mean'].iloc[0]:.1f}\t{ra_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    # CA
    if len(ca_data) > 0:
        row += f"\t{ca_data['Mean'].iloc[0]:.1f}\t{ca_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    # DCA
    if len(dca_data) > 0:
        row += f"\t{dca_data['Mean'].iloc[0]:.1f}\t{dca_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    output_lines.append(row)

# Process Honeycrisp Orchard 1
hc_o1 = table3_data[(table3_data['cultivar'] == 'Honeycrisp') & (table3_data['orchard_id'] == 'HC_O1')]
for i, interval in enumerate([3, 6, 9]):
    duration = f"{interval} months + 7 days"
    
    ra_data = hc_o1[(hc_o1['interval_months'] == interval) & (hc_o1['technology'] == 'RA')]
    ca_data = hc_o1[(hc_o1['interval_months'] == interval) & (hc_o1['technology'] == 'CA')]
    dca_data = hc_o1[(hc_o1['interval_months'] == interval) & (hc_o1['technology'] == 'DCA')]
    
    if i == 0:
        row = f"Honeycrisp\nOrchard 1\n\t{duration}"
    else:
        row = f"\t{duration}"
    
    if len(ra_data) > 0:
        row += f"\t{ra_data['Mean'].iloc[0]:.1f}\t{ra_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    if len(ca_data) > 0:
        row += f"\t{ca_data['Mean'].iloc[0]:.1f}\t{ca_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    if len(dca_data) > 0:
        row += f"\t{dca_data['Mean'].iloc[0]:.1f}\t{dca_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    output_lines.append(row)

# Process Honeycrisp Orchard 2
hc_o2 = table3_data[(table3_data['cultivar'] == 'Honeycrisp') & (table3_data['orchard_id'] == 'HC_O2')]
for i, interval in enumerate([3, 6, 9]):
    duration = f"{interval} months + 7 days"
    
    ra_data = hc_o2[(hc_o2['interval_months'] == interval) & (hc_o2['technology'] == 'RA')]
    ca_data = hc_o2[(hc_o2['interval_months'] == interval) & (hc_o2['technology'] == 'CA')]
    dca_data = hc_o2[(hc_o2['interval_months'] == interval) & (hc_o2['technology'] == 'DCA')]
    
    if i == 0:
        row = f"Honeycrisp\nOrchard 2\t{duration}"
    else:
        row = f"\t{duration}"
    
    if len(ra_data) > 0:
        row += f"\t{ra_data['Mean'].iloc[0]:.1f}\t{ra_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    if len(ca_data) > 0:
        row += f"\t{ca_data['Mean'].iloc[0]:.1f}\t{ca_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    if len(dca_data) > 0:
        row += f"\t{dca_data['Mean'].iloc[0]:.1f}\t{dca_data['IQR'].iloc[0]}"
    else:
        row += "\tN/A\tN/A"
    
    output_lines.append(row)

# Add note
output_lines.append("Note: IQR is the Interquartile range which represents the range between the 25th and 75th percentiles of marketable fruit percentages from Monte Carlo simulations. The range contains 50% of simulation outcomes (25th-75th percentiles), indicating variability around the mean of marketable fruit percentage.")

# Write to file - Use config output directory
# Table 3 doesn't need simulation results, but use timestamped output if available
import sys
from pathlib import Path
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "04_visualization"))
    from _helpers import find_latest_timestamped_output
    _, _, _, tables_path, _ = find_latest_timestamped_output()
except:
    tables_path = config.TABLES_DIR

output_file = tables_path / "Table_3.sty"
os.makedirs(config.TABLES_DIR, exist_ok=True)
with open(output_file, 'w') as f:
    f.write("\n".join(output_lines))

print(f"✓ Updated Table 3: {output_file}")
print("\nTable values updated based on latest simulation with revised energy factors.")
