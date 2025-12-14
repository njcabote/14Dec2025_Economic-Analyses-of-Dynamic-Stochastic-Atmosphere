"""
Create Figures 4, 5, and 6: Weather sensitivity - DCA>CA dominance probabilities
under normal versus harvest heat-stress conditions by month

- Figure 4: Gala Orchard 1
- Figure 5: Honeycrisp Orchard 1
- Figure 6: Honeycrisp Orchard 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config

# ============================================================================
# STYLE SETTINGS - Use config colors
# ============================================================================
PROFESSIONAL_COLORS = config.PROFESSIONAL_COLORS

# Use config matplotlib settings
plt.rcParams.update(config.MATPLOTLIB_SETTINGS)

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading simulation results...")

# Find latest timestamped output directory
from ._helpers import find_latest_timestamped_output
timestamp_dir, sim_path, figures_path, tables_path, timestamp = find_latest_timestamped_output()

if timestamp:
    print(f"Using latest simulation results from: {sim_path} (timestamp: {timestamp})")
else:
    print(f"Using simulation results from: {sim_path}")

# Load dominance probabilities
hc_file = sim_path / "dominance_probabilities_Honeycrisp_orchard_year_weather.csv"
ga_file = sim_path / "dominance_probabilities_Gala_orchard_year_weather.csv"

if not hc_file.exists() or not ga_file.exists():
    raise FileNotFoundError(
        f"Simulation results not found. Expected files:\n"
        f"  {hc_file}\n"
        f"  {ga_file}\n"
        f"\nPlease run the simulation first: python 02_simulation/run_monte_carlo_simulation.py"
    )

hc_dom = pd.read_csv(hc_file)
ga_dom = pd.read_csv(ga_file)

print(f"✓ Loaded: {len(hc_dom)} Honeycrisp scenarios, {len(ga_dom)} Gala scenarios")

# Month ordering - Use config months
months = config.MONTHS
month_order = {m: i for i, m in enumerate(months)}

# ============================================================================
# CREATE FIGURES
# ============================================================================
# Use timestamped output directory (same timestamp as simulation)
output_dir = figures_path
os.makedirs(output_dir, exist_ok=True)

def create_weather_sensitivity_figure(df, orchard_id, figure_num, orchard_name):
    """Create weather sensitivity figure comparing normal vs harvest heat conditions"""
    
    # Filter data for this orchard
    orchard_data = df[df['orchard_id'] == orchard_id].copy()
    
    if len(orchard_data) == 0:
        print(f"⚠️  No data found for {orchard_id}")
        return None
    
    # Check if harvest_heat column exists
    if 'harvest_heat' not in orchard_data.columns:
        print(f"⚠️  harvest_heat column not found for {orchard_id}")
        return None
    
    # Add month number for sorting
    orchard_data['month_num'] = orchard_data['month'].map(month_order)
    orchard_data = orchard_data.sort_values(['harvest_heat', 'month_num'])
    
    # Separate by harvest heat condition
    with_heat = orchard_data[orchard_data['harvest_heat'] == 1].copy()
    without_heat = orchard_data[orchard_data['harvest_heat'] == 0].copy()
    
    print(f"\n  {orchard_id}:")
    print(f"    With harvest heat: {len(with_heat)} observations")
    print(f"    Without harvest heat: {len(without_heat)} observations")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate mean probabilities by month for each condition
    with_heat_means = []
    without_heat_means = []
    
    for month in months:
        # With heat
        month_with = with_heat[with_heat['month'] == month]
        if len(month_with) > 0:
            with_heat_means.append(month_with['Pr[DCA>CA]'].mean())
        else:
            with_heat_means.append(np.nan)
        
        # Without heat
        month_without = without_heat[without_heat['month'] == month]
        if len(month_without) > 0:
            without_heat_means.append(month_without['Pr[DCA>CA]'].mean())
        else:
            without_heat_means.append(np.nan)
    
    # Bar positions
    x_pos = np.arange(len(months))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x_pos - width/2, without_heat_means, width,
                  label='Without harvest heat',
                  color=PROFESSIONAL_COLORS['without_heat'],
                  alpha=0.9,
                  edgecolor=PROFESSIONAL_COLORS['edge_white'],
                  linewidth=1.2)
    
    bars2 = ax.bar(x_pos + width/2, with_heat_means, width,
                  label='With harvest heat',
                  color=PROFESSIONAL_COLORS['with_heat'],
                  alpha=0.9,
                  edgecolor=PROFESSIONAL_COLORS['edge_white'],
                  linewidth=1.2)
    
    # Formatting
    ax.set_xlabel('Month', fontsize=12, fontweight='normal')
    ax.set_ylabel('Probability that DCA revenues exceed CA revenues', fontsize=12, fontweight='normal')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m[:3] for m in months], rotation=45, ha='right', fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.5', '0.6', '0.8', '1.0'], fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Add 0.5 reference line (no label)
    ax.axhline(y=0.5, color=PROFESSIONAL_COLORS['reference_line'], 
              linestyle='--', linewidth=1.2, alpha=0.6, zorder=0)
    
    # Save
    output_file = output_dir / f"Figure_{figure_num}_{orchard_id}_Weather_Sensitivity.png"
    fig.savefig(str(output_file), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ Saved Figure {figure_num}: {output_file}")
    return output_file

# Create Figure 4: Gala Orchard 1
print("\nCreating Figure 4: Gala Orchard 1...")
create_weather_sensitivity_figure(ga_dom, 'GA_O1', 4, 'Gala Orchard 1')

# Create Figure 5: Honeycrisp Orchard 1
print("\nCreating Figure 5: Honeycrisp Orchard 1...")
create_weather_sensitivity_figure(hc_dom, 'HC_O1', 5, 'Honeycrisp Orchard 1 (Othello, Adams County)')

# Create Figure 6: Honeycrisp Orchard 2
print("\nCreating Figure 6: Honeycrisp Orchard 2...")
create_weather_sensitivity_figure(hc_dom, 'HC_O2', 6, 'Honeycrisp Orchard 2 (Quincy, Grant County)')

print("\n✓ All three weather sensitivity figures created successfully!")
print(f"\nOutput location: {output_dir}/")
