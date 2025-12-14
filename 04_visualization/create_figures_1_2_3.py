"""
Create Figures 1, 2, and 3 as bar charts:
- Figure 1: Probability that DCA revenues exceed CA revenues by month, cultivar, and orchard-year (Gala Orchard 1)
- Figure 2: Probability that DCA revenues exceed CA revenues by month, cultivar, and orchard-year (Honeycrisp Orchard 1)
- Figure 3: Probability that DCA revenues exceed CA revenues by month, cultivar, and orchard-year (Honeycrisp Orchard 2)
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

# Find latest simulation results directory
# Look for the most recent timestamped output directory
output_base = config.OUTPUT_DIR
timestamp_dirs = [d for d in output_base.iterdir() 
                  if d.is_dir() and d.name[0].isdigit() and len(d.name) == 15]  # YYYYMMDD_HHMMSS format

if timestamp_dirs:
    # Get most recent timestamp directory
    latest_timestamp_dir = sorted(timestamp_dirs, key=lambda x: x.stat().st_mtime)[-1]
    sim_path = latest_timestamp_dir / "simulation_results"
    print(f"Using latest simulation results from: {sim_path}")
else:
    # Fallback: check old structure
    sim_results_dir = config.SIMULATION_RESULTS_DIR
    sim_dirs = [d for d in sim_results_dir.iterdir() 
                if d.is_dir() and 'run_monte_carlo_simulation' in d.name]
    if sim_dirs:
        sim_path = sorted(sim_dirs, key=lambda x: x.stat().st_mtime)[-1]
    elif (sim_results_dir / "dominance_probabilities_Gala_orchard_year_weather.csv").exists():
        sim_path = sim_results_dir
    else:
        raise FileNotFoundError(
            f"No simulation results found. Please run simulation first:\n"
            f"  python 02_simulation/run_monte_carlo_simulation.py"
        )
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

# Year colors - professional muted palette
year_colors = {
    2022: PROFESSIONAL_COLORS['year_2022'],
    2023: PROFESSIONAL_COLORS['year_2023'],
    2024: PROFESSIONAL_COLORS['year_2024']
}

# ============================================================================
# CREATE FIGURES
# ============================================================================
# Use timestamped output directory (same timestamp as simulation)
output_dir = figures_path
output_dir.mkdir(parents=True, exist_ok=True)

def create_orchard_bar_chart(df, orchard_id, figure_num, orchard_name):
    """Create bar chart for a specific orchard showing Pr[DCA>CA] by month and year"""
    
    # Filter data for this orchard
    orchard_data = df[df['orchard_id'] == orchard_id].copy()
    
    if len(orchard_data) == 0:
        print(f"⚠️  No data found for {orchard_id}")
        return None
    
    # Add month number for sorting
    orchard_data['month_num'] = orchard_data['month'].map(month_order)
    orchard_data = orchard_data.sort_values(['year', 'month_num'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get years
    years = sorted(orchard_data['year'].unique())
    
    # Bar width and positions
    x_pos = np.arange(len(months))
    width = 0.25
    spacing = 0.02
    
    # Calculate bar positions for each year
    bar_positions = []
    for i, year in enumerate(years):
        offset = (i - len(years)/2 + 0.5) * (width + spacing)
        bar_positions.append(x_pos + offset)
    
    # Plot bars for each year
    bars_list = []
    for i, year in enumerate(years):
        year_data = orchard_data[orchard_data['year'] == year].sort_values('month_num')
        
        # Get values for each month
        values = []
        for month in months:
            month_data = year_data[year_data['month'] == month]
            if len(month_data) > 0:
                values.append(month_data['Pr[DCA>CA]'].iloc[0])
            else:
                values.append(np.nan)
        
        # Plot bars
        bars = ax.bar(bar_positions[i], values, width,
                     label=str(year),
                     color=year_colors.get(year, PROFESSIONAL_COLORS['text_light']),
                     alpha=0.9,
                     edgecolor=PROFESSIONAL_COLORS['edge_white'],
                     linewidth=1.2)
        bars_list.append(bars)
    
    # Formatting
    ax.set_xlabel('Month', fontsize=12, fontweight='normal')
    ax.set_ylabel('Probability that DCA revenues exceed CA revenues', fontsize=12, fontweight='normal')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m[:3] for m in months], rotation=45, ha='right', fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.5', '0.6', '0.8', '1.0'], fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')
    ax.legend(loc='best', fontsize=11, framealpha=0.9, title='Year', title_fontsize=11)
    
    # Add 0.5 reference line (no label)
    ax.axhline(y=0.5, color=PROFESSIONAL_COLORS['reference_line'], 
              linestyle='--', linewidth=1.2, alpha=0.6, zorder=0)
    
    # Save
    output_file = output_dir / f"Figure_{figure_num}_{orchard_id}_DCA_vs_CA_Probability.png"
    fig.savefig(str(output_file), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✓ Saved Figure {figure_num}: {output_file}")
    return output_file

# Create Figure 1: Gala Orchard 1
print("\nCreating Figure 1: Gala Orchard 1...")
create_orchard_bar_chart(ga_dom, 'GA_O1', 1, 'Gala Orchard 1')

# Create Figure 2: Honeycrisp Orchard 1
print("\nCreating Figure 2: Honeycrisp Orchard 1...")
create_orchard_bar_chart(hc_dom, 'HC_O1', 2, 'Honeycrisp Orchard 1 (Othello, Adams County)')

# Create Figure 3: Honeycrisp Orchard 2
print("\nCreating Figure 3: Honeycrisp Orchard 2...")
create_orchard_bar_chart(hc_dom, 'HC_O2', 3, 'Honeycrisp Orchard 2 (Quincy, Grant County)')

print("\n✓ All three figures created successfully!")
print(f"\nOutput location: {output_dir}/")
