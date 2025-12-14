# Quick Start Guide

This guide provides step-by-step instructions to run the complete DCA economic analysis replication.

## Prerequisites

1. **Python 3.8+** installed
2. **Required packages** installed:
   ```bash
   pip install -r requirements.txt
   ```

## Step-by-Step Execution

### Step 1: Verify Data Files

Ensure all data files are in the `data/` directory:
- `data/packout_data.xlsx` - Marketable fruit percentage data
- `data/weather_assumptions.csv` - Weather flags by orchard×year

### Step 2: Run Data Preparation (Optional)

If you want to validate and prepare data:
```bash
python 01_data_preparation/load_and_validate_data.py
```

This will:
- Load and validate packout data
- Load weather assumptions
- Create cell-level and replicate-level summaries
- Save validated data to `outputs/simulation_results/`

### Step 3: Run Monte Carlo Simulation

**This is the main step that generates all core results:**

```bash
python 02_simulation/run_monte_carlo_simulation.py
```

This script:
- Fits Beta distributions to packout data
- Runs 10,000 Monte Carlo iterations per scenario
- Calculates dominance probabilities (Pr[DCA>CA], Pr[DCA>RA])
- Generates revenue distributions and quantiles
- Merges weather flags with results
- Saves all outputs to `outputs/simulation_results/`

**Expected runtime:** 5-15 minutes depending on your system

### Step 4: Generate Figures

Generate publication-ready figures:

```bash
# Main manuscript figures (DCA vs CA)
python 04_visualization/create_figures_1_2_3.py

# Weather sensitivity figures
python 04_visualization/create_figures_4_5_6.py
```

Figures are saved to `outputs/figures/`

### Step 5: Generate Tables

Generate formatted Excel tables:

```bash
python 05_tables/generate_table_3.py
python 05_tables/generate_table_6.py
python 05_tables/generate_table_7.py
```

Tables are saved to `outputs/tables/`

## Output Organization

All outputs are organized in the `outputs/` directory:

```
outputs/
├── figures/              # Publication-ready figures (PNG, 300 DPI)
├── tables/               # Formatted Excel tables
└── simulation_results/  # Raw simulation outputs (CSV files)
    ├── dominance_probabilities_*.csv
    ├── net_revenue_quantiles_*.csv
    ├── break_even_delta_*.csv
    └── weather_stratified_*.csv
```

## Key Output Files

### Simulation Results
- `dominance_probabilities_Gala_orchard_year_weather.csv` - Gala dominance probabilities
- `dominance_probabilities_Honeycrisp_orchard_year_weather.csv` - Honeycrisp dominance probabilities
- `net_revenue_quantiles_*.csv` - Revenue distribution quantiles
- `break_even_delta_*.csv` - Break-even packout requirements

### Figures
- `Figure_1_GA_O1_DCA_vs_CA_Probability.png` - Gala Orchard 1 bar chart
- `Figure_2_HC_O1_DCA_vs_CA_Probability.png` - Honeycrisp Orchard 1 bar chart
- `Figure_3_HC_O2_DCA_vs_CA_Probability.png` - Honeycrisp Orchard 2 bar chart
- `Figure_4_GA_O1_Weather_Sensitivity.png` - Gala weather sensitivity
- `Figure_5_HC_O1_Weather_Sensitivity.png` - Honeycrisp Orchard 1 weather sensitivity
- `Figure_6_HC_O2_Weather_Sensitivity.png` - Honeycrisp Orchard 2 weather sensitivity

### Tables
- `Table_3_Marketable_Fruit_Percentages.xlsx` - Marketable fruit ranges
- `Table_6_Revenue_Differences.xlsx` - Revenue difference distributions
- `Table_7_Economic_Advantages.xlsx` - Economic advantages and success probabilities

## Troubleshooting

### Common Issues

1. **FileNotFoundError for data files**
   - Ensure data files are in `data/` directory
   - Check file names match exactly (case-sensitive)

2. **Import errors**
   - Run `pip install -r requirements.txt` to install all dependencies
   - Ensure you're using Python 3.8 or higher

3. **Memory errors during simulation**
   - The simulation uses ~2-4 GB RAM
   - Close other applications if needed
   - Consider reducing `MONTE_CARLO_ITERATIONS` in `config/config.py` (default: 10000)

4. **Long runtime**
   - Normal: 5-15 minutes for full simulation
   - If taking >30 minutes, check system resources
   - Reduce iterations in config if needed for testing

## Next Steps

After running the complete analysis:

1. **Review outputs** in `outputs/simulation_results/` for raw data
2. **Check figures** in `outputs/figures/` for visualizations
3. **Verify tables** in `outputs/tables/` match manuscript
4. **Compare results** with published manuscript values

## Customization

To modify analysis parameters, edit `config/config.py`:
- `MONTE_CARLO_ITERATIONS` - Number of MC draws (default: 10000)
- `ENERGY_FACTOR` - Energy multipliers for RA/CA/DCA
- `PRICES` - Monthly price assumptions
- `COST_MODEL_MODE` - "COMPONENT" or "TRIANGULAR"

See `config/config.py` for all available parameters.
