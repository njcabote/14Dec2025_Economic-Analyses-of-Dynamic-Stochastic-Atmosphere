# DCA Economic Analysis - Replication Package

This package provides a complete, modular replication of the Dynamic Controlled Atmosphere (DCA) storage technology economic analysis published in HortScience. The analysis compares three storage technologies (Regular Atmosphere/RA, Controlled Atmosphere/CA, and Dynamic Controlled Atmosphere/DCA) for apple storage using Monte Carlo simulation and decision analysis frameworks.

## ✅ Validation Status

**✅ VALIDATED:** This package has been tested and produces **identical results** to the December 13, 2025 original script (`13Dec2025 DCA Simulations for HortScience.py`). 

**Validation Results:**
- ✅ All key metrics match exactly (0.00% difference)
- ✅ Uses updated weather assumptions (granular period-specific flags)
- ✅ Uses updated energy cost assumptions (component-based model)
- ✅ All 108 scenarios (36 Gala + 72 Honeycrisp) produce identical results

See `FINAL_VALIDATION_REPORT.md` for complete validation details.

## Package Structure

```
DCA_Replication_Package_Final/
├── README.md                          # This file
├── QUICK_START.md                     # Step-by-step execution guide
├── FINAL_VALIDATION_REPORT.md         # Validation results
├── config/
│   └── config.py                     # Centralized configuration parameters
├── data/                              # Input data files
│   ├── packout_data.xlsx             # Replicate-level marketable fruit data
│   └── weather_assumptions.csv       # Weather flags by orchard×year
├── 01_data_preparation/              # Data loading and validation
│   └── load_and_validate_data.py
├── 02_simulation/                     # Monte Carlo simulation
│   └── run_monte_carlo_simulation.py
├── 04_visualization/                  # Figure generation scripts
│   ├── _helpers.py                    # Helper functions for finding outputs
│   ├── create_figures_1_2_3.py       # Main manuscript figures (DCA vs CA)
│   ├── create_figures_4_5_6.py       # Weather sensitivity figures
│   ├── create_figures_4_5_6_lines.py # Weather sensitivity line graphs
│   └── create_figures_1_2_3_CA_vs_RA.py # CA vs RA figures
├── 05_tables/                         # Table generation scripts
│   ├── generate_table_3.py
│   ├── generate_table_6.py
│   └── generate_table_7.py
├── outputs/                           # All generated outputs (timestamped)
│   └── YYYYMMDD_HHMMSS/               # Each run creates a timestamped folder
│       ├── simulation_results/         # Raw simulation outputs (CSV)
│       ├── figures/                   # Publication-ready figures (PNG)
│       └── tables/                    # Formatted tables (STY)
└── validate_replication.py            # Validation script
```

## Quick Start

Execute scripts in numerical order:

```bash
cd DCA_Replication_Package_Final

# 1. (Optional) Validate data
python3 01_data_preparation/load_and_validate_data.py

# 2. Run Monte Carlo simulation (REQUIRED - takes 10-15 minutes)
python3 02_simulation/run_monte_carlo_simulation.py

# 3. Generate figures
python3 04_visualization/create_figures_1_2_3.py
python3 04_visualization/create_figures_4_5_6.py

# 4. Generate tables
python3 05_tables/generate_table_3.py
python3 05_tables/generate_table_6.py
python3 05_tables/generate_table_7.py

# 5. (Optional) Validate replication
python3 validate_replication.py
```

## Key Features

- ✅ **Modular design** - Each step is a separate, well-annotated script
- ✅ **Centralized configuration** - All parameters in `config/config.py`
- ✅ **Updated assumptions** - Uses granular weather flags and component-based cost model
- ✅ **Validated results** - Produces identical results to original December 13 script
- ✅ **Self-contained** - All data files included
- ✅ **Organized outputs** - Timestamped folders for each run (YYYYMMDD_HHMMSS format)

## Output Organization

**Each simulation run creates a timestamped folder:**
- Format: `outputs/YYYYMMDD_HHMMSS/`
- Contains: `simulation_results/`, `figures/`, `tables/`
- Example: `outputs/20251214_094525/`

This ensures:
- ✅ Multiple runs don't overwrite each other
- ✅ Easy tracking of when results were generated
- ✅ Clear organization of outputs by run

## Configuration

All parameters are centralized in `config/config.py`:
- Simulation parameters (iterations, random seed)
- Economic parameters (prices, costs, energy factors)
- Data paths
- Visualization settings

Modify values in `config/config.py` to adjust any aspect of the analysis.

## Output Files

### Simulation Results
- `outputs/YYYYMMDD_HHMMSS/simulation_results/`
  - `dominance_probabilities_Gala_orchard_year_weather.csv`
  - `dominance_probabilities_Honeycrisp_orchard_year_weather.csv`
  - `break_even_delta_*.csv`
  - `beta_fit_summary_by_cell.csv`

### Figures
- `outputs/YYYYMMDD_HHMMSS/figures/`
  - `Figure_1_GA_O1_DCA_vs_CA_Probability.png`
  - `Figure_2_HC_O1_DCA_vs_CA_Probability.png`
  - `Figure_3_HC_O2_DCA_vs_CA_Probability.png`
  - `Figure_4_GA_O1_Weather_Sensitivity.png`
  - `Figure_5_HC_O1_Weather_Sensitivity.png`
  - `Figure_6_HC_O2_Weather_Sensitivity.png`

### Tables
- `outputs/YYYYMMDD_HHMMSS/tables/`
  - `Table_3.sty`
  - `Table_6.sty`
  - `Table_7.sty`

## Validation

The package has been validated against the December 13, 2025 original script:
- ✅ All key metrics match exactly
- ✅ Same parameters (energy costs, weather flags, etc.)
- ✅ Same data files
- ✅ Identical results (within floating-point precision)

Run `python3 validate_replication.py` to verify results match the original.

## Documentation

- `README.md` - This file (package overview)
- `QUICK_START.md` - Detailed step-by-step guide
- `FINAL_VALIDATION_REPORT.md` - Complete validation results
- `VALIDATION_SUMMARY.md` - Quick validation summary

## Citation

[Citation information to be added]

## Contact

nickson.cabote@wsu.edu
