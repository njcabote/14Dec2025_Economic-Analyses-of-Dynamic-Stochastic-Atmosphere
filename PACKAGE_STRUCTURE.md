# Package Structure

## Final Replication Package Organization

```
DCA_Replication_Package_Final/
├── README.md                          # Main documentation
├── QUICK_START.md                     # Step-by-step execution guide
├── FINAL_VALIDATION_REPORT.md         # Validation results
├── VALIDATION_SUMMARY.md              # Quick validation summary
├── PACKAGE_STRUCTURE.md               # This file
│
├── config/                            # Configuration
│   └── config.py                     # All parameters centralized here
│
├── data/                              # Input data (required)
│   ├── packout_data.xlsx             # Replicate-level marketable fruit data
│   └── weather_assumptions.csv       # Weather flags by orchard×year
│
├── 01_data_preparation/              # Data loading and validation
│   └── load_and_validate_data.py
│
├── 02_simulation/                     # Monte Carlo simulation
│   └── run_monte_carlo_simulation.py
│
├── 04_visualization/                  # Figure generation
│   ├── _helpers.py                    # Helper functions for finding outputs
│   ├── create_figures_1_2_3.py       # Main manuscript figures (DCA vs CA)
│   ├── create_figures_4_5_6.py       # Weather sensitivity figures
│   ├── create_figures_4_5_6_lines.py # Weather sensitivity line graphs
│   └── create_figures_1_2_3_CA_vs_RA.py # CA vs RA figures
│
├── 05_tables/                         # Table generation
│   ├── generate_table_3.py
│   ├── generate_table_6.py
│   └── generate_table_7.py
│
├── outputs/                           # All generated outputs (timestamped)
│   └── YYYYMMDD_HHMMSS/               # Each run creates a timestamped folder
│       ├── .timestamp                 # Timestamp marker file
│       ├── simulation_results/         # Raw simulation outputs (CSV)
│       │   ├── dominance_probabilities_Gala_orchard_year_weather.csv
│       │   ├── dominance_probabilities_Honeycrisp_orchard_year_weather.csv
│       │   ├── break_even_delta_*.csv
│       │   ├── beta_fit_summary_by_cell.csv
│       │   └── [other diagnostic files]
│       ├── figures/                   # Publication-ready figures (PNG)
│       │   ├── Figure_1_GA_O1_DCA_vs_CA_Probability.png
│       │   ├── Figure_2_HC_O1_DCA_vs_CA_Probability.png
│       │   ├── Figure_3_HC_O2_DCA_vs_CA_Probability.png
│       │   ├── Figure_4_GA_O1_Weather_Sensitivity.png
│       │   ├── Figure_5_HC_O1_Weather_Sensitivity.png
│       │   └── Figure_6_HC_O2_Weather_Sensitivity.png
│       └── tables/                    # Formatted tables (STY)
│           ├── Table_3.sty
│           ├── Table_6.sty
│           └── Table_7.sty
│
└── validate_replication.py            # Validation script
```

## Output Organization

**Key Feature: Timestamped Output Folders**

Each simulation run creates a new timestamped folder:
- Format: `outputs/YYYYMMDD_HHMMSS/`
- Example: `outputs/20251214_094525/`

This ensures:
- ✅ Multiple runs don't overwrite each other
- ✅ Easy tracking of when results were generated
- ✅ Clear organization of outputs by run
- ✅ All outputs (simulation, figures, tables) for a run are together

## File Sizes

- **Package size:** ~59 MB (includes test outputs)
- **Input files:** 55 files (scripts, data, documentation)
- **Test outputs:** Included in `outputs/20251214_094525/` from validation run

## Required Files for Replication

**Minimum required to run:**
1. `config/config.py` - Configuration
2. `data/packout_data.xlsx` - Input data
3. `data/weather_assumptions.csv` - Weather flags
4. All Python scripts in numbered directories

**Optional but recommended:**
- Documentation files (README.md, QUICK_START.md, etc.)
- Validation script (validate_replication.py)
- Test outputs in `outputs/20251214_094525/` (for reference)
