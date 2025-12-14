"""
Centralized Configuration File for DCA Economic Analysis Replication Package

This file contains all parameters, paths, and settings used throughout the analysis.
Modify values here to adjust simulation parameters, file paths, or visualization settings.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

# Base directory (package root)
BASE_DIR = Path(__file__).parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Output directory structure
# Outputs are organized by timestamp (YYYYMMDD_HHMMSS) for each run
OUTPUT_DIR = BASE_DIR / "outputs"
# Note: SIMULATION_RESULTS_DIR, FIGURES_DIR, and TABLES_DIR are created with timestamps
# in the respective scripts. Base OUTPUT_DIR is created here for structure.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper function to get timestamped output directories
def get_timestamped_outputs(timestamp=None):
    """Get timestamped output directories for a specific run.
    If timestamp is None, creates a new timestamp."""
    from datetime import datetime
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_output = OUTPUT_DIR / timestamp
    return {
        'base': base_output,
        'simulation': base_output / "simulation_results",
        'figures': base_output / "figures",
        'tables': base_output / "tables"
    }

# For backward compatibility, define base directories
# These will be overridden by timestamped directories in scripts
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
SIMULATION_RESULTS_DIR = OUTPUT_DIR / "simulation_results"

# Input data files
PACKOUT_DATA_FILE = DATA_DIR / "packout_data.xlsx"
WEATHER_DATA_FILE = DATA_DIR / "weather_assumptions.csv"

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Monte Carlo settings
MONTE_CARLO_ITERATIONS = 10000  # Number of draws per scenario
RANDOM_SEED = 42  # For reproducibility
KAPPA_MIN = 20  # Minimum Beta concentration parameter
KAPPA_MAX = 300  # Maximum Beta concentration parameter

# Year mapping (if data uses different year labels)
YEAR_MAPPING = {2021: 2022, 2022: 2023, 2023: 2024}

# Storage intervals (months)
STORAGE_INTERVALS = [3, 6, 9]

# ============================================================================
# ECONOMIC PARAMETERS
# ============================================================================

# Monthly predicted prices ($/kg) from manuscript Table 4
PRICES = {
    "Gala": {
        "September": 0.88, "October": 0.94, "November": 1.10, "December": 1.13,
        "January": 1.20, "February": 1.21, "March": 1.25, "April": 1.19,
        "May": 1.20, "June": 1.15, "July": 1.04, "August": 0.91
    },
    "Honeycrisp": {
        "September": 3.16, "October": 3.23, "November": 3.48, "December": 3.36,
        "January": 3.41, "February": 3.46, "March": 3.59, "April": 3.49,
        "May": 3.64, "June": 3.65, "July": 3.78, "August": 3.51
    }
}

# Optional price uncertainty (standard deviation, $/kg)
# Set to None to use deterministic prices
USE_PRICE_NOISE = True
PRICE_NOISE_SCALE = 1.0  # Global multiplier for price uncertainty

PRICES_SD = {
    "Gala": {
        "September": 0.03, "October": 0.03, "November": 0.04, "December": 0.04,
        "January": 0.05, "February": 0.05, "March": 0.05, "April": 0.05,
        "May": 0.05, "June": 0.05, "July": 0.06, "August": 0.06
    },
    "Honeycrisp": {
        "September": 0.08, "October": 0.08, "November": 0.09, "December": 0.09,
        "January": 0.10, "February": 0.10, "March": 0.11, "April": 0.11,
        "May": 0.12, "June": 0.12, "July": 0.15, "August": 0.12
    }
}

# Room capacity parameters
ROOMS = {
    "Gala": {"bins": 2000, "kg_per_bin": 420},  # ~925 lb per bin
    "Honeycrisp": {"bins": 2000, "kg_per_bin": 397},  # ~875 lb per bin
}

# ============================================================================
# COST MODEL PARAMETERS
# ============================================================================

# Cost model mode: "TRIANGULAR" or "COMPONENT"
COST_MODEL_MODE = "COMPONENT"

# Energy anchor (per ton-month)
ENERGY_USD_PER_TON_MONTH = 4.0

# Technology-specific energy multipliers
# Updated per meeting consensus: CA and DCA require additional energy for
# nitrogen generation, oxygen control, and CO2 scrubbing
ENERGY_FACTOR = {
    "RA": 1.00,  # Baseline
    "CA": 1.25,  # Additional energy for O2 control and CO2 scrubbing
    "DCA": 1.50  # Additional energy beyond CA for dynamic control
}

# Fixed O&M + capital recovery per month (non-energy components)
FIXED_OPEX_CAPEX_BASE = {
    "RA": 2500.0,   # Room upkeep, fans, refrigeration overhead
    "CA": 6000.0,   # + scrubber/O2 control O&M and capex recovery
    "DCA": 6500.0   # + DCA supervisory layer (excl. pod rental)
}

# DCA pod economics
DCA_POD_ANNUAL_USD = 8300.0  # Per pod per year
DCA_PODS_PER_ROOM = 1
POD_SHARING_ROOMS = 1  # If pods shared across rooms, >1 reduces per-room cost

# Season/month accounting
MONTHS_IN_OPERATION = 9  # Months of active CA/DCA control (Dec-Aug)
MONTHS_IN_YEAR = 12

# Cost noise (month-to-month variability around component mean)
COST_NOISE_REL_WIDTH = {
    "RA": 0.20,
    "CA": 0.18,
    "DCA": 0.18
}

# Alternative: Triangular cost distributions (if COST_MODEL_MODE = "TRIANGULAR")
COSTS_TRI_SCENARIOS = {
    "Low": {
        "RA": (2500, 3500, 5000),
        "CA": (6500, 8500, 11500),
        "DCA": (7000, 9000, 13000)
    },
    "Base": {
        "RA": (3000, 4000, 6000),
        "CA": (7000, 9000, 12000),
        "DCA": (8000, 10000, 14000)
    },
    "High": {
        "RA": (4000, 5000, 7500),
        "CA": (8000, 10500, 14000),
        "DCA": (9000, 12000, 17000)
    }
}
COST_SCENARIO = "Base"  # Options: "Low", "Base", "High"

# ============================================================================
# MONTH MAPPING
# ============================================================================

# Months in order
MONTHS = [
    "September", "October", "November", "December", "January", "February",
    "March", "April", "May", "June", "July", "August"
]

# Month to storage interval mapping
# Packout updates occur at 3, 6, and 9 months
MONTH_TO_INTERVAL = {
    # Dec-Feb → 3 months
    "December": 3, "January": 3, "February": 3,
    # Mar-May → 6 months
    "March": 6, "April": 6, "May": 6,
    # Jun-Aug → 9 months
    "June": 9, "July": 9, "August": 9,
    # Sep-Nov (pre-step) → handled separately
    "September": None, "October": None, "November": None
}

# ============================================================================
# DECISION ANALYSIS PARAMETERS
# ============================================================================

CVaR_ALPHA = 0.10  # Worst 10% tail expected outcome
ADOPTION_BAND = 0.10  # +/- band around A = 0 to define "Amber" zone
CVaR_GUARDRAIL_ABS = 0.0  # Set >0 if allowing small worst-decile losses

# ============================================================================
# WEATHER PARAMETERS
# ============================================================================

# Weather stress scenario (optional, OFF by default)
# When ON, inflates DCA variance under specified shocks for sensitivity analysis
WEATHER_STRESS_SCENARIO = False

WEATHER_STRESS = {
    "Honeycrisp": {
        "heatwave": {"factor": 0.75},
        "harvest_heat": {"factor": 0.80},
        "drought": {"factor": 0.85}
    },
    "Gala": {
        "heatwave": {"factor": 0.90},
        "harvest_heat": {"factor": 0.90},
        "drought": {"factor": 0.90}
    }
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Washington State University Color Palette
PROFESSIONAL_COLORS = {
    # Year colors
    'year_2022': '#00a5bd',  # WSU Blue Accent
    'year_2023': '#A60F2D',  # WSU Crimson
    'year_2024': '#b67233',  # WSU Orange
    # Orchard colors
    'HC_O1': '#A60F2D',      # WSU Crimson
    'HC_O2': '#4f868e',      # WSU Blue
    'HC_Combined': '#8f7e35',  # WSU Green
    'GA_O1': '#b67233',      # WSU Orange
    # Weather sensitivity colors
    'without_heat': '#4f868e',  # WSU Blue
    'with_heat': '#A60F2D',     # WSU Crimson
    # Reference and UI elements
    'reference_line': '#4D4D4D',  # WSU Gray
    'grid_color': '#E5E5E5',      # Light gray
    'text_dark': '#2C2C2C',       # Near black
    'text_light': '#4D4D4D',     # WSU Gray
    'background': '#FFFFFF',     # White
    'edge_white': '#FFFFFF'       # White for bar edges
}

# Matplotlib settings for publication-quality figures
MATPLOTLIB_SETTINGS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.edgecolor': PROFESSIONAL_COLORS['text_light'],
    'axes.labelcolor': PROFESSIONAL_COLORS['text_dark'],
    'text.color': PROFESSIONAL_COLORS['text_dark'],
    'grid.color': PROFESSIONAL_COLORS['grid_color'],
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
    'figure.facecolor': PROFESSIONAL_COLORS['background'],
    'axes.facecolor': PROFESSIONAL_COLORS['background']
}

# Figure dimensions (width, height in inches)
FIGURE_SIZES = {
    'standard': (14, 8),
    'wide': (16, 8),
    'tall': (10, 12),
    'square': (10, 10)
}

# ============================================================================
# DATA VALIDATION
# ============================================================================

# Required columns in packout data
REQUIRED_PACKOUT_COLUMNS = [
    "cultivar", "orchard_id", "year", "technology",
    "interval_months", "day_offset", "replicate_id", "marketable_pct"
]

# Expected technologies
TECHNOLOGIES = ["RA", "CA", "DCA"]

# Expected cultivars
CULTIVARS = ["Gala", "Honeycrisp"]

# Expected orchards
ORCHARDS = {
    "Gala": ["GA_O1"],
    "Honeycrisp": ["HC_O1", "HC_O2"]
}

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Output profile: "full" or "minimal"
OUTPUT_PROFILE = "full"

# Save intermediate results
SAVE_INTERMEDIATE_RESULTS = True

# File formats
FIGURE_FORMAT = "png"  # Options: "png", "pdf", "svg"
TABLE_FORMAT = "xlsx"  # Options: "xlsx", "csv"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_output_path(subdirectory, filename):
    """Get full path for output file in specified subdirectory."""
    if subdirectory == "figures":
        return FIGURES_DIR / filename
    elif subdirectory == "tables":
        return TABLES_DIR / filename
    elif subdirectory == "simulation_results":
        return SIMULATION_RESULTS_DIR / filename
    else:
        return OUTPUT_DIR / subdirectory / filename

def validate_paths():
    """Validate that all required input files exist."""
    missing = []
    if not PACKOUT_DATA_FILE.exists():
        missing.append(str(PACKOUT_DATA_FILE))
    if not WEATHER_DATA_FILE.exists():
        missing.append(str(WEATHER_DATA_FILE))
    
    if missing:
        raise FileNotFoundError(
            f"Required input files not found:\n" + "\n".join(missing)
        )
    return True
