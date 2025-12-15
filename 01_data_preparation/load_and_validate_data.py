"""
Module 1: Data Loading and Validation

This module loads and validates the input data files required for the DCA economic analysis.
It handles:
- Packout data (marketable fruit percentages)
- Weather assumptions (weather flags by orchard×year)
- Data type normalization and validation
- Year mapping (if needed)

Author: Nickson Cabote
Date: December 2025
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config


def read_packout_data(file_path):
    """
    Load and validate packout data from Excel or CSV file.
    
    This function:
    1. Reads the input file (supports both .xlsx and .csv)
    2. Validates required columns are present
    3. Normalizes data types and cleans strings
    4. Converts marketable_pct to fraction [0,1]
    5. Filters to relevant storage intervals (3, 6, 9 months)
    
    Parameters
    ----------
    file_path : str or Path
        Path to packout data file (.xlsx or .csv)
    
    Returns
    -------
    pd.DataFrame
        Cleaned packout data with columns:
        - cultivar, orchard_id, year, technology
        - interval_months, day_offset, replicate_id
        - marketable_pct (original percentage)
        - marketable (fraction in [0,1])
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If required columns are missing
    """
    print(f"\n{'='*80}")
    print("STEP 1: LOADING PACKOUT DATA")
    print(f"{'='*80}")
    print(f"Reading file: {file_path}")
    
    # Check if file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Packout data file not found: {file_path}")
    
    # Read file based on extension
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        print(f"✓ File loaded successfully: {len(df)} rows")
    except Exception as e:
        raise RuntimeError(f"Could not read file at {file_path}: {e}")
    
    # Validate required columns
    missing_cols = [col for col in config.REQUIRED_PACKOUT_COLUMNS 
                    if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in packout data: {missing_cols}\n"
            f"Required columns: {config.REQUIRED_PACKOUT_COLUMNS}"
        )
    print(f"✓ All required columns present")
    
    # Clean and normalize data types
    print("Cleaning and normalizing data types...")
    
    # String columns: strip whitespace
    df["cultivar"] = df["cultivar"].astype(str).str.strip()
    df["orchard_id"] = df["orchard_id"].astype(str).str.strip()
    df["technology"] = df["technology"].astype(str).str.upper().str.strip()
    
    # Numeric columns: convert to appropriate types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["interval_months"] = pd.to_numeric(df["interval_months"], errors="coerce").astype("Int64")
    df["day_offset"] = pd.to_numeric(df["day_offset"], errors="coerce").astype("Int64")
    df["replicate_id"] = pd.to_numeric(df["replicate_id"], errors="coerce").astype("Int64")
    
    # Normalize marketable_pct to fraction [0,1]
    # Handles both percentage (0-100) and fraction (0-1) formats
    def to_fraction(x):
        """Convert marketable_pct to fraction [0,1]."""
        if pd.isna(x):
            return np.nan
        v = float(x)
        # If value > 1, assume it's a percentage and divide by 100
        return v / 100.0 if v > 1.0 else v
    
    df["marketable"] = df["marketable_pct"].apply(to_fraction)
    
    # Filter to relevant storage intervals (3, 6, 9 months)
    initial_rows = len(df)
    df = df[df["interval_months"].isin(config.STORAGE_INTERVALS)].copy()
    filtered_rows = len(df)
    print(f"✓ Filtered to storage intervals {config.STORAGE_INTERVALS}: "
          f"{filtered_rows} rows (removed {initial_rows - filtered_rows} rows)")
    
    # Remove rows with missing marketable values
    before_dropna = len(df)
    df = df.dropna(subset=["marketable"])
    after_dropna = len(df)
    if before_dropna > after_dropna:
        print(f"✓ Removed {before_dropna - after_dropna} rows with missing marketable values")
    
    # Validate data ranges
    print("Validating data ranges...")
    
    # Check marketable values are in [0,1]
    invalid_marketable = df[(df["marketable"] < 0) | (df["marketable"] > 1)]
    if len(invalid_marketable) > 0:
        print(f"⚠ Warning: {len(invalid_marketable)} rows with marketable outside [0,1]")
        # Clip to valid range
        df.loc[df["marketable"] < 0, "marketable"] = 0
        df.loc[df["marketable"] > 1, "marketable"] = 1
    
    # Check technologies
    valid_techs = set(config.TECHNOLOGIES)
    invalid_techs = set(df["technology"].unique()) - valid_techs
    if invalid_techs:
        print(f"⚠ Warning: Unexpected technologies found: {invalid_techs}")
        print(f"  Expected: {valid_techs}")
    
    # Check cultivars
    valid_cultivars = set(config.CULTIVARS)
    invalid_cultivars = set(df["cultivar"].unique()) - valid_cultivars
    if invalid_cultivars:
        print(f"⚠ Warning: Unexpected cultivars found: {invalid_cultivars}")
        print(f"  Expected: {valid_cultivars}")
    
    # Apply year mapping if needed
    if config.YEAR_MAPPING:
        print(f"Applying year mapping: {config.YEAR_MAPPING}")
        df['year'] = df['year'].map(config.YEAR_MAPPING).fillna(df['year'])
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Total rows: {len(df)}")
    print(f"Cultivars: {sorted(df['cultivar'].unique())}")
    print(f"Orchards: {sorted(df['orchard_id'].unique())}")
    print(f"Years: {sorted(df['year'].dropna().unique())}")
    print(f"Technologies: {sorted(df['technology'].unique())}")
    print(f"Storage intervals: {sorted(df['interval_months'].unique())}")
    print(f"Marketable range: {df['marketable'].min():.3f} to {df['marketable'].max():.3f}")
    print(f"Marketable mean: {df['marketable'].mean():.3f}")
    print(f"{'='*80}\n")
    
    return df


def read_weather_data(file_path):
    """
    Load and validate weather assumptions data.
    
    This function:
    1. Reads weather flags CSV file
    2. Validates required columns (orchard_id, year, county)
    3. Normalizes all flag columns to 0/1 binary values
    4. Applies year mapping if needed
    
    Parameters
    ----------
    file_path : str or Path
        Path to weather assumptions CSV file
    
    Returns
    -------
    pd.DataFrame or None
        Weather data with orchard_id, year, county, and all weather flags.
        Returns None if file doesn't exist (weather analysis will be skipped).
    
    Raises
    ------
    ValueError
        If required columns are missing
    """
    print(f"\n{'='*80}")
    print("STEP 2: LOADING WEATHER DATA")
    print(f"{'='*80}")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"⚠ Weather file not found: {file_path}")
        print("  Continuing without weather stratification...")
        return None
    
    print(f"Reading file: {file_path}")
    
    try:
        # Read CSV with appropriate dtypes
        df = pd.read_csv(file_path, dtype={"orchard_id": str, "county": str})
        print(f"✓ File loaded successfully: {len(df)} rows")
    except Exception as e:
        raise RuntimeError(f"Could not read weather file at {file_path}: {e}")
    
    # Validate required columns
    required_cols = ["orchard_id", "year", "county"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in weather data: {missing_cols}\n"
            f"Required columns: {required_cols}"
        )
    print(f"✓ All required columns present")
    
    # Normalize year column
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    
    # Apply year mapping if weather file uses old years (2021-2023)
    if df['year'].min() == 2021 and config.YEAR_MAPPING:
        print(f"Applying year mapping: {config.YEAR_MAPPING}")
        df['year'] = df['year'].map(config.YEAR_MAPPING).fillna(df['year'])
    
    # Identify flag columns (all columns except metadata)
    metadata_cols = ["orchard_id", "year", "county", "notes"]
    flag_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Normalize all flag columns to 0/1 binary
    print(f"Normalizing {len(flag_cols)} weather flag columns...")
    for col in flag_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    
    # Summary
    print(f"\nWeather data summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Orchards: {sorted(df['orchard_id'].unique())}")
    print(f"  Years: {sorted(df['year'].dropna().unique())}")
    print(f"  Weather flags: {len(flag_cols)}")
    
    # Check for Honeycrisp-specific flags
    hc_flags = [c for c in flag_cols if 'honeycrisp' in c.lower()]
    if hc_flags:
        print(f"  Honeycrisp phenological flags: {len(hc_flags)}")
    
    print(f"{'='*80}\n")
    
    return df


def pool_day_offsets(df):
    """
    Pool replicate-level data by averaging over day_offset within each cell.
    
    Within each storage interval (3, 6, 9 months), we pool observations
    from different day_offset values (+1 day, +7 days) as they represent
    within-window handling replicates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Replicate-level packout data
    
    Returns
    -------
    pd.DataFrame
        Cell-level data (pooled over day_offset) with columns:
        - cultivar, orchard_id, year, technology, interval_months
        - n_reps (number of replicates)
        - mean_marketable (mean across replicates)
        - sd_marketable (standard deviation across replicates)
    """
    print(f"\n{'='*80}")
    print("STEP 3: POOLING DAY OFFSETS")
    print(f"{'='*80}")
    print("Pooling replicate-level data over day_offset within each cell...")
    
    # Group by cell keys (excluding day_offset and replicate_id)
    pool_keys = ["cultivar", "orchard_id", "year", "technology", "interval_months"]
    
    # Aggregate: count replicates, mean and SD of marketable
    cell_df = (
        df.groupby(pool_keys, as_index=False)
        .agg(
            n_reps=("marketable", "size"),
            mean_marketable=("marketable", "mean"),
            sd_marketable=("marketable", "std")
        )
    )
    
    print(f"✓ Pooled to {len(cell_df)} cells")
    print(f"  Average replicates per cell: {cell_df['n_reps'].mean():.1f}")
    print(f"  Range: {cell_df['n_reps'].min()} to {cell_df['n_reps'].max()} replicates")
    print(f"{'='*80}\n")
    
    return cell_df


def save_replicate_level_data(df, output_dir):
    """
    Save replicate-level data (pooled over day_offset only).
    
    This preserves replicate granularity while pooling day_offset,
    which is useful for Beta distribution fitting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Replicate-level packout data
    output_dir : Path
        Directory to save output files
    """
    print("Saving replicate-level data...")
    
    # Group by all keys except day_offset
    rep_df = (
        df.groupby(
            ["cultivar", "orchard_id", "year", "technology", 
             "interval_months", "replicate_id"],
            as_index=False
        )
        .agg(marketable=("marketable", "mean"))
    )
    
    output_file = output_dir / "replicate_level_marketables.csv"
    rep_df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    
    return rep_df


if __name__ == "__main__":
    """
    Main execution: Load and validate all input data.
    """
    print("\n" + "="*80)
    print("DCA ECONOMIC ANALYSIS - DATA PREPARATION")
    print("="*80)
    
    # Validate paths
    try:
        config.validate_paths()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure all required data files are in the 'data/' directory.")
        sys.exit(1)
    
    # Load packout data
    packout_df = read_packout_data(config.PACKOUT_DATA_FILE)
    
    # Save raw snapshot
    snapshot_file = config.SIMULATION_RESULTS_DIR / "raw_packout_snapshot.csv"
    packout_df.to_csv(snapshot_file, index=False)
    print(f"✓ Saved raw data snapshot: {snapshot_file}")
    
    # Pool day offsets to get cell-level data
    cell_df = pool_day_offsets(packout_df)
    
    # Save cell-level data
    cell_file = config.SIMULATION_RESULTS_DIR / "table_marketables_orchard_year.csv"
    cell_df.to_csv(cell_file, index=False)
    print(f"✓ Saved cell-level data: {cell_file}")
    
    # Save replicate-level data (for Beta fitting)
    rep_df = save_replicate_level_data(packout_df, config.SIMULATION_RESULTS_DIR)
    
    # Load weather data (optional)
    weather_df = read_weather_data(config.WEATHER_DATA_FILE)
    
    if weather_df is not None:
        weather_file = config.SIMULATION_RESULTS_DIR / "weather_data_loaded.csv"
        weather_df.to_csv(weather_file, index=False)
        print(f"✓ Saved weather data: {weather_file}")
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nNext step: Run Monte Carlo simulation")
    print(f"  python 02_simulation/run_monte_carlo_simulation.py")
    print("="*80 + "\n")
