"""
Helper functions for visualization scripts to find timestamped output directories
"""
from pathlib import Path
from config import config

def find_latest_timestamped_output():
    """Find the most recent timestamped output directory.
    Returns (timestamp_dir, simulation_path, figures_path, tables_path)"""
    output_base = config.OUTPUT_DIR
    
    # Look for timestamp directories (YYYYMMDD_HHMMSS format)
    timestamp_dirs = [d for d in output_base.iterdir() 
                     if d.is_dir() and len(d.name) == 15 and d.name[0].isdigit() 
                     and '_' in d.name and d.name.replace('_', '').isdigit()]
    
    if timestamp_dirs:
        # Get most recent timestamp directory
        latest_timestamp_dir = sorted(timestamp_dirs, key=lambda x: x.stat().st_mtime)[-1]
        timestamp = latest_timestamp_dir.name
        
        simulation_path = latest_timestamp_dir / "simulation_results"
        figures_path = latest_timestamp_dir / "figures"
        tables_path = latest_timestamp_dir / "tables"
        
        return latest_timestamp_dir, simulation_path, figures_path, tables_path, timestamp
    else:
        # Fallback: check old structure
        sim_results_dir = config.SIMULATION_RESULTS_DIR
        if sim_results_dir.exists():
            sim_dirs = [d for d in sim_results_dir.iterdir() 
                       if d.is_dir() and 'run_monte_carlo_simulation' in d.name]
            if sim_dirs:
                sim_path = sorted(sim_dirs, key=lambda x: x.stat().st_mtime)[-1]
                return None, sim_path, config.FIGURES_DIR, config.TABLES_DIR, None
            elif (sim_results_dir / "dominance_probabilities_Gala_orchard_year_weather.csv").exists():
                return None, sim_results_dir, config.FIGURES_DIR, config.TABLES_DIR, None
        
        raise FileNotFoundError(
            f"No simulation results found. Please run simulation first:\n"
            f"  python 02_simulation/run_monte_carlo_simulation.py"
        )
