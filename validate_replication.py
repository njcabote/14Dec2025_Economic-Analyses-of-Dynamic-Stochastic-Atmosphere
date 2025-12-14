"""
Validation Script: Compare Replication Package Results with Original Results

This script compares key outputs from the replication package with the original
simulation results to verify that the replication is working correctly.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent))
from config import config

def load_original_results(original_dir):
    """Load original simulation results for comparison"""
    original_dir = Path(original_dir)
    
    files = {
        'gala_dom': original_dir / "dominance_probabilities_Gala_orchard_year_weather.csv",
        'honeycrisp_dom': original_dir / "dominance_probabilities_Honeycrisp_orchard_year_weather.csv",
        'gala_be': original_dir / "break_even_delta_Gala_orchard_year_weather.csv",
        'honeycrisp_be': original_dir / "break_even_delta_Honeycrisp_orchard_year_weather.csv",
    }
    
    results = {}
    for key, filepath in files.items():
        if filepath.exists():
            results[key] = pd.read_csv(filepath)
            print(f"✓ Loaded original {key}: {len(results[key])} rows")
        else:
            print(f"⚠️  Original file not found: {filepath}")
            results[key] = None
    
    return results

def load_replication_results(replication_dir):
    """Load replication package results"""
    replication_dir = Path(replication_dir)
    
    files = {
        'gala_dom': replication_dir / "dominance_probabilities_Gala_orchard_year_weather.csv",
        'honeycrisp_dom': replication_dir / "dominance_probabilities_Honeycrisp_orchard_year_weather.csv",
        'gala_be': replication_dir / "break_even_delta_Gala_orchard_year_weather.csv",
        'honeycrisp_be': replication_dir / "break_even_delta_Honeycrisp_orchard_year_weather.csv",
    }
    
    results = {}
    for key, filepath in files.items():
        if filepath.exists():
            results[key] = pd.read_csv(filepath)
            print(f"✓ Loaded replication {key}: {len(results[key])} rows")
        else:
            print(f"⚠️  Replication file not found: {filepath}")
            results[key] = None
    
    return results

def compare_dataframes(orig_df, repl_df, name, tolerance=1e-6):
    """Compare two dataframes and report differences"""
    if orig_df is None or repl_df is None:
        print(f"\n⚠️  Cannot compare {name}: Missing data")
        return False
    
    print(f"\n{'='*80}")
    print(f"COMPARING: {name}")
    print(f"{'='*80}")
    
    # Check dimensions
    if orig_df.shape != repl_df.shape:
        print(f"⚠️  Dimension mismatch: Original {orig_df.shape} vs Replication {repl_df.shape}")
        return False
    
    # Check columns
    orig_cols = set(orig_df.columns)
    repl_cols = set(repl_df.columns)
    if orig_cols != repl_cols:
        missing_in_repl = orig_cols - repl_cols
        extra_in_repl = repl_cols - orig_cols
        if missing_in_repl:
            print(f"⚠️  Columns missing in replication: {missing_in_repl}")
        if extra_in_repl:
            print(f"⚠️  Extra columns in replication: {extra_in_repl}")
        # Continue with common columns
        common_cols = orig_cols & repl_cols
    else:
        common_cols = orig_cols
    
    # Sort both dataframes by same key columns for comparison
    key_cols = ['cultivar', 'orchard_id', 'year', 'month']
    if all(col in orig_df.columns for col in key_cols):
        orig_df = orig_df.sort_values(key_cols).reset_index(drop=True)
        repl_df = repl_df.sort_values(key_cols).reset_index(drop=True)
    
    # Compare numeric columns
    numeric_cols = [col for col in common_cols 
                   if orig_df[col].dtype in [np.float64, np.int64, float, int]]
    
    differences = []
    for col in numeric_cols:
        orig_vals = orig_df[col].fillna(0)
        repl_vals = repl_df[col].fillna(0)
        
        # Calculate differences
        diff = orig_vals - repl_vals
        max_diff = diff.abs().max()
        mean_diff = diff.abs().mean()
        
        if max_diff > tolerance:
            differences.append({
                'column': col,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_diff_pct': (max_diff / (orig_vals.abs().max() + 1e-10)) * 100
            })
    
    if differences:
        print(f"\n⚠️  Found {len(differences)} columns with differences > {tolerance}:")
        for diff in sorted(differences, key=lambda x: x['max_diff'], reverse=True)[:10]:
            print(f"  {diff['column']}: max_diff={diff['max_diff']:.6f}, mean_diff={diff['mean_diff']:.6f}, "
                  f"max_diff_pct={diff['max_diff_pct']:.2f}%")
        return False
    else:
        print(f"✓ All numeric columns match within tolerance ({tolerance})")
        return True

def compare_key_metrics(orig_df, repl_df, name):
    """Compare key summary metrics"""
    if orig_df is None or repl_df is None:
        return
    
    print(f"\n{'='*80}")
    print(f"KEY METRICS: {name}")
    print(f"{'='*80}")
    
    # Key columns to compare
    key_cols = {
        'Pr[DCA>CA]': 'DCA Success Probability',
        'DCA-CA_median': 'Median Revenue Difference',
        'DCA-CA_mean': 'Mean Revenue Difference',
        'p_star': 'Break-even Probability',
        'adoption_index': 'Adoption Index'
    }
    
    for col, label in key_cols.items():
        if col in orig_df.columns and col in repl_df.columns:
            orig_mean = orig_df[col].mean()
            repl_mean = repl_df[col].mean()
            orig_std = orig_df[col].std()
            repl_std = repl_df[col].std()
            
            diff = abs(orig_mean - repl_mean)
            diff_pct = (diff / (abs(orig_mean) + 1e-10)) * 100
            
            print(f"\n{label} ({col}):")
            print(f"  Original:  mean={orig_mean:.6f}, std={orig_std:.6f}")
            print(f"  Replication: mean={repl_mean:.6f}, std={repl_std:.6f}")
            print(f"  Difference: {diff:.6f} ({diff_pct:.2f}%)")
            
            if diff_pct > 1.0:  # More than 1% difference
                print(f"  ⚠️  WARNING: >1% difference detected")

def main():
    """Main validation function"""
    print("="*80)
    print("REPLICATION VALIDATION")
    print("="*80)
    
    # Find original results - prioritize December 13, 2025 script runs
    # Note: Original script outputs to parent directory (without "Manuscript and Revisions")
    original_base_parent = Path("/Users/nicksoncabote/Desktop/DCA Paper Simulations")
    original_base_current = Path("/Users/nicksoncabote/Desktop/DCA Paper Manuscript and Revisions /DCA Paper Simulations")
    
    # First, try to find December 13 script runs in parent directory (where script actually outputs)
    dec13_pattern = "13Dec2025 DCA Simulations for HortScience_*"
    dec13_dirs = []
    
    # Check parent directory first (where the script actually writes)
    if original_base_parent.exists():
        dec13_dirs.extend([d for d in original_base_parent.glob(dec13_pattern) if d.is_dir()])
    
    # Also check current directory
    if original_base_current.exists():
        dec13_dirs.extend([d for d in original_base_current.glob(dec13_pattern) if d.is_dir()])
    
    if dec13_dirs:
        # Use most recent December 13 run
        original_dir = sorted(dec13_dirs, key=lambda x: x.stat().st_mtime)[-1]
        print(f"\nUsing December 13 original results from: {original_dir}")
        print(f"   (Found {len(dec13_dirs)} December 13 runs, using most recent)")
    else:
        # Fallback to older runs
        patterns = [
            "18Sep2025 DCA Simulations for HortScience_*",
        ]
        original_dirs = []
        for pattern in patterns:
            if original_base_parent.exists():
                found = list(original_base_parent.glob(pattern))
                found = [d for d in found if d.is_dir()]
                original_dirs.extend(found)
            if original_base_current.exists():
                found = list(original_base_current.glob(pattern))
                found = [d for d in found if d.is_dir()]
                original_dirs.extend(found)
        
        if not original_dirs:
            print("⚠️  No original simulation results found")
            print("   Looking for: 13Dec2025 DCA Simulations for HortScience_*")
            print("   Checked:")
            print(f"     - {original_base_parent}")
            print(f"     - {original_base_current}")
            print("\n   Please run the original December 13 script first:")
            print("   python '13Dec2025 DCA Simulations for HortScience.py'")
            return
        
        original_dir = sorted(original_dirs, key=lambda x: x.stat().st_mtime)[-1]
        print(f"\n⚠️  Using older original results from: {original_dir}")
        print(f"   (December 13 run not found - may have different parameters)")
    
    # Find replication results (most recent)
    replication_base = config.SIMULATION_RESULTS_DIR
    # Look in the simulation_results directory itself
    replication_dirs = [d for d in replication_base.iterdir() 
                       if d.is_dir() and 'run_monte_carlo_simulation' in d.name]
    
    if not replication_dirs:
        print("\n⚠️  No replication results found")
        print(f"   Looking in: {replication_base}")
        print("   Please run the simulation first:")
        print("   python 02_simulation/run_monte_carlo_simulation.py")
        return
    
    replication_dir = sorted(replication_dirs, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Using replication results from: {replication_dir}")
    
    # Load results
    print("\n" + "="*80)
    print("LOADING RESULTS")
    print("="*80)
    orig_results = load_original_results(original_dir)
    repl_results = load_replication_results(replication_dir)
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARING RESULTS")
    print("="*80)
    
    comparisons = [
        ('Gala Dominance Probabilities', 'gala_dom', orig_results['gala_dom'], repl_results['gala_dom']),
        ('Honeycrisp Dominance Probabilities', 'honeycrisp_dom', orig_results['honeycrisp_dom'], repl_results['honeycrisp_dom']),
        ('Gala Break-Even', 'gala_be', orig_results['gala_be'], repl_results['gala_be']),
        ('Honeycrisp Break-Even', 'honeycrisp_be', orig_results['honeycrisp_be'], repl_results['honeycrisp_be']),
    ]
    
    all_match = True
    for name, key, orig_df, repl_df in comparisons:
        if orig_df is not None and repl_df is not None:
            # Compare key metrics first
            compare_key_metrics(orig_df, repl_df, name)
            
            # Then detailed comparison
            match = compare_dataframes(orig_df, repl_df, name, tolerance=1e-3)
            if not match:
                all_match = False
    
    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    if all_match:
        print("✅ REPLICATION SUCCESSFUL: All results match within tolerance")
    else:
        print("⚠️  REPLICATION ISSUES: Some differences detected")
        print("   Review differences above and check:")
        print("   1. Random seed is set correctly (should be 42)")
        print("   2. Monte Carlo iterations match (should be 10,000)")
        print("   3. All parameters in config.py match original script")
        print("   4. Data files are identical")

if __name__ == "__main__":
    main()
