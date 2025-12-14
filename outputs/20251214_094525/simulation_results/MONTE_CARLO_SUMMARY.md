
# Monte Carlo Simulation Summary Report
**Generated:** 2025-12-14 09:46:27
**Simulation settings:** 10,000 iterations, COMPONENT cost model, Base scenario

## Overall Dominance Results
- **Total scenarios analyzed:** 108 (orchard × year × month combinations)
- **Strong DCA dominance (Pr[DCA>CA] ≥ 0.8):** 6 scenarios (5.6%)
- **Weak DCA dominance (Pr[DCA>CA] ≤ 0.2):** 60 scenarios (55.6%)
- **Intermediate cases (0.2 < Pr[DCA>CA] < 0.8):** 42 scenarios (38.9%)

## By Cultivar Analysis
### Honeycrisp
- **Strong DCA dominance:** 3 scenarios (4.2%)
- **Weak DCA dominance:** 45 scenarios (62.5%)
- **Mean Pr[DCA>CA]:** 0.234

### Gala
- **Strong DCA dominance:** 3 scenarios (8.3%)
- **Weak DCA dominance:** 15 scenarios (41.7%)
- **Mean Pr[DCA>CA]:** 0.304

## Technology Comparison
- **Mean Pr[DCA>CA]:** 0.257
- **Mean Pr[DCA>RA]:** 0.288

## Break-even Analysis
- **Median break-even Δ packout:** 0.001 percentage points
- **Mean break-even Δ packout:** 0.030 percentage points
- **Range:** -0.198 to 0.434 percentage points

## Decision Rule Interpretation
**Note:** These are summary statistics only. Actual decisions use the **A/CVaR methodology** [2, 4]:
- **ADOPT:** A ≥ 0.10 **and** CVaR₁₀%(Δ) ≥ 0
- **PILOT:** |A| < 0.10 or small acceptable tail losses
- **CONSIDER CA:** otherwise

## Cost Model Details
- **Mode:** COMPONENT
- **Scenario:** Base
- **Price noise:** Enabled
- **Weather integration:** Enabled
- **Weather data source:** Time-series derived flags (location-specific: Othello for HC_O1, Quincy for HC_O2/GA_O1)
- **Weather flags:** All 6 flags included (heatwave, harvest_heat, drought, cold_spring, humidity_high, frost_event)
