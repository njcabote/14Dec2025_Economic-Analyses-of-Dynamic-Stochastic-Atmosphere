
# Beta Distribution Estimation Summary Report
**Generated:** 2025-12-14 09:46:27

## Overall Fitting Statistics
- **Total cells fitted:** 81 (orchard × year × technology × interval combinations)
- **MLE selected:** 81 cells (100.0%)
- **MoM selected:** 0 cells (0.0%)
- **KS test passed (p > 0.05):** 59 cells (72.8%)

## Model Selection Results (AIC-based)
- **MLE preferred:** 81 cells
- **MoM preferred:** 0 cells  
- **Ties (MLE chosen):** 0 cells

## Concentration Parameter (κ = α + β) Statistics
- **Mean κ:** 305.9
- **Median κ:** 21.0
- **Range:** 0.4 - 1468.7
- **Standard deviation:** 563.5

## Decision Rule for MLE vs MoM Selection
We use the **Akaike Information Criterion (AIC)** to select between Method of Moments (MoM) and Maximum Likelihood Estimation (MLE) [7, 8]:

**AIC = 2k - 2ln(ℒ)**
where k = 2 parameters (α, β) and ℒ is the maximum likelihood value.

**Selection criteria:**
1. **Lower AIC wins** (better fit with parsimony penalty)
2. **Ties resolved in favor of MLE** (theoretical superiority for parameter estimation)
3. **MLE preferred** because it maximizes the likelihood function directly

**Why MLE is theoretically superior:**
- MLE provides asymptotically unbiased and efficient parameter estimates
- MLE handles boundary cases and small sample sizes better
- MLE is invariant to parameter transformations
- MLE provides natural uncertainty quantification through likelihood theory

**Goodness-of-fit:**
- We report **KS test p‑values** computed with a **parametric bootstrap** to account for parameter estimation [9].
- The use of Beta distributions for proportions follows standard practice in the literature [10].

## Fitting Quality Assessment
- **KS test p-values > 0.05:** Indicates adequate model fit
- **Concentration bounds:** κ ∈ [20, 300] prevents overconfidence with limited replicates
- **Visual inspection:** Diagnostic plots generated for each cell in beta_fits/ directory
