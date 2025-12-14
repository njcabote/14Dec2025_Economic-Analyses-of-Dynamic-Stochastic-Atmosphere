
# Interpretation Guide — Economics-First DCA Analysis
**Run date:** 2025-12-14 09:46:30

This guide explains how to read each output, in the **economics-first** scope where **marketable (disorder-free + decay-free)** shares are treated as **exogenous inputs** to a price-driven revenue mapping. Prices are predicted **independently of quality** (Table 4 path).  
**RA** is temperature-controlled but does not manipulate O₂/CO₂; CA/DCA do. (We avoid storage setpoints here; see companion postharvest methods paper.)  

---

## A. What the descriptive tables show
- **table_marketables_orchard_year.csv** — Mean marketable shares at **3/6/9 months** for each **orchard × year × technology** (pooled +1/+7 days). Use this to verify heterogeneity.
- **table_clean_fruit_percentages.csv** — Clean fruit (%) by **cultivar × tech × interval** (pooled over orchards/years).
- **table_implied_decay_percentages.csv** — 100 − clean% from the previous table, i.e., implied cumulative decay.
- **table_price_benchmarks.csv** — Harvest baseline (September) and **peak** price (month and level) by cultivar.
- **table_revenue_index_9mo_peak.csv** — Relative revenue at **9 months** using peak prices (**CA=100** baseline).
- **beta_fit_summary_by_cell.csv** — Beta distribution parameters (α,β) for each orchard×year×tech×interval cell, comparing Method-of-Moments vs Maximum Likelihood, with AIC selection and KS test p-values.

**How to read:**  
For **Gala**, the 9‑month clean fruit % usually shows **DCA > CA ≫ RA**, making extended storage more profitable with DCA if your clean % profile resembles ours.  
For **Honeycrisp**, **CA** typically retains the most clean fruit by 9 months, aligning best with the later **July** price peak.

---

## B. What the figures show
- **Fig1_marketable_by_tech_duration_*.png** — Boxplots of marketable share distributions by tech × (3,6,9) months (non-identifying).
- **Fig2_predicted_prices_*.png** — Monthly price paths ($/kg) used in revenue mapping.
- **Fig3_expected_revenue_*_{RA,CA,DCA}.png** — Expected revenue lines by month, using step updates at **Dec/Mar/Jun** (3/6/9 months).

**How to read:**  
Expected revenue rises with price seasonality **and** with step changes in marketable shares at 3/6/9 months.

---

## C. Monte Carlo outputs (decision-ready)
- **dominance_probabilities_*_orchard_year.csv**  
  For each orchard × year and month, we report **Pr[DCA>CA]** and **Pr[DCA>RA]**, plus the distribution of revenue differences.

  **Decision rule:** Uses A/CVaR methodology: ADOPT (A ≥ 0.05, CVaR ≥ -50k), PILOT (moderate A/probability), CONSIDER CA (A ≤ -0.05 or low probability).

- **break_even_delta_*_orchard_year.csv**  
  For each orchard × year and month, we report **Median** and **10–90%** bands of the **break-even Δ-packout**:
  
ΔPackout_BE(m) = (NetRev_CA − NetRev_DCA) / (Price_m × RoomMass)

  Interpreted as "how many percentage points DCA must improve marketable share to tie CA."

- **fig_PrDCAgtCA_*_orch*_yr*.png** — Stop‑light curves of **Pr[DCA>CA]** across months (per orchard×year).
- **fig_BEdelta_*_orch*_yr*.png** — Median + 10–90% ribbons for Δ‑packout by month (per orchard×year).

**Cultivar-specific guidance (threshold framing):**
- **Gala:** Seasonal price uplift can offset cumulative decay up to ~**45–50%**.  
  At 9 months, **DCA ≈ 58.4% clean (≈41.6% decay)** is **within** this envelope, **CA ≈ 49.27% clean (≈50.7% decay)** is borderline, **RA** exceeds it.  
  → Expect DCA dominance in late winter–spring if your packout trajectory resembles this profile.
- **Honeycrisp:** Profitable late storage generally requires **≤ ~20% decay**; **CA ≈ 82.6% clean (~17.4% decay)** meets it; **RA** marginal; **DCA** exceeds.  
  → Expect **CA** dominance into summer when July prices peak.

---

## D. Caveats & scope
- We do **not** estimate storage setpoints or disorder causality here. Packouts are **observed inputs**. Prices are **predicted independently** of quality (per manuscript Table 4).  
- Operational costs vary by facility; we include **Low/Base/High** monthly room-cost scenarios as robustness.
- Use your **own packout history** to read off probabilities and break‑even thresholds from these outputs.
- **Prices inside MC include an optional small uncertainty per cultivar/month (toggle USE_PRICE_NOISE and PRICE_NOISE_SCALE), which broadens probability bands without changing the base price path used in descriptive figures.**
- **Room costs inside MC can be switched via COST_SCENARIO = Low/Base/High to stress‑test net revenue rankings.**
- **Cost modeling uses two approaches: (i) triangular Low/Base/High scenarios and (ii) component model with energy per ton-month, fixed O&M/capital recovery by technology, and DCA pod rental ($8,300/year per pod). Toggle via COST_MODEL_MODE.**

---

## E. Weather-informed interpretation (orchard × year)
We incorporate **orchard×year weather flags** derived from **location-specific time series meteorological data** (Othello and Quincy weather stations, 2021-2025). Weather flags include: heatwave, drought, cold_spring, harvest_heat, humidity_high, and frost_event. These flags **explain** when DCA can underperform CA without altering measured packouts.

**Orchard locations and weather data sources:**
- **HC_O1**: Othello (Adams County) — uses Othello weather station time series data
- **HC_O2**: Quincy (Grant County) — uses Quincy weather station time series data  
- **GA_O1**: Quincy (Grant County) — uses Quincy weather station time series data

**Weather flag derivation:**
- Flags derived from objective meteorological thresholds applied to time series data
- Location-specific attribution ensures accurate weather conditions for each orchard
- Addresses previous issue where same flags were used for orchards in different locations

**Outputs:**
- `dominance_probabilities_*_orchard_year_weather.csv` — dominance tables merged with weather flags (all flags included).
- `weather_stratified_PrDCAgtCA_*.csv` — mean Pr[DCA>CA] by flag (=0/1) and month (POOLED analysis).
- `weather_stratified_by_orchard_*.csv` — **ORCHARD-DISAGGREGATED** mean Pr[DCA>CA] by flag, orchard, and month.
- `weather_effect_bootstrap_*.csv` — bootstrap 95% CIs (POOLED analysis).
- `weather_effect_bootstrap_by_orchard_*.csv` — **ORCHARD-DISAGGREGATED** bootstrap CIs.
- `weather_sensitivity_summary_by_orchard_*.csv` — comprehensive summary for ALL weather flags by orchard.
- `Weather_Impact_by_Orchard_*_*.png` — orchard-disaggregated plots for each weather flag.
- `FigW_Weather_Impact_Harvest_Heat.png` — pooled harvest heat analysis (for comparison).
- `diagnostic_*_*.csv` — diagnostic checks for pooling effects, year confounding, and attribution issues.

**How to read:**
- **Orchard-disaggregated analysis** (recommended): Use `weather_stratified_by_orchard_*.csv` and `weather_effect_bootstrap_by_orchard_*.csv` to see location-specific weather effects. This addresses the counterintuitive finding where pooled analysis masked location differences.
- **Location-specific differences**: HC_O1 (Othello) and HC_O2 (Quincy) have different weather patterns (e.g., 2023 cold_spring: HC_O1=1, HC_O2=0), which explains orchard-level heterogeneity.
- **All weather flags**: Analysis includes all six flags (heatwave, harvest_heat, drought, cold_spring, humidity_high, frost_event) from time series data.
- **Pooled vs disaggregated**: Compare pooled results (old method) with orchard-disaggregated results (new method) to understand how pooling masked location-specific effects.
- Use the **bootstrap CI** to assess whether weather-associated changes in Pr[DCA>CA] are practically meaningful.

**Key finding:**
Initial pooled analysis suggested counterintuitive results (heat stress appearing to increase DCA probabilities). Orchard-disaggregated analysis using location-specific weather flags reveals the expected pattern: weather stress generally reduces DCA performance, consistent with physiological expectations.

**References: [2] Hardaker et al. (agricultural risk & decision analysis); [4] Rockafellar & Uryasev (CVaR); [7,8] Akaike; Burnham & Anderson (AIC model selection); [9] Efron & Tibshirani (bootstrap); [10] Ferrari & Cribari‑Neto (Beta for proportions).

**Optional sensitivity (OFF by default):**
Setting `WEATHER_STRESS_SCENARIO=True` increases DCA **uncertainty** (not the mean) for flagged Orchard×Year cells during MC draws, reflecting the weather summary's climate‑stress narrative while **preserving an economics-first scope**.

