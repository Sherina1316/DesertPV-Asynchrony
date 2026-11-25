# v24.18 methods: authoritative 2024 census, historical reconstruction and 2050 projection

## 1. Reference-year data model

The manually corrected vector `/home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.shp` is the complete binary census of active photovoltaic (PV) panels in 2024. After one canonical fractional rasterisation, every pixel with non-zero vector overlap is a 2024 PV pixel and every pixel outside the vector is a 2024 non-PV pixel. Fractional overlap is retained for area accounting, while the binary raster is reused without modification in all later stages.

The 2024 vector is an endpoint observation, not a historical footprint. It is never copied to 2011–2023. The final 2024 active map is exactly the rasterised census and is not supplemented or eroded by image inference.

## 2. Complete-census supervision and calibration

The complete 2024 binary raster supplies both positive and negative labels. Units are spatially divided into training, calibration and independent test cohorts. Annual Landsat features include six reflectance bands, vegetation/water/bare-soil indices, tasseled-cap components, low-albedo and shade fractions from constrained spectral mixture analysis, reconstruction error, and PV-sensitive spectral evidence. Candidate models are compared under spatial blocking. The selected ensemble is calibrated using complete-census labels, and the operating threshold is chosen from a predeclared precision–recall/FPR objective.

## 3. Annual image evidence

For each pixel and year from 2011 to 2025, a calibrated image-evidence probability combines:

- ensemble probability;
- similarity to 2024 PV spectral prototypes;
- LSMA PV abundance;
- visible/NIR/SWIR darkness and local contrast;
- water, vegetation, built-up and bare-soil confuser evidence;
- texture, directional coherence and stable-sand risk;
- epistemic uncertainty.

Environmental anomalies may only reduce the reliability of image evidence in atypical years. They never increase a weak observation into a historical PV label.

## 4. Endpoint-constrained installation reconstruction

For each pixel in the 2024 census, candidate installation years are evaluated by an ordered change-point likelihood. The objective rewards low PV probability before installation and sustained high probability after installation. A historical onset is accepted only when post-installation probability, support-year count and pre/post gain satisfy the predeclared rules. Otherwise, onset defaults to 2024. A 2011 pre-existing facility requires support in both 2011 and 2012.

Outside the 2024 census, historical PV is retained only as a multi-year persistent track that is inactive by 2024. A 2025 outside-census site is accepted only under a high-confidence end-of-series rule. Small isolated components, stable sand and single-year artefacts are rejected.

## 5. Dual annual states

Two products are maintained:

- **Observed active PV footprint**: PV supported in the current year; the 2024 active footprint equals the corrected census exactly.
- **Cumulative installed footprint**: all accepted historical installations, including previously installed areas that later became inactive or were removed.

This distinction permits historical retirement analysis without altering the current 2024 distribution.

## 6. Complete-reference validation

Before the 2024 endpoint is imposed, independent model probabilities are compared with the complete 2024 census. Metrics include precision, recall, F1, IoU, PR-AUC, ROC-AUC, one-pixel boundary F1, false-positive rate and relative area error. After endpoint enforcement, exact pixel equality and IoU=1 are audited as data-contract checks rather than model-accuracy claims.

## 7. PV dynamics and ecology

Annual active area, cumulative installed area, gain, loss, expansion direction, object transitions and uncertainty are derived on canonical 30 m grids. Ecological analyses use fixed spatial supports for the PV core, edge buffers and background controls. Metrics include PAVG, NDVI/EVI/tasseled-cap greenness, NDMI/tasseled-cap wetness, vegetation abundance, response lag and stabilisation time. Climate covariates are imputed and used in fixed-effect/event-study adjustment, same-year not-yet-treated controls and matching diagnostics. Results are interpreted as climate-adjusted associations, not automatic causal effects.

## 8. Forecast to 2050

The forecast module uses strict rolling-origin historical hindcasting. All target-derived predictors are lagged within each unit. PV growth is modelled in two parts: expansion occurrence and positive gain magnitude. Recursive forecasts apply capacity, coverage-ratio and growth-rate constraints. Environmental drivers are projected conservatively from bounded recent trends or explicit future scenarios. Spatial allocation excludes the current footprint and uses radiation, terrain and other available constraints. Model spread and residual uncertainty produce scenario envelopes and quantiles through 2050.

## 9. Publication graphics

The pipeline exports 600-dpi PNG maps and vector PDF/SVG figures. Annual PV maps use one mutually exclusive classification layer, shared extents and external area annotations. Main outputs include annual atlases, complete-reference accuracy, national dynamics, ecological gradients, response lags, 2050 sensitivity, two Sankey diagrams and full reproducibility manifests.
