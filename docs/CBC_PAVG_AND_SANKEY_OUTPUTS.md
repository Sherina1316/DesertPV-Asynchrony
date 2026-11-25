# v24.6 CBC-PAVG and Sankey outputs

## CBC-PAVG

CBC-PAVG means **Change-Boundary-Coherence Probability Aggregation for PV Growth**.  It is a temporal evidence index for early photovoltaic installation recovery inside the 2024 SHP anchor.  It integrates annual classifier probability, LSMA photovoltaic fraction, 2024 prototype spectral similarity, low-confuser evidence, installation boundary jump, greenness/brightness/bare-soil transition, and future persistence.

The index is designed to avoid two failure modes:

1. A bare or sandy patch cannot become PV only because its spectral value is close to the 2024 PV prototype.
2. A true early PV block is not rejected only because its per-pixel probability is weak before 2016.

Candidate installation years must have **current-year evidence**. Future support proves persistence only and must not shift installation to a pre-construction year.

## New Sankey outputs

Step 18 writes the following files to `output_v24_6/18_figures`:

- `figure_10_ecological_response_sankey.{png,pdf,svg}`
- `figure_11_multimetric_ecology_sankey.{png,pdf,svg}`
- `ecology_response_sankey_nodes.csv`
- `ecology_response_sankey_links.csv`
- `ecology_response_sankey_unit_classification.csv`
- `multimetric_ecology_sankey_nodes.csv`
- `multimetric_ecology_sankey_links.csv`
- `multimetric_ecology_sankey_unit_classification.csv`
- `sankey_status.json`

## Technical route

Step 20 writes an upgraded Chinese technical-route figure to `output_v24_6/20_report` with explicit CBC-PAVG, ecological Sankey and 2050-scenario modules.
