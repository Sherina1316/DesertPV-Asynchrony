# Methods: v24.22 full-domain 2025 remote-site census

## Causal endpoint

The audited 2024 PV inventory is finalized using only observations up to 2024. All 2025 detection is evaluated after that endpoint is frozen.

## Independent terminal-year census

For every valid 2025 pixel outside the 2024 inventory, the method derives:

1. absolute darkness relative to the 2025 unit-level brightness distribution;
2. brightness decline from the 2021–2023 robust baseline;
3. probability, prototype, LSMA, local-darkness and multi-cue change evidence;
4. sand and water/vegetation confuser penalties;
5. local candidate density at site scale.

The full-domain likelihood does not include distance to the 2024 inventory. High-confidence seeds are propagated through weaker but physically supported pixels, with morphological bridging to connect panel rows. Components are accepted only when they satisfy minimum area, changed-pixel fraction, dark-pixel fraction, density, solidity, bounding-box fill and confuser limits.

## False-positive control

A dark object is not accepted merely because it is black. It must be temporally new relative to the pre-2024 baseline. Stable dark sand therefore has low remote-change support. Small speckles, hollow edges, water/vegetation confusers and sparse texture are rejected at object level.

## Temporal propagation

The exact Stage-7 source raster is preserved by Stage 8. Source-code 4 pixels enter the 2025 active and cumulative footprints and receive first-installation year 2025. No source-4 pixels are backfilled into earlier years.
