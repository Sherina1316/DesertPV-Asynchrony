# v24.22 Methods: full-domain remote PV census for 2025

The 2024 inventory is finalized causally using observations no later than 2024. The 2025 image is then analysed independently.

The terminal-year detector has four pathways: gain-supported adjacent expansion, strict model-supported remote sites, component-conditioned perimeter extension, and a new novelty-gated remote change census. The full-domain pathway scans every valid 2025 pixel outside the frozen 2024 inventory. It does not use distance to the inventory and does not require the annual segmentation mask.

For each pixel, the pathway computes absolute 2025 darkness, brightness drop relative to the 2021–2023 baseline, multi-cue change, model/prototype/LSMA support, local candidate density, sand risk and land-cover confuser risk. High-confidence seeds are propagated through weaker site-consistent support. Morphological bridging joins panel rows separated by narrow internal gaps.

A remote object is retained only if it satisfies minimum area, seed fraction, changed-pixel fraction, dark-pixel fraction, likelihood, change, solidity, bounding-box fill and maximum sand/confuser constraints. Stable dark sand is rejected because it has weak terminal change even when its absolute brightness is low.

Stage 8 consumes the Stage-7 source raster without recomputation. Full-domain source-code-4 pixels enter the 2025 active and cumulative footprints, and their first installation year is 2025. They are never copied into earlier years.
