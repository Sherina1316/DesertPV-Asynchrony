# v24.22 existing-code area-only route finalization

This is a compatibility fix for the existing v24.22 codebase. It does not change detection thresholds, the 2024 reference-boundary algorithm, 2025 novelty detection, historical temporal reconstruction, ecological metrics, or 2050 models.

Explicit area-only units: 83, 104, and 141.

Pipeline behavior:

- Stage 7 preserves 15 annual records for each explicit area-only unit and does not build local prototypes or fabricate terminal diagnostic rasters.
- Stage 7 contracts require diagnostic rasters only for pixel-mapped routes.
- Stage 8 carries all Stage-7 `area_only_fallback` units into the existing area-only inference branch.
- Stage 8 emits the complete downstream schema, with empty raster paths and `pixel_map_available=false` for area-only rows.
- Stage 9–21 retain these units in area summaries, uncertainty, ecology, events, maps, reports, and area forecasts while avoiding fabricated pixel footprints and 2050 candidate rasters.
- Duplicate `(Id, year)` rows are rejected before the Stage-8 registry is written.

Validated locally with Python compilation and 109 regression tests.
