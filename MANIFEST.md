# v24.22 package manifest

Core changes:

- `src/pvchange/growth_front.py`: novelty-gated terminal detection.
- `src/pvchange/authoritative_reference.py`: new diagnostic rasters and object audit fields.
- `tools/summarize_reference_test.py`: expanded terminal novelty audit figure.
- `config/analysis_config.yaml`: novelty and persistence thresholds.
- `tests/test_novelty_gated_remote_change_v24_22.py`: new regression tests.

The package retains all inherited 21 workflow scripts, ecological modules, publication mapping, Sankey outputs and strict no-leakage forecast modules.

## v24.22.5 R4 dual-stage additions

- `src/pvchange/dual_stage_reference.py`: independent 2024-SHP internal build-out reconstruction and fusion.
- `src/pvchange/preparation.py`: authoritative parcel presence separated from strong training cores.
- `src/pvchange/quality.py`: spatial route retained for authoritative mixed-pixel parcels.
- `src/pvchange/inference.py`: reference-internal route can continue without a pure-pixel prototype.
- `src/pvchange/authoritative_temporal.py`: Stage-8 two-branch calculation, fusion and diagnostics.
- `tests/test_dual_stage_reference_v24_22_5.py`: eight regression/integration tests.
- `DUAL_STAGE_REFERENCE_FUSION_V24_22_5_CN.md`: algorithm contract.
- `V24_22_5_R4_DUAL_STAGE_OPERATION_GUIDE_CN.md`: server workflow.
