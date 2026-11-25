# v24.8 Landsat Object-PAVG test version

This version returns to the v24 baseline and adds a Landsat-composite object evidence layer for early-year recovery and endpoint-hole suppression.

## Main correction

The model does not decide PV from spectral similarity alone. Annual evidence must form a coherent object inside the 2024 SHP or near the reference anchor. The new object-PAVG score combines:

- calibrated annual PV probability;
- LSMA PV fraction evidence;
- spectral similarity to the 2024 PV reference;
- local dark-object contrast relative to the surrounding desert background;
- texture and edge evidence from Landsat annual composites;
- low-confuser evidence;
- future persistence support.

## Test mode

The project supports non-contiguous unit subsets via:

```yaml
study:
  selected_unit_ids: [1, 6, 18, 28]
```

This enables a full Step 1--20 test run on Units 1, 6, 18, and 28 without processing all 142 units.
