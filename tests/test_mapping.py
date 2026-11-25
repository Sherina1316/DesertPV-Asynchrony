from __future__ import annotations

import numpy as np
from rasterio.coords import BoundingBox

from pvchange.mapping import _crop_extent, _valid_crop


def test_valid_crop_removes_large_empty_canvas() -> None:
    valid = np.zeros((1000, 1000), dtype=bool)
    valid[400:600, 300:550] = True
    crop = _valid_crop(valid, 0.04)
    assert crop[0].stop - crop[0].start < 300
    assert crop[1].stop - crop[1].start < 350
    extent = _crop_extent(BoundingBox(0, 0, 30000, 30000), valid.shape, crop)
    assert extent[1] - extent[0] < 12000
    assert extent[3] - extent[2] < 10000
