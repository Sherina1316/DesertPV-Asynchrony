from __future__ import annotations

PIXEL_MODES = frozenset({"direct_pixel", "anchor_guided_candidate"})
AREA_ONLY_MODES = frozenset({"area_only_fallback", "anchor_analog_area", "unanchored_analog_area"})


def is_pixel_mode(value: object) -> bool:
    return str(value) in PIXEL_MODES


def is_area_only_mode(value: object) -> bool:
    return not is_pixel_mode(value)
