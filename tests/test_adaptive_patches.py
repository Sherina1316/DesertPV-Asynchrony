from pvchange.preparation import _effective_patch_shape, _window_offsets


def test_smaller_than_configured_patch_uses_observed_extent() -> None:
    ph, pw, adaptive = _effective_patch_shape(111, 118, 128)
    assert (ph, pw, adaptive) == (111, 118, True)
    assert _window_offsets(111, ph, min(64, ph)) == [0]
    assert _window_offsets(118, pw, min(64, pw)) == [0]


def test_only_one_dimension_is_adaptive() -> None:
    ph, pw, adaptive = _effective_patch_shape(96, 300, 128)
    assert (ph, pw, adaptive) == (96, 128, True)
    assert _window_offsets(96, ph, min(64, ph)) == [0]
    assert _window_offsets(300, pw, 64) == [0, 64, 128, 172]


def test_large_raster_keeps_configured_patch() -> None:
    ph, pw, adaptive = _effective_patch_shape(256, 256, 128)
    assert (ph, pw, adaptive) == (128, 128, False)


if __name__ == "__main__":
    test_smaller_than_configured_patch_uses_observed_extent()
    test_only_one_dimension_is_adaptive()
    test_large_raster_keeps_configured_patch()
    print("adaptive patch tests passed")
