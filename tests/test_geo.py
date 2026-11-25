from pvchange.geo import aligned_bounds


def test_bounds_align_to_30m_grid() -> None:
    bounds = aligned_bounds((11.1, 22.2, 1001.2, 902.9), 30, 30)
    assert all(abs(value / 30 - round(value / 30)) < 1e-9 for value in bounds)
