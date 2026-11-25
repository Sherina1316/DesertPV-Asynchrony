def test_figures_module_imports():
    import pvchange.figures as figures
    assert callable(figures.generate_figures)
