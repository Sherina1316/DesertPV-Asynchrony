import numpy as np

from pvchange.inference import _predict_weighted_ensemble


class DummyModel:
    def __init__(self, scale: float, bias: float):
        self.scale = scale
        self.bias = bias

    def predict_proba(self, x):
        z = np.clip(x[:, 0] * self.scale + self.bias, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


def test_weighted_ensemble_matches_direct_stack():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1003, 4)).astype(np.float32)
    models = [DummyModel(0.8, -0.2), DummyModel(1.3, 0.1)]
    weights = np.array([0.35, 0.65], dtype=np.float64)
    member = np.stack([m.predict_proba(x)[:, 1] for m in models], axis=0).astype(np.float32)
    expected_mean = np.average(member, axis=0, weights=weights)
    expected_std = np.sqrt(np.average((member - expected_mean.reshape(1, -1)) ** 2, axis=0, weights=weights))
    mean, std = _predict_weighted_ensemble(models, weights, x, batch_rows=127)
    assert np.allclose(mean, expected_mean, rtol=1e-6, atol=1e-7)
    assert np.allclose(std, expected_std, rtol=1e-5, atol=1e-7)
