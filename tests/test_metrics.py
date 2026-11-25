import numpy as np

from pvchange.metrics import binary_confusion, boundary_f1, metrics_from_confusion


def test_perfect_binary_metrics() -> None:
    truth = np.zeros((16, 16), bool); truth[4:10, 4:10] = True
    prediction = truth.copy()
    metrics = metrics_from_confusion(binary_confusion(truth, prediction))
    assert metrics["iou"] == 1.0
    assert metrics["f1"] == 1.0
    assert boundary_f1(truth, prediction, 1) == 1.0
