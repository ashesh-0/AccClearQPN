import numpy as np
import torch

from utils.performance_diagram import PerformanceDiagram


def test_compute_PD_SR():
    pd = PerformanceDiagram([1, 5])
    target = torch.Tensor([
        [10, 2, 0, 4],
        [1, 20, 0, 40],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ])
    prediction = torch.Tensor([
        [6, 4, 1, 8],
        [1, 10, 5, 30],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ])
    output = pd.compute_PD_SR(prediction, target, 1)
    recall = torch.Tensor([1, 1, 0.5, -1e6])
    precision = torch.Tensor([0.75, 0.75, 1, -1e6])
    assert torch.all(output[0] == recall).item()
    assert torch.all(output[1] == precision).item()

    output = pd.compute_PD_SR(prediction, target, 5)
    recall = torch.Tensor([1, 1, -1e6, -1e6])
    precision = torch.Tensor([0.5, 2 / 3, -1e6, -1e6])
    assert torch.all(output[0] == recall).item()
    assert torch.all(output[1] == precision).item()


def test_compute():
    pd = PerformanceDiagram(thresholds=[1, 5], weights=[1, 1])
    target = torch.Tensor([
        [10, 2, 0, 4],
        [1, 20, 0, 40],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ]).view(4, 2, 2)
    prediction = torch.Tensor([
        [6, 4, 1, 8],
        [1, 10, 5, 30],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]).view(4, 2, 2)

    output = pd.compute(prediction, target)
    # threshold = 1
    recall = torch.mean(torch.Tensor([1, 1, 0.5]))
    precision = torch.mean(torch.Tensor([0.75, 0.75, 1]))
    m1 = (precision + recall) / np.sqrt(2)

    # threshold 5
    recall = torch.mean(torch.Tensor([1, 1]))
    precision = torch.mean(torch.Tensor([0.5, 2 / 3]))
    m2 = (precision + recall) / np.sqrt(2)
    assert output == (m1 + m2) / 2
