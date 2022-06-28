import torch

from utils.analysis_utils import batch_CSI, batch_HSS, batch_precision, batch_recall


def test_batch_precision_should_work_for_torch():
    # N,
    target = torch.Tensor([
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ]).type(torch.bool)
    prediction = torch.Tensor([
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
    ]).type(torch.bool)

    expected = torch.Tensor([1, 0.5, 1, -1e6])
    predicted = batch_precision(prediction, target)
    assert torch.all(expected == predicted).item()


def test_batch_recall_should_work_for_torch():
    # N,
    target = torch.Tensor([
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ]).type(torch.bool)
    prediction = torch.Tensor([
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
    ]).type(torch.bool)

    expected = torch.Tensor([0.5, 0.5, 1, -1e6])
    predicted = batch_recall(prediction, target)
    assert torch.all(expected == predicted).item()


def test_batch_CSI_should_work_for_torch():
    # N,
    target = torch.Tensor([
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ]).type(torch.bool)
    prediction = torch.Tensor([
        [0, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
    ]).type(torch.bool)

    # TP/(TP+FN+FP)
    expected = torch.Tensor([1 / (1 + 1), 1 / (1 + 1 + 2), 2 / (2 + 0 + 0), 0 / (0 + 1)])
    predicted = batch_CSI(prediction, target)
    assert torch.all(expected == predicted).item()


def test_batch_HSS_should_work_for_torch():
    # N,
    target = torch.Tensor([
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
    ]).type(torch.bool)
    prediction = torch.Tensor([
        [0, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
    ]).type(torch.bool)

    # TP*TN -FP*FN/((TP+FN)*(TN+FN) + (TP+FP)*(TN+FP)
    expected = torch.Tensor([
        (1 * 2 - 0) / ((1 + 1) * (2 + 1) + (1 + 0) * (2 + 0)),
        (1 * 0 - 2 * 1) / ((1 + 1) * (0 + 1) + (1 + 2) * (0 + 2)),
        (2 * 2 - 0 * 0) / (8),
        (0 * 3 - 1 * 0) / (3),
    ])
    predicted = batch_HSS(prediction, target)
    assert torch.all(expected == predicted).item()
