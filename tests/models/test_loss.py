import numpy as np
import torch

from models.loss import KernelWeightedMaeLoss, compute_weights


def test_compute_weights():
    target = np.array([0, 1, 1.5, 2, 2.5, 5, 8, 10, 15, 30, 35, 100]).reshape(4, 3)
    expt_w = np.array([1, 1, 1.0, 2, 2.0, 5, 5, 10, 10, 30, 30, 30]).reshape(4, 3)
    mask = np.ones_like(target, dtype=int)
    w = compute_weights(torch.Tensor(target).cuda(), torch.Tensor(mask).cuda()).cpu().numpy()
    assert (expt_w == w).all()


def test_compute_weights_handles_mask():
    target = np.array([0, 1, 1.5, 2, 2.5, 5, 8, 10, 15, 30, 35, 100]).reshape(4, 3)
    expt_w = np.array([1, 1, 1.0, 2, 2.0, 5, 5, 10, 10, 30, 30, 30]).reshape(4, 3)
    mask = np.ones_like(target, dtype=int)

    mask[1, 2] = 0
    expt_w[1, 2] = 0

    mask[2, 0] = 0
    expt_w[2, 0] = 0

    w = compute_weights(torch.Tensor(target).cuda(), torch.Tensor(mask).cuda()).cpu().numpy()
    assert (expt_w == w).all()


def test_kernel_mae():
    loss = KernelWeightedMaeLoss(kernel_size=1, use_weights=False)
    target = torch.Tensor([
        [0, 1, 20, 5],
        [1, 2, 5, 10],
        [3, 15, 7, 1],
        [6, 3, 8, 1],
    ]).view((1, 1, 4, 4))
    prediction = torch.Tensor([
        [0, 10, 0, 5],
        [0, 2, 5, 20],
        [5, 5, 3, 1],
        [5, 3, 6, 8],
    ]).view((1, 1, 4, 4))

    predicted_loss = loss.spatial_loss(prediction, target, torch.ones_like(target))
    expected = torch.Tensor([
        [0., 5., 1., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 0., 1., 0.],
    ]).view((1, 1, 4, 4))

    assert torch.all(expected == predicted_loss)


def test_kernel_weighted_mae():
    loss = KernelWeightedMaeLoss(kernel_size=1, use_weights=True)
    target = torch.Tensor([
        [0, 1, 20, 5],
        [1, 2, 5, 10],
        [3, 15, 7, 1],
        [6, 3, 8, 1],
    ]).view((1, 1, 4, 4))
    prediction = torch.Tensor([
        [0, 10, 0, 5],
        [0, 2, 5, 20],
        [5, 5, 3, 1],
        [5, 3, 6, 8],
    ]).view((1, 1, 4, 4))

    predicted_loss = loss.spatial_loss(prediction, target, torch.ones_like(target))
    # [1, 2, 5, 10, 30]
    expected = torch.Tensor([
        [0., 9., 1., 0.],
        [0., 0., 0., 0.],
        [4., 0., 0., 0.],
        [4., 0., 5., 0.],
    ]).view((1, 1, 4, 4))
    assert torch.all(expected == predicted_loss)
