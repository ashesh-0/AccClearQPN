import torch
import torch.nn as nn

from core.constants import BALANCING_WEIGHTS
from core.loss_type import BlockAggregationMode, LossType
from models.ssim_based_loss import NormalizedSSIMBasedLoss, SSIMBasedLoss


def get_criterion(loss_kwargs):
    loss_type = loss_kwargs['type']
    assert isinstance(loss_type, int)
    if loss_type == LossType.BlockWeightedMAE:
        criterion = BlockWeightedMaeLoss(
            loss_kwargs['kernel_size'],
            aggregation_mode=loss_kwargs['aggregation_mode'],
        )
    elif loss_type == LossType.WeightedMAE:
        criterion = WeightedMaeLoss()
    elif loss_type == LossType.BlockWeightedMAEDiversify:
        criterion = BlockWeightedMaeLossDiversify(
            loss_kwargs['kernel_size'],
            aggregation_mode=loss_kwargs['aggregation_mode'],
        )
    elif loss_type == LossType.WeightedAbsoluteMAE:
        criterion = WeightedAbsoluteMaeLoss()
    elif loss_type == LossType.MAE:
        criterion = MaeLoss()
    elif loss_type == LossType.ReluWeightedMaeLoss:
        criterion = ReluWeightedMaeLoss()
    elif loss_type == LossType.BalancedWeightedMaeLoss:
        criterion = BalancedWeightedMaeLoss(w=loss_kwargs['w'])
    elif loss_type == LossType.WeightedMaeLossDiversify:
        criterion = WeightedMaeLossDiversify(w=loss_kwargs['w'])
    elif loss_type == LossType.WeightedMAEWithBuffer:
        criterion = WeightedMaeLoss(boundary_bbox=loss_kwargs['padding_bbox'])
    elif loss_type == LossType.KernelWeightedMAE:
        criterion = KernelWeightedMaeLoss(kernel_size=loss_kwargs['kernel_size'])
    elif loss_type == LossType.WeightedMAEandMSE:
        criterion = WeightedMAEandMSE()
    elif loss_type == LossType.WeightedMSE:
        criterion = WeightedMSE()
    elif loss_type == LossType.ClassificationBCE:
        criterion = BCELoss()
    elif loss_type == LossType.SSIMBasedLoss:
        criterion = SSIMBasedLoss(mae_w=loss_kwargs['mae_w'], ssim_w=loss_kwargs['ssim_w'])
    elif loss_type == LossType.NormalizedSSIMBasedLoss:
        criterion = NormalizedSSIMBasedLoss(mae_w=loss_kwargs['mae_w'], ssim_w=loss_kwargs['ssim_w'])
    return criterion


def ignore_boundary(bbox, predicted, target, mask):
    xl, xr, yl, yr = bbox
    Nx, Ny = predicted.shape[-2:]
    predicted = predicted[..., xl:Nx - xr, yl:Ny - yr]
    target = target[..., xl:Nx - xr, yl:Ny - yr]
    mask = mask[..., xl:Nx - xr, yl:Ny - yr]
    return predicted, target, mask


def compute_weights(target, mask):
    weights = torch.ones_like(mask) * BALANCING_WEIGHTS[0]
    for i, threshold in enumerate(BALANCING_WEIGHTS):
        if i == 0:
            continue
        weights = weights + (BALANCING_WEIGHTS[i] - BALANCING_WEIGHTS[i - 1]) * (target >= threshold).float()
    weights = weights * mask.float()
    return weights


class BCELoss(nn.Module):
    def forward(self, predicted, target, mask):
        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)
        # target = target.reshape(batch_size, -1)
        # predicted = predicted.view(batch_size, -1)

        loss = nn.BCELoss()(predicted, target)
        return loss


class MaeLoss(nn.Module):
    def forward(self, predicted, target, mask):
        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)
        return torch.mean((torch.abs((predicted - target))))


class WeightedMaeLossPixel(nn.Module):
    def __init__(self, boundary_bbox=None):
        super().__init__()
        self._bbox = boundary_bbox
        if self._bbox is not None:
            print(f'[{self.__class__.__name__}] Bbox:{self._bbox}')

    def forward(self, predicted, target, mask):
        if self._bbox is not None:
            predicted, target, mask = ignore_boundary(self._bbox, predicted, target, mask)
        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)

        weights = compute_weights(target, mask)
        return weights * (torch.abs((predicted - target)))


class WeightedMaeLoss(WeightedMaeLossPixel):
    def forward(self, predicted, target, mask):
        return torch.mean(super().forward(predicted, target, mask))


class WeightedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        print(f'[{self.__class__.__name__}]')

    def forward(self, predicted, target, mask):

        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)

        weights = compute_weights(target, mask)
        mse = torch.mean(weights * (torch.square((predicted - target))))
        return mse


class WeightedMAEandMSE(nn.Module):
    def __init__(self):
        super().__init__()
        print(f'[{self.__class__.__name__}]')

    def forward(self, predicted, target, mask):

        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)

        weights = compute_weights(target, mask)
        mae = torch.mean(weights * (torch.abs((predicted - target))))
        mse = torch.mean(weights * (torch.square((predicted - target))))
        return mae + mse


class WeightedMAEandMSE2(nn.Module):
    def __init__(self):
        super().__init__()
        print(f'[{self.__class__.__name__}]')

    def forward(self, predicted, target, mask):

        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)

        weights = compute_weights(target, mask)
        mae = torch.sum(weights * (torch.abs((predicted - target))), (2, 3))
        mse = torch.mean(weights * (torch.square((predicted - target))), (2, 3))
        return torch.mean(mae + mse)


class KernelWeightedMaeLoss(nn.Module):
    def __init__(self, kernel_size=1, use_weights=True):
        super().__init__()
        self._ksz = kernel_size
        assert isinstance(use_weights, bool)
        self._use_weights = use_weights
        print(f'[{self.__class__.__name__} K:{self._ksz} Weighted:{self._use_weights}')

    def spatial_loss(self, predicted, target, mask):
        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)
        if self._use_weights:
            weights = compute_weights(target, mask)
        else:
            weights = torch.ones_like(target)
        loss2D = torch.abs((target - predicted) * weights)
        for x in range(-1 * self._ksz, self._ksz + 1):
            p_sx = max(0, x)
            p_ex = min(height, height + x)
            t_sx = max(0, -x)
            t_ex = min(height, height - x)
            # print('X', p_sx, p_ex, t_sx, t_ex)
            for y in range(-1 * self._ksz, self._ksz + 1):
                p_sy = max(0, y)
                p_ey = min(width, width + y)
                t_sy = max(0, -y)
                t_ey = min(width, width - y)
                # print('Y', p_sy, p_ey, t_sy, t_ey)
                target_shifted = target[..., t_sx:t_ex, t_sy:t_ey]
                weights_shifted = weights[..., t_sx:t_ex, t_sy:t_ey]
                pred_shifted = predicted[..., p_sx:p_ex, p_sy:p_ey]

                loss = torch.abs((target_shifted - pred_shifted) * weights_shifted)
                loss2D[..., t_sx:t_ex, t_sy:t_ey] = torch.min(loss, loss2D[..., t_sx:t_ex, t_sy:t_ey].clone())

        return loss2D

    def forward(self, predicted, target, mask):
        return torch.mean(self.spatial_loss(predicted, target, mask))


class ReluWeightedMaeLoss(WeightedMaeLoss):
    def forward(self, predicted, target, mask):
        predicted = nn.ReLU()(predicted)
        return super().forward(predicted, target, mask)


class ZeroRainLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print(f'[{self.__class__.__name__}]')

    def forward(self, predicted, target, mask):
        predicted = predicted.permute(1, 0, 2, 3)
        diff = (torch.abs((predicted - target)))
        z_loss = 0
        z_fraction = torch.mean(1 - mask)
        if z_fraction > 0:
            z_loss = torch.mean((1 - mask) * diff) / z_fraction
        else:
            z_loss = 0 * torch.mean((1 - mask) * diff)
        return z_loss


class BalancedWeightedMaeLoss(nn.Module):
    def __init__(self, w=0.5):
        super().__init__()
        self._w = w
        assert self._w > 0
        assert self._w <= 1
        print(f'[{self.__class__.__name__}] w:{self._w}')

    def forward(self, predicted, target, mask):
        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)

        weights = compute_weights(target, mask)
        diff = (torch.abs((predicted - target)))
        nz_loss = torch.mean(weights * diff)
        z_loss = 0
        z_fraction = torch.mean(1 - mask)
        if z_fraction > 0:
            z_loss = torch.mean((1 - mask) * diff) / z_fraction
        return nz_loss + ((1 - self._w) / self._w) * z_loss


class WeightedAbsoluteMaeLoss(nn.Module):
    def forward(self, predicted, target, mask):
        seq_len, batch_size, height, width = predicted.shape
        predicted = predicted.permute(1, 0, 2, 3)
        weights = torch.abs(target)
        return torch.mean(weights * (torch.abs((predicted - target))))


class BlockWeightedMaeLoss(WeightedMaeLoss):
    def __init__(self, kernel_size, aggregation_mode=BlockAggregationMode.MAX, stride=1):
        super().__init__()
        assert stride == 1, 'Please update the padding before changing the stride'
        self._amode = aggregation_mode
        lpad = (kernel_size - 1) // 2
        rpad = kernel_size - 1 - lpad
        padding = nn.ZeroPad2d((lpad, rpad, lpad, rpad))
        if self._amode == BlockAggregationMode.MAX:
            pool = nn.MaxPool2d(kernel_size, stride=stride)
        elif self._amode == BlockAggregationMode.MEAN:
            pool = nn.AvgPool2d(kernel_size, stride=stride)
        self._pool = nn.Sequential(padding, pool)
        print(f'[{self.__class__.__name__}] Kernel:{kernel_size} Stride:{stride} '
              f'AggMode:{BlockAggregationMode.name(self._amode)}')

    def forward(self, input, target, mask):
        input = self._pool(input)
        target = self._pool(target)
        if self._amode == BlockAggregationMode.MAX:
            mask = self._pool(mask)
        return super().forward(input, target, mask)


def diversification_loss(prediction, target):
    N = target.shape[0]
    assert prediction.shape[0] == N

    stdev_pred = torch.std(prediction.reshape(N, -1), dim=1)
    stdev_tar = torch.std(target.reshape(N, -1), dim=1)
    stdev_loss = nn.MSELoss()(stdev_pred, stdev_tar)
    return stdev_loss


class BlockWeightedMaeLossDiversify(BlockWeightedMaeLoss):
    def __init__(self, kernel_size, aggregation_mode=BlockAggregationMode.MAX, stride=1, w=0.99):
        super().__init__(kernel_size, aggregation_mode=aggregation_mode, stride=stride)
        self._w = w
        print(f'[{self.__class__.__name__}] w:{self._w}')

    def forward(self, prediction, target, mask):
        loss = super().forward(prediction, target, mask)
        prediction = prediction.permute(1, 0, 2, 3)
        stdev_loss = diversification_loss(prediction, target)
        return self._w * loss + (1 - self._w) * (stdev_loss)


class WeightedMaeLossDiversify(WeightedMaeLoss):
    def __init__(self, w=0.99):
        super().__init__()
        self._w = w
        print(f'[{self.__class__.__name__}] w:{self._w}')

    def forward(self, prediction, target, mask):
        loss = super().forward(prediction, target, mask)
        # batch_size, seq_len,height, width
        prediction = prediction.permute(1, 0, 2, 3)
        stdev_loss = diversification_loss(prediction, target)
        return self._w * loss + (1 - self._w) * (stdev_loss)


class KernelShiftedPrediction(nn.Module):
    def __init__(self, kernel_size=1, use_weights=False):
        super().__init__()
        self._ksz = kernel_size
        assert isinstance(use_weights, bool)
        self._use_weights = use_weights
        print(f'[{self.__class__.__name__} K:{self._ksz} Weighted:{self._use_weights}')

    def forward(self, predicted, target, mask):
        seq_len, batch_size, height, width = predicted.shape
        # batch_size, seq_len,height, width
        predicted = predicted.permute(1, 0, 2, 3)
        if self._use_weights:
            weights = compute_weights(target, mask)
        else:
            weights = torch.ones_like(target)

        k_predicted = predicted.clone()
        loss2D = torch.abs((target - predicted) * weights)
        for x in range(-1 * self._ksz, self._ksz + 1):
            p_sx = max(0, x)
            p_ex = min(height, height + x)
            t_sx = max(0, -x)
            t_ex = min(height, height - x)
            # print('X', p_sx, p_ex, t_sx, t_ex)
            for y in range(-1 * self._ksz, self._ksz + 1):
                p_sy = max(0, y)
                p_ey = min(width, width + y)
                t_sy = max(0, -y)
                t_ey = min(width, width - y)
                # print('Y', p_sy, p_ey, t_sy, t_ey)
                target_shifted = target[..., t_sx:t_ex, t_sy:t_ey]
                weights_shifted = weights[..., t_sx:t_ex, t_sy:t_ey]
                pred_shifted = predicted[..., p_sx:p_ex, p_sy:p_ey]

                # loss = torch.abs((target_shifted - pred_shifted))
                loss = torch.abs((target_shifted - pred_shifted) * weights_shifted)
                mask = (loss < loss2D[..., t_sx:t_ex, t_sy:t_ey]).int()
                k_predicted[..., t_sx:t_ex,
                            t_sy:t_ey] = k_predicted[..., t_sx:t_ex, t_sy:t_ey] * (1 - mask) + (pred_shifted) * mask
                loss2D[..., t_sx:t_ex, t_sy:t_ey] = torch.min(loss, loss2D[..., t_sx:t_ex, t_sy:t_ey].clone())


        return k_predicted.permute(1, 0, 2, 3)
