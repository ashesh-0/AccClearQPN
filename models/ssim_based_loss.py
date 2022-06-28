import torch
import torch.nn as nn

from pytorch_msssim import ssim


def normalize(prediction, target):
    N = target.shape[0]
    eps = 1e-2
    max_val = torch.max(target.view(N, -1), dim=1)[0]
    max_val[max_val == 0] = eps

    max_val = max_val.view(N, 1, 1, 1)
    target = target / max_val
    prediction = prediction / max_val
    return prediction, target


def ssim_loss(prediction, target):
    ssim_err = 1 - ssim(prediction, target, data_range=1.0, size_average=True)
    return ssim_err


class SSIMBasedLoss(nn.Module):
    def __init__(self, mae_w=0.1, ssim_w=0.02, normalize_ssim=False):
        super().__init__()
        self._mae_w = mae_w
        self._ssim_w = ssim_w
        self._normalize_ssim = normalize_ssim
        self._mse_w = 1 - (self._mae_w + self._ssim_w)
        print(f'[{self.__class__.__name__}] MAE_W:{self._mae_w} MSE_W:{self._mse_w} SSIM_W:{self._ssim_w}')

    def forward(self, prediction, target, mask):
        target = target * mask
        prediction = prediction.permute(1, 0, 2, 3).clone()
        prediction = prediction * mask
        mse_l = nn.MSELoss()(prediction, target)
        mae_l = nn.L1Loss()(prediction, target)
        if self._normalize_ssim:
            prediction, target = normalize(prediction, target)
        ssim_l = ssim_loss(prediction, target)
        return {
            'mse': mse_l,
            'mae': mae_l,
            'ssim': ssim_l,
            'total': mse_l * self._mse_w + mae_l * self._mae_w + ssim_l * self._ssim_w,
        }


class NormalizedSSIMBasedLoss(SSIMBasedLoss):
    def forward(self, prediction, target, mask):
        target = target * mask
        prediction = prediction.permute(1, 0, 2, 3).clone()
        prediction = prediction * mask
        prediction, target = normalize(prediction, target)

        mse_l = nn.MSELoss()(prediction, target)
        mae_l = nn.L1Loss()(prediction, target)
        ssim_l = ssim_loss(prediction, target)
        return {
            'mse': mse_l,
            'mae': mae_l,
            'ssim': ssim_l,
            'total': mse_l * self._mse_w + mae_l * self._mae_w + ssim_l * self._ssim_w,
        }

        return super().forward(prediction, target, mask)


if __name__ == '__main__':
    tar = 100 * torch.rand(16, 3, 12, 12)
    pred = 100 * torch.rand(16, 3, 12, 12)
    print(ssim_loss(pred, tar))
    print(ssim_loss(pred, pred))
