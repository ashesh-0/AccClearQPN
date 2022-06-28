import torch
import torch.nn as nn

from models.adverserial_model import AdvModel
from models.balanced_adverserial_model import BalAdvModel


class BiasModule(nn.Module):
    def __init__(self, in_channels, img_shape):
        super().__init__()
        self._img_shape = img_shape
        self._learnable_var = nn.Parameter(torch.ones((1, 1, 1, *img_shape), requires_grad=True, device=0))
        assert in_channels > 1
        self._fixed = torch.ones((1, 1, in_channels - 1, *img_shape), device=0)
        print(f'[{self.__class__.__name__}]')

    def forward(self, input):
        radar_bias = torch.cat([self._learnable_var, self._fixed], dim=2)
        return input * radar_bias


class BiasModule2(nn.Module):
    def __init__(self, in_channels, img_shape):
        super().__init__()
        self._img_shape = img_shape
        self._learnable_var = nn.Parameter(torch.zeros((1, 1, 1, *img_shape), requires_grad=True, device=0))
        assert in_channels > 1
        self._fixed = torch.zeros((1, 1, in_channels - 1, *img_shape), device=0)
        print(f'[{self.__class__.__name__}]')

    def forward(self, input):
        radar_bias = torch.cat([self._learnable_var, self._fixed], dim=2)
        return input + radar_bias


class BiasModule3(nn.Module):
    def __init__(self, in_channels, img_shape):
        super().__init__()
        self._img_shape = img_shape
        self._learnable_W = nn.Parameter(torch.ones((1, 1, 1, *img_shape), requires_grad=True, device=0))
        self._learnable_b = nn.Parameter(torch.zeros((1, 1, 1, *img_shape), requires_grad=True, device=0))

        assert in_channels > 1
        self._fixed_W = torch.ones((1, 1, in_channels - 1, *img_shape), device=0)
        self._fixed_b = torch.zeros((1, 1, in_channels - 1, *img_shape), device=0)
        print(f'[{self.__class__.__name__}]')

    def forward(self, input):
        W = torch.cat([self._learnable_W, self._fixed_W], dim=2)
        b = torch.cat([self._learnable_b, self._fixed_b], dim=2)
        return input * W + b


class BalAdvModelWPrior(BalAdvModel):
    def __init__(self, *args, in_channels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bias = BiasModule3(in_channels, self._img_shape)

    def configure_optimizers(self):
        gen_params = list(self.G.parameters()) + list(self._bias.parameters())
        opt_g = torch.optim.Adam(gen_params, lr=self.lr)

        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        return opt_g, opt_d

    def forward(self, input):
        input = self._bias(input)
        # radar is 0th channel. 0th channel is learnable
        return super().forward(input)


class AdvModelWPrior(AdvModel):
    def __init__(self, *args, in_channels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bias = BiasModule(in_channels, self._img_shape)

    def configure_optimizers(self):
        gen_params = list(self.G.parameters()) + list(self._bias.parameters())
        opt_g = torch.optim.Adam(gen_params, lr=self.lr)

        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        return opt_g, opt_d

    def forward(self, input):
        input = self._bias(input)
        # radar is 0th channel. 0th channel is learnable
        return super().forward(input)
