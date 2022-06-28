"""
A simple feed forward CNN model
"""
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

from models.base_model import BaseModel


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert kwargs.get('padding') is None
        lpad = (kernel_size - 1) // 2
        rpad = kernel_size - 1 - lpad
        self._pad = nn.ZeroPad2d((lpad, rpad, lpad, rpad))

    def forward(self, input):
        padded_inp = self._pad(input)
        return super().forward(padded_inp)


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        self._in = in_channel
        self._out = out_channel
        self._k = kernel_size
        self._model = nn.Sequential(
            Conv2dSamePadding(self._in, self._out, self._k),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(self._out),
        )

    def forward(self, input):
        return self._model(input)


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._b1 = Block(in_channels, 32, 5)
        self._b2 = Block(32, 64, 5)
        self._b3 = Block(64, 64, 5)
        self._b4 = Block(64, 64, 5)
        self._b5 = Block(64, 64, 5)
        self._b6 = Block(64, 64, 5)
        self._b7 = Block(64, 64, 5)
        self._model = nn.Sequential(self._b1, self._b2, self._b3, self._b4, self._b5, self._b6, self._b7)

    def forward(self, input):
        batch_size, seq_len, channel_count, h, w = input.shape
        input = input.view((-1, channel_count, h, w))
        out = checkpoint_sequential(self._model,7 , input)
        out = out.view(batch_size, seq_len, 64, h, w)
        out = torch.mean(out, dim=1)
        return out


class BaseLineModel(BaseModel):
    def __init__(
            self,
            loss_kwargs,
            checkpoint_prefix,
            in_channels=3,
            out_channels=3,
            checkpoint_directory=os.path.expanduser('~/'),
    ):
        super().__init__(loss_kwargs, checkpoint_prefix, checkpoint_directory)
        self._head = Conv2dSamePadding(64, out_channels, 5)
        self._f1 = FeatureExtractor(in_channels)

    def forward(self, input):
        batch_size, seq_len, channel_count, h, w = input.shape

        out = self._f1(input)
        out = self._head(out)
        # this is just for uniformity.
        out = out.permute(1, 0, 2, 3)
        return out
