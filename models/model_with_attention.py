import torch
import torch.nn as nn

from models.model import EF


class TargetAttentionLean(nn.Module):
    def __init__(self):
        super().__init__()

        self.atten1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, input, prediction):
        input = torch.sigmoid(input)
        h1 = torch.sigmoid(self.atten1(input))
        h2 = torch.sigmoid(self.atten1(h1))
        h3 = torch.sigmoid(self.atten1(h2))
        h1 = h1[None, :, 0, ...]
        h2 = h2[None, :, 0, ...]
        h3 = h3[None, :, 0, ...]

        attention = torch.cat([h1, h2, h3], dim=0)
        return prediction * attention


class EFAttentionModel(EF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = TargetAttentionLean()

    def forward(self, input):
        pred = super().forward(input)
        # NOTE most recent rain frame
        output = self.attention(input[:, -1, -1:], pred)
        return output
