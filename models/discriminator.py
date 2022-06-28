import numpy as np
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_shape, downsample=5):
        super().__init__()
        self._dsz = downsample
        self._downsample = nn.MaxPool2d(self._dsz, stride=self._dsz)
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape) / (self._dsz**2)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        print(f'[{self.__class__.__name__}] Downsample:{self._dsz}')

    def forward(self, img):
        img = self._downsample(img)
        seq, batch, height, width = img.shape
        img_flat = img.view(seq * batch, -1)
        validity = self.model(img_flat)
        validity = validity.view((seq, batch, 1))
        return validity


class Discriminator2(nn.Module):
    def __init__(self, img_shape, downsample=5):
        super().__init__()
        self._dsz = downsample
        lc = 32
        self.model2D = nn.Sequential(
            nn.Conv2d(1, lc, 5, stride=self._dsz),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(lc, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.model1D = nn.Sequential(
            nn.Linear(int(np.prod(img_shape) / (self._dsz**2)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        print(f'[{self.__class__.__name__}] Downsample:{self._dsz}')

    def forward(self, img):
        seq, batch, height, width = img.shape
        img = img.reshape((seq * batch, 1, height, width))
        emb = self.model2D(img)
        emb_flat = emb.view(seq * batch, -1)
        validity = self.model1D(emb_flat)
        validity = validity.view((seq, batch, 1))
        return validity


class Squish2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        seq_batch, ch, height, width = inp.shape
        assert height == 1 and width == 1
        return inp.view((seq_batch, ch))


class Discriminator3(nn.Module):
    def __init__(self, img_shape, downsample=5):
        super().__init__()
        self._dsz = downsample

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveMaxPool2d(1),
            Squish2D(),
            nn.Linear(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        print(f'[{self.__class__.__name__}] Downsample:{self._dsz}')

    def forward(self, img):
        seq, batch, height, width = img.shape
        img = img.reshape((seq * batch, 1, height, width))
        validity = self.model(img)
        validity = validity.view((seq, batch, 1))
        return validity


class Discriminator4(nn.Module):
    def __init__(self, img_shape, downsample=5):
        super().__init__()
        self._dsz = downsample

        self.model = nn.Sequential(
            nn.MaxPool2d(self._dsz, stride=self._dsz),
            nn.Conv2d(1, 16, 5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveMaxPool2d(1),
            Squish2D(),
            nn.Linear(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        print(f'[{self.__class__.__name__}] Downsample:{self._dsz}')

    def forward(self, img):
        seq, batch, height, width = img.shape
        img = img.reshape((seq * batch, 1, height, width))
        validity = self.model(img)
        validity = validity.view((seq, batch, 1))
        return validity
