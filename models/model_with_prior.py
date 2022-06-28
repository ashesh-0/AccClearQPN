import os

import numpy as np
import torch
import torch.nn as nn

from core.enum import DataType
from models.model import EF


class Prior2DTo2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, final_kernel_size=5):
        super().__init__()
        self._n = in_channels
        self._k = final_kernel_size
        assert self._k % 2 == 1
        self._conv = nn.Conv2d(
            in_channels=self._n, out_channels=out_channels, kernel_size=self._k, padding=self._k // 2)
        self._conv.bias.data.fill_(2)

    def forward(self, input):
        output = self._conv(input)
        return output


class PriorScalarTo2D(nn.Module):
    def __init__(self, image_shape, num_scalar_input, final_kernel_size=5):
        super().__init__()
        self._n = num_scalar_input
        self._shape = image_shape
        self._k = final_kernel_size
        self._model = nn.Linear(self._n, np.prod(self._shape))
        self._conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self._k, padding=self._k // 2)
        self._conv.bias.data.fill_(2)

    def forward(self, input):
        batch, nchan = input.shape
        output = self._model(input)
        output = nn.LeakyReLU()(output)
        output = output.view((batch, 1, *self._shape))
        output = self._conv(output)
        return output


class MultiModalPriorModel(nn.Module):
    def __init__(self, image_shape, dtype, out_channels=1):
        super().__init__()

        DataType.print(dtype, prefix='Prior')
        self._n1D = DataType.count1D(dtype)
        self._n2D = DataType.count2D(dtype)

        self._2D_out = 16

        self._shape = image_shape
        self._linear_model = None
        self._2D_models = None
        if self._n1D:
            self._linear_model = PriorScalarTo2D(image_shape, self._n1D)
        if self._n2D:
            self._2D_models = nn.ModuleList([Prior2DTo2D(out_channels=self._2D_out) for _ in range(self._n2D)])
        in1D = int(self._n1D > 0)
        in2D = self._n2D * self._2D_out
        self._head = nn.Conv2d(in_channels=in1D + in2D, out_channels=out_channels, kernel_size=1)
        self._head.bias.data.fill_(2)

    def forward1D(self, input_1D):
        return [self._linear_model(input_1D)]

    def forward2D(self, input_2D):
        features = []
        for i in range(self._n2D):
            inp = input_2D[:, i:i + 1]
            features.append(self._2D_models[i](inp))
        return features

    def forward_both(self, input_1D, input_2D):
        features = self.forward1D(input_1D)
        features += self.forward2D(input_2D)
        return features

    def forward(self, *args):
        if self._n1D == 0:
            features = self.forward2D(*args)
        elif self._n2D == 0:
            features = self.forward1D(*args)
        else:
            features = self.forward_both(*args)

        if len(features) == 1:
            output = features[0]
        else:
            output = torch.cat(features, dim=1)

        output = self._head(output)
        return nn.Sigmoid()(output)


class PriorModel(nn.Module):
    def __init__(self, image_shape, num_scalar_input):
        super().__init__()
        self._n = num_scalar_input
        self._shape = image_shape
        self._model = PriorScalarTo2D(image_shape, num_scalar_input)

    def forward(self, input):
        batch, nchan = input.shape
        output = self._model(input)
        return nn.Sigmoid()(output)


class ModelWithPrior(EF):
    def __init__(self,
                 encoder,
                 forecaster,
                 checkpoint_prefix,
                 loss_kwargs,
                 image_shape,
                 prior_dtype,
                 residual=False,
                 checkpoint_directory=os.path.expanduser('~/')):
        super().__init__(
            encoder,
            forecaster,
            checkpoint_prefix,
            loss_kwargs,
            residual=residual,
            checkpoint_directory=checkpoint_directory,
        )
        self._prior_model = MultiModalPriorModel(image_shape, prior_dtype, out_channels=1)

    def training_step(self, batch, batch_idx):
        assert self._residual is False
        train_data, train_label, train_mask = batch[:3]
        train_prior = batch[3:]
        output = self(train_data)
        prior = self._prior_model(*train_prior)
        # batch,sequence,h,w => sequence,batch,h,w
        prior = prior.permute(1, 0, 2, 3)
        output = output * prior
        loss = self._criterion(output, train_label, train_mask)
        N = train_data.shape[0]
        self._avg_loss.add(loss.item() * N, N)
        self.log('train_loss', self._avg_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        assert self._residual is False
        val_data, val_label, val_mask = batch[:3]
        val_prior = batch[3:]
        output = self(val_data)
        prior = self._prior_model(*val_prior)
        # batch,sequence,h,w => sequence,batch,h,w
        prior = prior.permute(1, 0, 2, 3)
        output = output * prior
        loss = self._criterion(output, val_label, val_mask)
        return {'val_loss': loss, 'N': val_label.shape[0]}

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output
