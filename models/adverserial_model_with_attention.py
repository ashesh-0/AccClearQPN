from collections import OrderedDict

import torch
import torch.nn as nn

from core.running_average import RunningAverage
from models.adverserial_model import AdvModel
from models.balanced_adverserial_model import BalAdvModel
from models.loss import ZeroRainLoss


class TargetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.atten1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 1, 1))

        self.atten2 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 1, 1))

        self.atten3 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 1, 1))

        # Set bias to 1
        with torch.no_grad():
            list(self.atten1.modules())[-1].bias.fill_(2.)
            list(self.atten2.modules())[-1].bias.fill_(2.)
            list(self.atten3.modules())[-1].bias.fill_(2.)

    def forward(self, input, prediction):
        h1 = self.atten1(input)
        h2 = self.atten2(h1)
        h3 = self.atten3(h2)
        h1 = torch.sigmoid(h1)[None, :, 0, ...]
        h2 = torch.sigmoid(h2)[None, :, 0, ...]
        h3 = torch.sigmoid(h3)[None, :, 0, ...]
        # h1 = torch.sigmoid(h1)[None, :, 0, ...]
        # h2 = torch.sigmoid(h2)[None, :, 0, ...]
        # h3 = torch.sigmoid(h3)[None, :, 0, ...]

        attention = torch.cat([h1, h2, h3], dim=0)
        return prediction * attention


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


class BalAdvAttentionModel(BalAdvModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = TargetAttentionLean()

    def forward(self, input):
        pred = super().forward(input)
        # NOTE most recent rain frame
        output = self.attention(input[:, -1, -1:], pred)
        return output

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(list(self.G.parameters()) + list(self.attention.parameters()), lr=self.lr)
        opt_d = torch.optim.Adam(list(self.D.parameters()), lr=self.lr)
        return opt_g, opt_d


class BalAdvAttentionZeroLossModel(BalAdvModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = TargetAttention()
        self._zr_loss = ZeroRainLoss()
        self._zr_w = 0.005
        self._zr_avg_loss = RunningAverage()

    def forward(self, input):
        pred = super().forward(input)
        # NOTE most recent rain frame
        return self.attention(input[:, -1, -1:], pred)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(list(self.G.parameters()) + list(self.attention.parameters()), lr=self.lr)
        opt_d = torch.optim.Adam(list(self.D.parameters()) + list(self.attention.parameters()), lr=self.lr)
        opt_a = torch.optim.Adam(self.attention.parameters(), lr=self.lr)
        return opt_g, opt_d, opt_a

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx in [0, 1]:
            return super().training_step(batch, batch_idx, optimizer_idx)

        train_data, train_label, train_mask = batch
        N = train_data.shape[0]
        predicted_reconstruction = self(train_data)
        loss = self._zr_loss(predicted_reconstruction, train_label, train_mask)

        self._zr_avg_loss.add(loss.item() * N, N)
        self.log('ZL', self._zr_avg_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        return self._zr_w * loss

    def on_epoch_end(self, *args):
        super().on_epoch_end(*args)
        self._zr_avg_loss.reset()


class BalAdvAttention3OptModel(AdvModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = TargetAttentionLean()
        # self._label_smoothing_alpha = 0.1

    def forward(self, input):
        pred = super().forward(input)
        # NOTE most recent rain frame
        return self.attention(input[:, -1, -1:], pred)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(list(self.G.parameters()) + list(self.attention.parameters()), lr=self.lr)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        opt_gd = torch.optim.Adam(self.attention.parameters())
        return opt_g, opt_d, opt_gd

    def adversarial_loss_fn(self, y_hat, y, smoothing=True):
        uniq_y = torch.unique(y)
        assert len(uniq_y) == 1
        if smoothing:
            # one sided smoothing.
            y = y * (1 - self._label_smoothing_alpha) + (1 - y) * self._label_smoothing_alpha
        return nn.BCELoss()(y_hat, y)

    def wmae_loss(self, input_data, target_label, target_mask):
        predicted_reconstruction = self(input_data)
        recons_loss = self._criterion(predicted_reconstruction, target_label, target_mask)
        loss = recons_loss * (1 - self._adv_w)
        output = OrderedDict({'loss': loss, 'prediction': predicted_reconstruction})
        return output

    def gen_adv_loss(self, input_data, target_label, target_mask):
        predicted_reconstruction = self(input_data)
        # train generator
        N = self._target_len * input_data.size(0)
        valid = torch.ones(N, 1)
        valid = valid.type_as(input_data)

        # adversarial loss is binary cross-entropy
        # ReLU is used since generator fools discriminator with -ve values
        adv_loss = self.adversarial_loss_fn(
            self.D(nn.ReLU()(predicted_reconstruction)).view(N, 1), valid, smoothing=False)
        # tqdm_dict = {'adv_loss': adv_loss}
        loss = adv_loss * self._adv_w
        output = OrderedDict({'loss': loss, 'prediction': predicted_reconstruction})
        # import pdb
        # pdb.set_trace()
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):

        train_data, train_label, train_mask = batch
        N = train_data.shape[0]
        # Weighted MAE
        if optimizer_idx == 0:
            loss_dict = self.wmae_loss(train_data, train_label, train_mask)
            self._recon_loss.add(loss_dict['loss'].item() / (1 - self._adv_w) * N, N)
            self.log('GRecon', self._recon_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        # train discriminator
        elif optimizer_idx == 1:
            loss_dict = self.discriminator_loss(train_data, train_label, train_mask)
            self._D_loss.add(loss_dict['loss'].item() * N, N)
            self.log('D', self._D_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
            loss_dict['loss'] = loss_dict['loss'] * self._adv_w

        # Attention
        elif optimizer_idx == 2:
            loss_dict = self.gen_adv_loss(train_data, train_label, train_mask)
            self._GD_loss.add(loss_dict['loss'].item() / self._adv_w * N, N)
            self.log('GD', self._GD_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        return loss_dict['loss']
