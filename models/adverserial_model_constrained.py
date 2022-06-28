from collections import OrderedDict

import torch
import torch.nn as nn

from models.adverserial_model import AdvModel


class AdvConsModel(AdvModel):
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        opt_gd = torch.optim.Adam(self.forecaster.parameters())
        return opt_g, opt_d, opt_gd

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
