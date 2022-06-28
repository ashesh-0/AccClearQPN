import os
from collections import OrderedDict

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

from core.loss_type import LossType
from core.running_average import RunningAverage
from models.loss import WeightedMaeLoss, get_criterion
from utils.performance_diagram import PerformanceDiagramStable


class SSIMModel(LightningModule):
    def __init__(self, encoder, forecaster, target_len, loss_kwargs, checkpoint_prefix, checkpoint_directory):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        self.G = torch.nn.Sequential(self.encoder, self.forecaster)

        self._img_shape = (540, 420)
        self.lr = 1e-04
        self._loss_type = loss_kwargs['type']
        assert self._loss_type in [LossType.SSIMBasedLoss, LossType.NormalizedSSIMBasedLoss]

        self._ckp_prefix = checkpoint_prefix
        self._ckp_dir = checkpoint_directory
        self._target_len = target_len
        self._criterion = get_criterion(loss_kwargs)
        self._mae_loss = RunningAverage()
        self._mse_loss = RunningAverage()
        self._ssim_loss = RunningAverage()
        self._total_loss = RunningAverage()
        # self._train_D_stats = DiscriminatorStats()
        self._val_criterion = PerformanceDiagramStable()
        self._val_wmae_criterion = WeightedMaeLoss()

        print(f'[{self.__class__.__name__}] Ckp:{os.path.join(self._ckp_dir,self._ckp_prefix)} ')

    def configure_optimizers(self):
        return torch.optim.Adam(self.G.parameters(), lr=self.lr)

    def get_loss(self, input_data, target_label, target_mask):
        predicted_reconstruction = self(input_data)
        loss_dict = self._criterion(predicted_reconstruction, target_label, target_mask)
        # train generator
        tqdm_dict = {
            'mse': loss_dict['mse'],
            'mae': loss_dict['mae'],
            'ssim': loss_dict['ssim'],
            'total': loss_dict['total'],
        }
        loss = loss_dict['total']
        output = OrderedDict({'loss': loss, 'progress_bar': tqdm_dict, 'prediction': predicted_reconstruction})
        return output

    def training_step(self, batch, batch_idx):

        train_data, train_label, train_mask = batch
        N = train_data.shape[0]
        loss_dict = self.get_loss(train_data, train_label, train_mask)
        self._mae_loss.add(loss_dict['progress_bar']['mae'].item() * N, N)
        self._mse_loss.add(loss_dict['progress_bar']['mse'].item() * N, N)
        self._ssim_loss.add(loss_dict['progress_bar']['ssim'].item() * N, N)
        self._total_loss.add(loss_dict['progress_bar']['total'].item() * N, N)

        self.log('mae', self._mae_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('mse', self._mse_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('ssim', self._ssim_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('total', self._total_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        val_data, val_label, val_mask = batch
        # generator
        loss_dict = self.get_loss(val_data, val_label, val_mask)
        aligned_prediction = loss_dict['prediction'].permute(1, 0, 2, 3)
        self._val_criterion.compute(aligned_prediction, val_label)
        wmae = self._val_wmae_criterion(loss_dict['prediction'], val_label, val_mask)
        log_data = loss_dict.pop('progress_bar')
        return {'val_loss': log_data['mae'], 'wmae': wmae, 'N': val_label.shape[0]}

    def validation_epoch_end(self, outputs):
        val_loss_sum = 0
        N = 0
        for output in outputs:
            val_loss_sum += output['wmae'] * output['N']
            # this may not have the entire batch. but we are still multiplying it by N
            N += output['N']

        val_loss_mean = val_loss_sum / N
        self.logger.experiment.add_scalar('WMAE Loss/val', val_loss_mean.item(), self.current_epoch)
        self.log('val_loss', val_loss_mean)

        pdsr = self._val_criterion.get()['Dotmetric']
        self._val_criterion.reset()
        self.log('pdsr', pdsr)

    def on_epoch_end(self, *args):
        self._mae_loss.reset()
        self._mse_loss.reset()
        self._ssim_loss.reset()
        self._total_loss.reset()

    def forward(self, input):
        return self.G(input)

    def get_checkpoint_callback(self):
        return ModelCheckpoint(filepath=os.path.join(
            self._ckp_dir, '_{epoch}_{val_loss:.3f}_{pdsr:.2f}_{D_auc:.2f}_{D_pos_acc:.2f}_{D_neg_acc:.2f}'),
                               save_top_k=1,
                               verbose=True,
                               monitor='val_loss',
                               mode='min',
                               prefix=self._ckp_prefix)
