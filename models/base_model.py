import os

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

from core.loss_type import LossType
from core.running_average import RunningAverage
from models.loss import get_criterion
from utils.performance_diagram import PerformanceDiagramStable


class BaseModel(LightningModule):
    def __init__(self, loss_kwargs, checkpoint_prefix, checkpoint_directory):
        super().__init__()
        assert isinstance(loss_kwargs, dict), f'loss_kwargs must be a dict. Passed value :{loss_kwargs}'
        self.lr = 1e-04
        self._loss_type = loss_kwargs['type']
        self._ckp_prefix = checkpoint_prefix
        self._ckp_dir = checkpoint_directory
        self._criterion = get_criterion(loss_kwargs)
        self._val_criterion = PerformanceDiagramStable()
        self._avg_loss = RunningAverage()
        print(f'[{self.__class__.__name__}] Ckp:{os.path.join(self._ckp_dir,self._ckp_prefix)} '
              f'Loss:{LossType.name(self._loss_type)}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        train_data, train_label, train_mask = batch
        output = self(train_data)
        loss = self._criterion(output, train_label, train_mask)

        N = train_data.shape[0]
        self._avg_loss.add(loss.item() * N, N)
        self.log('train_loss', self._avg_loss.get(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_epoch_end(self, *args):
        self._avg_loss.reset()

    def validation_step(self, batch, batch_idx):
        val_data, val_label, val_mask = batch
        output = self(val_data)
        loss = self._criterion(output, val_label, val_mask)

        if self._val_criterion is not None:
            self._val_criterion.compute(output.permute(1, 0, 2, 3), val_label)
        return {'val_loss': loss, 'N': val_label.shape[0]}

    def validation_epoch_end(self, outputs):
        val_loss_sum = 0
        N = 0
        for output in outputs:
            val_loss_sum += output['val_loss'] * output['N']
            # this may not have the entire batch. but we are still multiplying it by N
            N += output['N']

        val_loss_mean = val_loss_sum / N
        if self._val_criterion is not None:
            pdsr = self._val_criterion.get()['Dotmetric']
            self._val_criterion.reset()
            self.log('pdsr', pdsr)
        self.logger.experiment.add_scalar('Loss/val', val_loss_mean.item(), self.current_epoch)
        self.log('val_loss', val_loss_mean)

    def training_epoch_end(self, outputs):
        train_loss_mean = 0
        for output in outputs:
            train_loss_mean += output['loss']
        train_loss_mean /= len(outputs)
        self.logger.experiment.add_scalar('Loss/train', train_loss_mean.item(), self.current_epoch)

    def test_step(self, batch, batch_idx):
        loss = self.validation_step(batch, batch_idx)
        self.log('test_loss', loss['val_loss'])
        return {'test_loss': loss['val_loss'], 'N': loss['N']}

    def test_epoch_end(self, outputs):
        test_loss_sum = 0
        N = 0
        for output in outputs:
            test_loss_sum += output['test_loss'] * output['N']

        test_loss_mean = test_loss_sum / N
        self.log('test_loss', test_loss_mean)

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            filepath=os.path.join(self._ckp_dir, '_{epoch}_{val_loss:.3f}_{pdsr:.2f}'),
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=self._ckp_prefix)
