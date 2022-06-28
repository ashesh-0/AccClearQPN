import os

from models.base_model import BaseModel


class EF(BaseModel):
    def __init__(self,
                 encoder,
                 forecaster,
                 checkpoint_prefix,
                 loss_kwargs,
                 residual=False,
                 checkpoint_directory=os.path.expanduser('~/')):
        super().__init__(loss_kwargs, checkpoint_prefix, checkpoint_directory)
        self._residual = residual
        self.encoder = encoder
        self.forecaster = forecaster
        self._residual_loss = loss_kwargs.get('residual_loss', False)

        print(f'[{self.__class__.__name__}] Residual:{self._residual}')

    def training_step(self, batch, batch_idx):
        if self._residual is False:
            return super().training_step(batch, batch_idx)

        train_data, train_label, train_mask, label_past = batch
        output = self(train_data)
        if self._residual_loss:
            loss = self._criterion(output, train_label, train_mask)
        else:
            loss = self._criterion(output + label_past.permute(1, 0, 2, 3), train_label + label_past, train_mask)

        N = train_data.shape[0]
        self._avg_loss.add(loss.item() * N, N)
        self.log('train_loss', self._avg_loss.get(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self._residual is False:
            return super().validation_step(batch, batch_idx)

        val_data, val_label, val_mask, label_past = batch
        output = self(val_data)
        if self._residual_loss:
            loss = self._criterion(output, val_label, val_mask)
        else:
            loss = self._criterion(output + label_past.permute(1, 0, 2, 3), val_label + label_past, val_mask)

        return {'val_loss': loss, 'N': val_label.shape[0]}

    def forward(self, input):
        state = self.encoder(input)
        output = self.forecaster(state)
        return output
