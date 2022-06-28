from models.adverserial_model import AdvModel


class BalAdvModel(AdvModel):
    def training_step(self, batch, batch_idx, optimizer_idx):
        loss = super().training_step(batch, batch_idx, optimizer_idx)
        # balance discriminator
        if optimizer_idx == 1:
            return loss * self._adv_w
        return loss
