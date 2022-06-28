import torch

from models.model import EF


class ClassifierModel(EF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._val_criterion = None

    def forward(self, input):
        prediction = super().forward(input)
        return torch.sigmoid(prediction)
