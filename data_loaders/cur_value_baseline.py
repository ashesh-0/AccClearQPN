"""
This is a baseline. Here, for predicting one hour in advance, we just predict the last hour's rain.
"""
from datetime import datetime

from core.constants import INPUT_LEN
from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded


class CurrentValueBaseLine(DataLoaderAllLoaded):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k != 'pred_avg_length'})
        self._pred_avg_len = kwargs['pred_avg_length']

    def __getitem__(self, index):
        output = super().__getitem__(index)
        prediction = self._get_prediction(index)
        return (*output, prediction)

    def _get_prediction(self, index):
        index = self._get_internal_index(index)
        return self._get_most_recent_target(index, tavg_len=self._pred_avg_len)


if __name__ == '__main__':
    start = datetime(2018, 1, 1)
    end = datetime(2018, 1, 31, 23, 50)
    target_len = 1
    baseline = CurrentValueBaseLine(start, end, INPUT_LEN, target_len)
    inp, tar, mask, prediction = baseline[len(baseline) - 1]
    print(prediction.mean(), tar.mean())
    print(prediction.max(), tar.max())
