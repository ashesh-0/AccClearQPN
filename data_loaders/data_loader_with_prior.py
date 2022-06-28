import numpy as np

from core.enum import DataType
from data_loaders.data_loader_all_loaded import DataLoaderAllLoaded


class DataLoaderWithPrior(DataLoaderAllLoaded):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{k: v for k, v in kwargs.items() if k not in ['prior_dtype']})
        self._prior_dtype = kwargs['prior_dtype']

    def _get_2D_prior_inp(self, index):
        e_idx = self._input_end(index) - 1
        data = None
        rain = self._prior_dtype & DataType.Rain
        raindiff = self._prior_dtype & DataType.RainDiff
        if rain * raindiff:
            data = self._get_rain_data_from_range([e_idx - 1, e_idx])
            data[0] = data[1] - data[0]

        elif raindiff:
            data = self._get_rain_data_from_range([e_idx - 1, e_idx])
            data[0] = data[1] - data[0]
            data = data[:1]

        elif rain:
            data = self._get_rain_data_from_range([e_idx])

        return data

    def _get_hour_prior(self, index):
        time_data = [self._time[k] for k in self._input_range(index)]
        hour, month = zip(*[(tm.hour, tm.month) for tm in time_data])
        hour = np.mean(hour, dtype=np.float32)
        return hour

    def _get_month_prior(self, index):
        time_data = [self._time[k] for k in self._input_range(index)]
        hour, month = zip(*[(tm.hour, tm.month) for tm in time_data])
        month = np.mean(month, dtype=np.float32)
        return month

    def _get_1D_prior_inp(self, index):
        """
        There are some factors which can form a prior of the rain pattern.
        1. Time of Day (hour)
        2. Time of Year (month)
        """
        output = []
        if self._prior_dtype & DataType.Hour:
            output.append(self._get_hour_prior(index))
        if self._prior_dtype & DataType.Month:
            output.append(self._get_month_prior(index))

        if len(output) == 0:
            return None
        return np.array(output, dtype=np.float32)

    def _get_prior_inp(self, index):
        inp1D = self._get_1D_prior_inp(index)
        inp2D = self._get_2D_prior_inp(index)
        return (inp1D, inp2D)

    def __getitem__(self, input_index):
        output = super().__getitem__(input_index)
        index = self._get_internal_index(input_index)
        prior = self._get_prior_inp(index)
        has1D = self._prior_dtype & (DataType.Month + DataType.Hour)
        has2D = self._prior_dtype & (DataType.Rain + DataType.RainDiff)
        if not has2D:
            assert prior[1] is None
            prior = (prior[0], )
        elif not has1D:
            assert prior[0] is None
            prior = (prior[1], )

        return (*output, *prior)
